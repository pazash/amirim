import os
# 1. Disable the experimental bucket check that causes the warning
os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "false"
# 2. Force anonymous access globally
os.environ["GCSFS_TOKEN"] = "anon"
import xarray as xr
import pandas as pd
import numpy as np
import extremeweatherbench as ewb
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataloader import AuroraDataLoader
from src.forecaster import AuroraForecaster
from aurora import AuroraV1p5
import aiohttp


# --- 1. Helper Function: Convert Aurora Batch to Xarray Dataset ---
def aurora_batch_to_xarray(predicted_batch, init_time, lead_time_hours):
    """
    Converts a single PyTorch Aurora Batch object into an EWB-compatible xarray Dataset.
    """
    # Extract the surface temperature tensor and move to CPU/numpy
    # Assuming the variable is named "2t" and the shape is [lat, lon]
    t2m_tensor = predicted_batch.surf_vars["2t"].detach().cpu().numpy().squeeze()
    
    # Define the standard 0.25-degree grid
    lats = predicted_batch.metadata.lat
    lons = predicted_batch.metadata.lon
    
    # Create the xarray Dataset with exact EWB required dimensions
    ds = xr.Dataset(
        data_vars={
            # Keep the name '2t_aurora', we will map it during evaluation
            "2t_aurora": (["init_time", "lead_time", "latitude", "longitude"], 
                          t2m_tensor[np.newaxis, np.newaxis, :, :]),
        },
        coords={
            "init_time": [pd.to_datetime(init_time)],
            "lead_time": pd.to_timedelta([f"{lead_time_hours}h"]),
            "latitude": lats,
            "longitude": lons,
        }
    )
    return ds


# --- 2. Main Generation Loop ---
def generate_heatwave_forecasts(output_dir="ewb_forecasts/"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    data_pipeline = AuroraDataLoader(cache_dir=Path("weather_data"))
        
    aurora_model = AuroraV1p5()
    aurora_model.load_checkpoint("microsoft/aurora", "aurora-0.25-v1.5.ckpt", revision="main")
    
    forecaster = AuroraForecaster(model=aurora_model)
    all_cases = ewb.load_cases() 
    heatwave_cases = [case for case in all_cases if case.event_type == "heat_wave"]
    
    # We define the lead times we want to evaluate to the START of the event
    target_start_lead_times = [24, 72, 120] 
    
    for case in heatwave_cases:

        event_start = pd.to_datetime(case.start_date)
        event_end = pd.to_datetime(case.end_date)
        
        # Calculate the total duration of the event in hours
        event_duration_hours = int((event_end - event_start).total_seconds() / 3600)
        
        for start_lead in target_start_lead_times:
            # Initialize the model `start_lead` hours before the event begins
            init_time = event_start - pd.Timedelta(hours=start_lead)
            
            filename = Path(output_dir) / f"aurora_hw_forecast_{init_time.strftime('%Y%m%d')}_L{start_lead}_full_event.nc"
            
            if filename.exists():
                print(f"Skipping {case.title} at {start_lead}h start lead - File already exists")
                continue
                
            print(f"Processing: {case.title} | Init: {init_time} | Start Lead: {start_lead}h | Duration: {event_duration_hours}h")

            try:
                input_batch, _ = data_pipeline.get_batches(
                        init_time, 
                        history_steps=1, 
                        forecast_steps=0,
                        bbox=None
                    )
                
                # Calculate total steps needed: 
                # Steps to reach the start of the event + Steps to cover the duration of the event
                # Assuming Aurora steps in 6-hour increments
                total_hours_to_roll = start_lead + event_duration_hours
                total_steps_needed = int(total_hours_to_roll / 6)
                
                # predict_rollout returns a list of outputs for each step
                predicted_batches = forecaster.predict_rollout(
                    initial_batch=input_batch, 
                    steps=total_steps_needed,
                    fine_lead_times=[6.0]
                )
                
                # We only want to save the steps that fall WITHIN the event window.
                # Calculate the index where the event starts.
                start_index = int(start_lead / 6) - 1 # 0-indexed offset
                
                event_ds_list = []
                
                # Loop through only the predictions that fall inside the event window
                for i, batch in enumerate(predicted_batches[start_index:]):
                    # Calculate the specific lead time for this timestep
                    current_lead_time = start_lead + (i * 6)
                    
                    # Convert this single step to xarray
                    step_ds = aurora_batch_to_xarray(batch, init_time, current_lead_time)
                    event_ds_list.append(step_ds)
                    
                # Concatenate all the steps for this event into a single dataset along the lead_time dimension
                full_event_ds = xr.concat(event_ds_list, dim="lead_time")
                
                full_event_ds.to_netcdf(filename, engine='h5netcdf')
                
            except Exception as e:
                print(f"Failed on {init_time} for lead {start_lead}h: {e}")

class IntersectionOverUnion(ewb.ThresholdMetric):
    """
    Intersection Over Union (IOU) using a dynamically calculated 
    percentile threshold based on the target's distribution.
    """
    def __init__(self, percentile: float = 95.0, name: str = "Spatial_IOU", **kwargs):
        # We don't pass static thresholds to the parent anymore
        super().__init__(name=name, **kwargs)
        self.percentile = percentile

    def _compute_metric(self, forecast: xr.DataArray, target: xr.DataArray, **kwargs):
        
        # 1. Dynamically calculate the threshold based on the TRUE target
        # np.percentile calculates the value at the given percentile across the entire target array
        dynamic_threshold = float(np.percentile(target.values, self.percentile))
        
        # 2. Tell the transformed_contingency_manager to use this new dynamic threshold
        transformed = self.transformed_contingency_manager(
            forecast=forecast,
            target=target,
            forecast_threshold=dynamic_threshold,
            target_threshold=dynamic_threshold,
            preserve_dims=self.preserve_dims,
        )
        
        # 3. Calculate IOU
        counts = transformed.get_counts()
        tp = counts["tp_count"]
        fp = counts["fp_count"]
        fn = counts["fn_count"]
        
        union = tp + fp + fn
        return xr.where(union > 0, tp / union, 0.0)


class PeakMagnitudeBias(ewb.BaseMetric):
    """
    Peak Magnitude Bias evaluated ONLY on grid points where the true target 
    exceeds a dynamically calculated percentile threshold.
    """
    def __init__(self, percentile: float = 95.0, name: str = "Peak_Bias", **kwargs):
        super().__init__(name=name, **kwargs)
        self.percentile = percentile

    def _compute_metric(self, forecast: xr.DataArray, target: xr.DataArray, **kwargs) -> xr.DataArray:
        
        # 1. Dynamically calculate the threshold based on the TRUE target
        dynamic_threshold = float(np.percentile(target.values, self.percentile))
        
        # 2. Create the extreme mask
        extreme_mask = target >= dynamic_threshold
        
        # 3. Calculate Bias ONLY on the extreme grid points
        forecast_extremes = forecast.where(extreme_mask)
        target_extremes = target.where(extreme_mask)
        
        bias = forecast_extremes - target_extremes
        
        return bias.mean(
            dim=[d for d in bias.dims if d != self.preserve_dims]
        )

    
def evaluating():
    forecast_path = Path("ewb_forecasts")
    print(f"Starting Extreme Evaluator... Reading forecasts from {forecast_path}")
    
    # Find all NetCDF forecast files in the directory
    nc_files = sorted(list(forecast_path.glob("*.nc")))

    # xr.open_mfdataset automatically concatenates multiple single-timestep 
    # NetCDF files along the 'init_time' dimension.
    # compat="override" ensures coordinate metadata clashes don't throw errors.
    ds = xr.open_mfdataset(
        nc_files, 
        combine="nested", 
        concat_dim="init_time",
        compat="override",
        coords="minimal"
    )

    my_forecast = ewb.forecasts.XarrayForecast(
        ds=ds,
        name="aurora1.5",
        variables=["surface_air_temperature"],
        variable_mapping={
            "2t_aurora": "surface_air_temperature"
        },
    )

    hres_forecast = ewb.ZarrForecast(
        source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
        name="HRES",
        variable_mapping=ewb.HRES_metadata_variable_mapping,
        storage_options={"remote_options": {"anon": True}}, 
    )

    era5_heatwave_target = ewb.ERA5(
        variables=["surface_air_temperature"],
        storage_options={
            "token": "anon",
            "client_kwargs": {
                # Increase the timeout to 1 hour (3600 seconds) to prevent drops
                "timeout": aiohttp.ClientTimeout(total=3600, connect=60, sock_read=3600)
            }
        }
    )

    spatial_iou = IntersectionOverUnion(
        percentile=95.0,  # Focuses on the top 5% of extreme heat
        forecast_variable="surface_air_temperature",
        target_variable="surface_air_temperature"
    )
    
    peak_bias = PeakMagnitudeBias(
        percentile=95.0,
        forecast_variable="surface_air_temperature",
        target_variable="surface_air_temperature"
    )

    # Standard RMSE remains the same
    standard_rmse = ewb.metrics.RootMeanSquaredError(
        forecast_variable="surface_air_temperature",
        target_variable="surface_air_temperature"
    )

    heatwave_evaluation_list = [
        # 1. Evaluate your local Aurora predictions
        ewb.EvaluationObject(
            event_type="heat_wave",
            metric_list=[spatial_iou, peak_bias, standard_rmse],
            target=era5_heatwave_target,
            forecast=my_forecast,
        ),
        # 2. Evaluate the HRES physical model baseline
        ewb.EvaluationObject(
            event_type="heat_wave",
            metric_list=[spatial_iou, peak_bias, standard_rmse],
            target=era5_heatwave_target,
            forecast=hres_forecast,
        ),
    ]

    case_yaml = ewb.load_cases()

    ewb_instance = ewb.evaluation(
        case_metadata=case_yaml,
        evaluation_objects=heatwave_evaluation_list,
    )

    print("Running Extreme Weather Bench evaluations...")
    outputs = ewb_instance.run_evaluation()
    outputs.to_csv('evaluations.csv')
    print("Evaluations saved to evaluations.csv")


generate_heatwave_forecasts()
evaluating()