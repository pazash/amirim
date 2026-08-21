import os
import re
import warnings
import logging
from pathlib import Path
import sys
import gc
import torch

import aiohttp
import numpy as np
import pandas as pd
import xarray as xr
import extremeweatherbench as ewb

# Suppress warnings that aren't useful during evaluation
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Use anonymous access for GCS (ERA5, HRES)
os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "false"
os.environ["GCSFS_TOKEN"] = "anon"

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dataloader import AuroraDataLoader
from src.forecaster import AuroraForecaster
from aurora import AuroraV1p5

# =============================================================================
# Section 1: Pure Metric Functions
# =============================================================================

def peak_amplitude_error(forecast_2d: np.ndarray, target_2d: np.ndarray) -> float:
    if np.isnan(target_2d).all() or np.isnan(forecast_2d).all():
        return np.nan
    return float(np.nanmax(forecast_2d) - np.nanmax(target_2d))

def conditional_bias_extremes(forecast_2d: np.ndarray, target_2d: np.ndarray, percentile: float = 90.0) -> float:
    if np.isnan(target_2d).all():
        return np.nan
    threshold = np.nanpercentile(target_2d, percentile)
    mask = target_2d >= threshold
    if not np.any(mask):
        return np.nan
    return float(np.nanmean(forecast_2d[mask] - target_2d[mask]))

def rmse(forecast_2d: np.ndarray, target_2d: np.ndarray) -> float:
    if np.isnan(target_2d).all() or np.isnan(forecast_2d).all():
        return np.nan
    diff = forecast_2d - target_2d
    return float(np.sqrt(np.nanmean(diff**2)))

def intersection_over_union(forecast_2d: np.ndarray, target_2d: np.ndarray, percentile: float = 90.0) -> float:
    if np.isnan(target_2d).all() or np.isnan(forecast_2d).all():
        return np.nan
    threshold = np.nanpercentile(target_2d, percentile)
    f_mask = forecast_2d >= threshold
    t_mask = target_2d >= threshold
    
    intersection = (f_mask & t_mask).sum()
    union = (f_mask | t_mask).sum()
    return float(intersection / union) if union > 0 else np.nan

def extreme_area_ratio(forecast_2d: np.ndarray, target_2d: np.ndarray, percentile: float = 90.0) -> float:
    if np.isnan(target_2d).all() or np.isnan(forecast_2d).all():
        return np.nan
    threshold = np.nanpercentile(target_2d, percentile)
    t_count = (target_2d >= threshold).sum()
    if t_count == 0:
        return np.nan
    f_count = (forecast_2d >= threshold).sum()
    return float(f_count / t_count)

# =============================================================================
# Section 2: Data Loading & Memory-Efficient Alignment Helpers
# =============================================================================

def standardize_longitude(ds: xr.Dataset) -> xr.Dataset:
    if "longitude" in ds.coords:
        new_lon = ds["longitude"] % 360
        ds = ds.assign_coords(longitude=new_lon)
        ds = ds.sortby("longitude")
    return ds

def load_era5_zarr() -> xr.Dataset:
    logger.info("Connecting to ARCO ERA5 zarr store...")
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        storage_options={"token": "anon"},
    )
    ds = ds.rename({"2m_temperature": "surface_air_temperature"})
    return ds[["surface_air_temperature"]]

def load_hres_zarr() -> xr.Dataset:
    logger.info("Connecting to HRES WeatherBench2 zarr store...")
    ds = xr.open_zarr(
        "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
        storage_options={"token": "anon"},
    )
    ds = ds.rename(
        {
            "2m_temperature": "surface_air_temperature",
            "prediction_timedelta": "lead_time",
            "time": "init_time",
        }
    )
    return ds[["surface_air_temperature"]]

def align_and_subset_2d(fcst_da: xr.DataArray, tgt_da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    fcst_sub = fcst_da.interp_like(tgt_da, method="nearest", kwargs={"fill_value": "extrapolate"})
    return fcst_sub.values, tgt_da.values

def aurora_batch_to_xarray(predicted_batch, init_time, lead_time_hours):
    t2m_tensor = predicted_batch.surf_vars["2t"].detach().cpu().numpy().squeeze()
    lats = predicted_batch.metadata.lat
    lons = predicted_batch.metadata.lon
    ds = xr.Dataset(
        data_vars={
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

def get_normal_dates(case, all_extreme_years, num_years=3):
    event_start = pd.to_datetime(case.start_date)
    event_end = pd.to_datetime(case.end_date)
    
    normal_dates = []
    normal_year = event_start.year - 1
    
    while len(normal_dates) < num_years:
        if normal_year not in all_extreme_years:
            normal_start = event_start.replace(year=normal_year)
            normal_end = event_end.replace(year=normal_year)
            normal_dates.append((normal_start, normal_end))
        normal_year -= 1
        
    return normal_dates

# =============================================================================
# Section 3: Generator
# =============================================================================

def generate_normal_forecasts(output_dir="normal_forecasts/"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if files already exist to skip model loading if possible
    existing_files = list(Path(output_dir).glob("*.nc"))
    
    all_cases = ewb.load_cases() 
    heatwave_cases = [case for case in all_cases if case.event_type == "heat_wave"]
    all_hw_years = [pd.to_datetime(case.start_date).year for case in heatwave_cases]
    
    target_start_lead_times = [24, 72, 120] 
    
    # First, let's see what needs generating
    to_generate = []
    for case in heatwave_cases:
        normal_dates_list = get_normal_dates(case, all_hw_years)
        for normal_start, normal_end in normal_dates_list:
            for start_lead in target_start_lead_times:
                init_time = normal_start - pd.Timedelta(hours=start_lead)
                filename = Path(output_dir) / f"aurora_hw_forecast_{init_time.strftime('%Y%m%d')}_L{start_lead}_full_event.nc"
                if not filename.exists():
                    to_generate.append((case, normal_start, normal_end, start_lead, init_time, filename))
                
    if not to_generate:
        logger.info("All normal forecasts already generated.")
        return
        
    logger.info(f"Generating {len(to_generate)} normal forecast rollout files...")
    data_pipeline = AuroraDataLoader(cache_dir=Path("weather_data"))
    aurora_model = AuroraV1p5()
    aurora_model.load_checkpoint("microsoft/aurora", "aurora-0.25-v1.5.ckpt", revision="main")
    forecaster = AuroraForecaster(model=aurora_model)
    
    for case, normal_start, normal_end, start_lead, init_time, filename in to_generate:
        event_duration_hours = int((normal_end - normal_start).total_seconds() / 3600)
        logger.info(f"Processing: Normal baseline for {case.title} | Init: {init_time} | Start Lead: {start_lead}h")

        try:
            input_batch, _ = data_pipeline.get_batches(
                    init_time, 
                    history_steps=1, 
                    forecast_steps=0,
                    bbox=None
                )
            
            total_hours_to_roll = start_lead + event_duration_hours
            total_steps_needed = int(total_hours_to_roll / 6)
            
            predicted_batches = forecaster.predict_rollout(
                initial_batch=input_batch, 
                steps=total_steps_needed,
                fine_lead_times=[6.0]
            )
            
            start_index = int(start_lead / 6) - 1 
            event_ds_list = []
            
            for i, batch in enumerate(predicted_batches[start_index:]):
                current_lead_time = start_lead + (i * 6)
                step_ds = aurora_batch_to_xarray(batch, init_time, current_lead_time)
                event_ds_list.append(step_ds)
                
            full_event_ds = xr.concat(event_ds_list, dim="lead_time")
            full_event_ds.to_netcdf(filename, engine='h5netcdf')
            
            # Handle and empty GPU memory
            del input_batch
            del predicted_batches
            del event_ds_list
            del full_event_ds
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed on {init_time} for lead {start_lead}h: {e}")
            
            # Automatically delete corrupted cache files so they are re-downloaded next time
            error_str = str(e)
            if "HDF error" in error_str:
                match = re.search(r"'(.*\.nc)'", error_str)
                if match:
                    corrupt_file = Path(match.group(1))
                    if corrupt_file.exists():
                        logger.warning(f"Deleting corrupted cache file: {corrupt_file}")
                        corrupt_file.unlink()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# =============================================================================
# Section 4: Evaluator
# =============================================================================

def get_aurora_files(forecast_dir: str = "normal_forecasts") -> list[Path]:
    forecast_path = Path(forecast_dir)
    nc_files = sorted(list(forecast_path.glob("*.nc")))
    return nc_files

def evaluate_normal(output_csv: str = "normal_evaluations.csv"):
    logger.info("=" * 60)
    logger.info("Starting Direct Normal Evaluator Pipeline")
    logger.info("=" * 60)

    era5_ds = load_era5_zarr()
    hres_ds = load_hres_zarr()
    aurora_files = get_aurora_files()

    all_cases = ewb.load_cases()
    hw_cases = [c for c in all_cases if c.event_type == "heat_wave"]
    all_hw_years = [pd.to_datetime(case.start_date).year for case in hw_cases]

    results = []

    for case_idx, case in enumerate(hw_cases):
        normal_dates_list = get_normal_dates(case, all_hw_years)
        
        for normal_start, normal_end in normal_dates_list:
            logger.info(f"\nEvaluating Normal Baseline for Case {case.case_id_number} ({case_idx+1}/{len(hw_cases)})")
            logger.info(f"  Target Event Date: {case.start_date} to {case.end_date}")
            logger.info(f"  Normal Date: {normal_start} to {normal_end}")
    
            if not isinstance(case.location, ewb.regions.BoundingBoxRegion):
                continue
    
            logger.info("  Downloading ERA5 subset for this case...")
            era5_case = era5_ds.sel(time=slice(normal_start, normal_end))
            era5_case = case.location.mask(era5_case).compute()
            era5_case = standardize_longitude(era5_case)
    
            for nc_file in aurora_files:
                match = re.search(r"_L(\d+)_", nc_file.name)
                lead_bucket = f"L{match.group(1)}" if match else "L?"
                model_name = f"aurora1.5_{lead_bucket}"
                
                aurora_fcst = xr.open_dataset(nc_file)
                init_t = aurora_fcst.init_time.values[0]
                lead_times = aurora_fcst.lead_time.values
                
                valid_times = init_t + lead_times
                if not np.any((valid_times >= normal_start) & (valid_times <= normal_end)):
                    aurora_fcst.close()
                    continue
                
                aurora_fcst = case.location.mask(aurora_fcst).compute()
                aurora_fcst = standardize_longitude(aurora_fcst)
                
                if "2t_aurora" in aurora_fcst.data_vars:
                    aurora_var = "2t_aurora"
                else:
                    aurora_var = [v for v in aurora_fcst.data_vars if v != "latitude" and v != "longitude"][0]
                
                try:
                    hres_fcst = hres_ds.sel(init_time=init_t, method="pad")
                    hres_init_t = hres_fcst.init_time.values
                    hres_fcst = case.location.mask(hres_fcst).compute()
                    hres_fcst = standardize_longitude(hres_fcst)
                    has_hres = True
                except KeyError:
                    has_hres = False
    
                for lt in aurora_fcst.lead_time.values:
                    valid_time = init_t + lt
                    lt_hours = int(pd.Timedelta(lt).total_seconds() / 3600)
                    
                    if pd.to_datetime(normal_start) <= pd.to_datetime(valid_time) <= pd.to_datetime(normal_end):
                        try:
                            tgt_2d = era5_case["surface_air_temperature"].sel(time=valid_time)
                        except KeyError:
                            continue
                        
                        fcst_2d_aurora = aurora_fcst[aurora_var].sel(init_time=init_t, lead_time=lt)
                        fcst_arr_aurora, tgt_arr_aurora = align_and_subset_2d(fcst_2d_aurora, tgt_2d)
                        
                        metrics_aurora = {
                            "Peak_Amplitude_Error": peak_amplitude_error(fcst_arr_aurora, tgt_arr_aurora),
                            "Conditional_Bias_Extremes": conditional_bias_extremes(fcst_arr_aurora, tgt_arr_aurora),
                            "RootMeanSquaredError": rmse(fcst_arr_aurora, tgt_arr_aurora),
                            "Spatial_IOU": intersection_over_union(fcst_arr_aurora, tgt_arr_aurora),
                            "Extreme_Area_Ratio": extreme_area_ratio(fcst_arr_aurora, tgt_arr_aurora),
                        }
                        
                        for m_name, value in metrics_aurora.items():
                            results.append({
                                "case_id_number": case.case_id_number,
                                "event_type": "heat_wave_normal_baseline",
                                "forecast_source": model_name,
                                "target_source": "ERA5",
                                "metric": m_name,
                                "forecast_variable": "surface_air_temperature",
                                "target_variable": "surface_air_temperature",
                                "init_time": init_t,
                                "valid_time": valid_time,
                                "lead_time": lt_hours,
                                "value": value
                            })
                            
                        if has_hres:
                            hres_lt = valid_time - hres_init_t
                            hres_lt_hours = int(pd.Timedelta(hres_lt).total_seconds() / 3600)
                            
                            try:
                                fcst_2d_hres = hres_fcst["surface_air_temperature"].sel(lead_time=hres_lt_hours)
                            except KeyError:
                                pass
                            else:
                                fcst_arr_hres, tgt_arr_hres = align_and_subset_2d(fcst_2d_hres, tgt_2d)
                                
                                metrics_hres = {
                                    "Peak_Amplitude_Error": peak_amplitude_error(fcst_arr_hres, tgt_arr_hres),
                                    "Conditional_Bias_Extremes": conditional_bias_extremes(fcst_arr_hres, tgt_arr_hres),
                                    "RootMeanSquaredError": rmse(fcst_arr_hres, tgt_arr_hres),
                                    "Spatial_IOU": intersection_over_union(fcst_arr_hres, tgt_arr_hres),
                                    "Extreme_Area_Ratio": extreme_area_ratio(fcst_arr_hres, tgt_arr_hres),
                                }
                                
                                for m_name, value in metrics_hres.items():
                                    results.append({
                                        "case_id_number": case.case_id_number,
                                        "event_type": "heat_wave_normal_baseline",
                                        "forecast_source": "HRES",
                                        "target_source": "ERA5",
                                        "metric": m_name,
                                        "forecast_variable": "surface_air_temperature",
                                        "target_variable": "surface_air_temperature",
                                        "init_time": hres_init_t,
                                        "valid_time": valid_time,
                                        "lead_time": hres_lt_hours,
                                        "value": value
                                    })
                            
                aurora_fcst.close()

    if results:
        df = pd.DataFrame(results)
        col_order = [
            "value", "lead_time", "init_time", "valid_time", 
            "target_variable", "metric", "forecast_source", "target_source", 
            "case_id_number", "event_type", "forecast_variable"
        ]
        for col in col_order:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[col_order]
        
        output_path = Path(output_csv)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path.resolve()}")
        return df
    else:
        logger.warning("No results were generated!")
        return pd.DataFrame()


if __name__ == "__main__":
    generate_normal_forecasts()
    evaluate_normal()
