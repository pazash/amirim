import os
import re
import logging
from pathlib import Path
from datetime import datetime, timedelta
import gc

import torch
import numpy as np
import pandas as pd
import xarray as xr

# Import classes from src
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dataloader import AuroraDataLoader
from src.forecaster import AuroraForecaster
from aurora import AuroraV1p5

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Evaluation bounding box — all scopes are evaluated on this region.
# Shifted east from the original Mediterranean bbox so that all prediction scopes
# can be perfectly centered (Aurora requires lon in [0, 360) AND strictly increasing,
# so we cannot cross the prime meridian).
ORIGINAL_MED_BBOX = {
    "lon_min": 18.0,
    "lon_max": 61.75,
    "lat_max": 53.25,
    "lat_min": 25.5
}

# Prediction scopes — all centered on (39.875°E, 39.375°N).
# Grid dimensions (W x H) are exact multiples of 16 for Aurora's patch-based architecture.
PREDICTION_SCOPES = {
    "Original": {  # 176x112 points — same as evaluation box
        "lon_min": 18.0,
        "lon_max": 61.75,
        "lat_max": 53.25,
        "lat_min": 25.5
    },
    "Enlarged_Small": {  # 224x176 points — ~+6° centered padding
        "lon_min": 12.0,
        "lon_max": 67.75,
        "lat_max": 61.25,
        "lat_min": 17.5
    },
    "Enlarged_Large": {  # 320x240 points — ~+18° centered padding
        "lon_min": 0.0,
        "lon_max": 79.75,
        "lat_max": 69.25,
        "lat_min": 9.5
    },
    "Global": None
}


def rmse(forecast_2d: np.ndarray, target_2d: np.ndarray) -> float:
    if np.isnan(target_2d).all() or np.isnan(forecast_2d).all():
        return np.nan
    diff = forecast_2d - target_2d
    return float(np.sqrt(np.nanmean(diff**2)))

def aurora_batch_to_xarray(predicted_batch, init_time, lead_time_hours):
    t2m_tensor = predicted_batch.surf_vars["2t"].detach().cpu().numpy().squeeze()
    lats = predicted_batch.metadata.lat
    lons = predicted_batch.metadata.lon
    ds = xr.Dataset(
        data_vars={
            "2t_aurora": (["latitude", "longitude"], t2m_tensor),
        },
        coords={
            "init_time": [pd.to_datetime(init_time)],
            "lead_time": pd.to_timedelta([f"{lead_time_hours}h"]),
            "latitude": lats,
            "longitude": lons,
        }
    )
    return ds

def align_and_subset_2d(fcst_da: xr.DataArray, tgt_da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    fcst_sub = fcst_da.interp_like(tgt_da, method="nearest", kwargs={"fill_value": "extrapolate"})
    return fcst_sub.values, tgt_da.values

def generate_forecasts(output_dir: str = "med_comparison_forecasts"):
    """Runs Aurora predictions and saves them to NetCDF."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cache_dir = Path("weather_data")
    
    # 10 arbitrary diverse dates across different seasons
    selected_dates = [
        datetime(2021, 6, 15, 12),
        datetime(2021, 7, 25, 12),
        datetime(2022, 1, 10, 12),
        datetime(2022, 4, 18, 12),
        datetime(2022, 11, 20, 12),
        datetime(2021, 3, 8, 12),
        datetime(2021, 10, 12, 12),
        datetime(2022, 8, 5, 12),
        datetime(2021, 12, 22, 12),
        datetime(2022, 6, 30, 12),
    ]
    
    lead_times_h = [6, 24, 72, 120]
    
    logger.info("Loading Aurora Model for inference...")
    aurora_model = AuroraV1p5()
    aurora_model.load_checkpoint("microsoft/aurora", "aurora-0.25-v1.5.ckpt", revision="main")
    forecaster = AuroraForecaster(model=aurora_model)
    
    # Create separate DataLoaders per scope to isolate their caches
    data_pipelines = {
        scope_name: AuroraDataLoader(cache_dir=cache_dir / scope_name)
        for scope_name in PREDICTION_SCOPES.keys()
    }
    
    for init_dt in selected_dates:
        logger.info(f"--- Generating Forecasts for Init Time: {init_dt} ---")
        init_str = init_dt.strftime("%Y%m%d_%H")
        
        for scope_name, bbox in PREDICTION_SCOPES.items():
            data_pipeline = data_pipelines[scope_name]
            out_file = Path(output_dir) / f"aurora_{scope_name}_{init_str}.nc"
            
            if not out_file.exists():
                logger.info(f"Running Aurora for scope: {scope_name}...")
                try:
                    input_batch, _ = data_pipeline.get_batches(init_dt, history_steps=1, forecast_steps=0, bbox=bbox)
                    forecast_batches = forecaster.predict_rollout(
                        initial_batch=input_batch,
                        steps=int(max(lead_times_h) / 6),
                        fine_lead_times=[6.0]
                    )
                    del input_batch
                    
                    # Combine and save
                    ds_list = []
                    for lt_h in lead_times_h:
                        step_idx = int(lt_h / 6) - 1
                        batch = forecast_batches[step_idx]
                        ds_list.append(aurora_batch_to_xarray(batch, init_dt, lt_h))
                    
                    xr.concat(ds_list, dim="lead_time").to_netcdf(out_file, engine='h5netcdf')
                    del forecast_batches
                    del ds_list
                except Exception as e:
                    logger.error(f"{scope_name} run failed for {init_dt}: {e}")
            else:
                logger.info(f"{scope_name} forecast for {init_dt} already exists.")
                
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def evaluate_forecasts(forecast_dir: str = "med_comparison_forecasts"):
    """Evaluates the saved forecasts against ERA5 strictly on the ORIGINAL bounding box."""
    logger.info("Starting Evaluation...")
    cache_dir = Path("weather_data")
    forecast_path = Path(forecast_dir)
    
    # Use the "Original" cache dir to fetch ground truth matching the original bounding box
    data_pipeline = AuroraDataLoader(cache_dir=cache_dir / "Original")
    
    lead_times_h = [6, 24, 72, 120]
    results = []
    
    # Get all unique init times from generated files
    all_files = list(forecast_path.glob("aurora_Original_*.nc"))
    
    if not all_files:
        logger.warning(f"No forecast files found in {forecast_dir}. Please run generation first.")
        return
        
    for original_file in all_files:
        match = re.search(r"aurora_Original_(\d{8}_\d{2})\.nc", original_file.name)
        if not match:
            continue
            
        init_str = match.group(1)
        init_dt = datetime.strptime(init_str, "%Y%m%d_%H")
        
        logger.info(f"--- Evaluating Forecasts for Init Time: {init_dt} ---")
        
        # Ground Truth (ERA5 strictly on Original Med bbox)
        target_dts = [init_dt + timedelta(hours=lt) for lt in lead_times_h]
        try:
            target_ds = data_pipeline._fetch_era5_combined(target_dts, bbox=ORIGINAL_MED_BBOX)
        except Exception as e:
            logger.error(f"Failed to load target data for {init_dt}: {e}")
            continue

        # Evaluate across all scopes
        for scope_name in PREDICTION_SCOPES.keys():
            scope_file = forecast_path / f"aurora_{scope_name}_{init_str}.nc"
            if not scope_file.exists():
                logger.warning(f"Missing {scope_name} forecast for {init_dt}, skipping.")
                continue
                
            try:
                scope_ds = xr.open_dataset(scope_file)
            except Exception as e:
                logger.error(f"Failed to open forecast file {scope_file}: {e}")
                continue
                
            for lt_h in lead_times_h:
                valid_dt = init_dt + timedelta(hours=lt_h)
                valid_dt_str = valid_dt.strftime("%Y-%m-%dT%H:00:00")
                
                try:
                    tgt_2d = target_ds["t2m"].sel(time=valid_dt_str)
                except KeyError:
                    logger.warning(f"Target data missing for {valid_dt_str}, skipping.")
                    continue
                    
                # Evaluate on Original Box
                fcst_2d = scope_ds["2t_aurora"].sel(lead_time=pd.to_timedelta(f"{lt_h}h")).squeeze()
                fcst_cropped = fcst_2d.sel(
                    latitude=slice(ORIGINAL_MED_BBOX["lat_max"], ORIGINAL_MED_BBOX["lat_min"]),
                    longitude=slice(ORIGINAL_MED_BBOX["lon_min"], ORIGINAL_MED_BBOX["lon_max"])
                )
                fcst_arr, tgt_arr = align_and_subset_2d(fcst_cropped, tgt_2d)
                rmse_val = rmse(fcst_arr, tgt_arr)
                
                logger.info(f"Lead time {lt_h}h | Model: {scope_name} | RMSE: {rmse_val:.3f}")
                
                results.append({
                    "init_time": init_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "valid_time": valid_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "lead_time_hours": lt_h,
                    "metric": "RMSE",
                    "model": f"Aurora_{scope_name}",
                    "value": rmse_val
                })
                
            scope_ds.close()

    if results:
        df = pd.DataFrame(results)
        df.to_csv("aurora_context_comparison_rmse.csv", index=False)
        logger.info("Saved results to aurora_context_comparison_rmse.csv")
    else:
        logger.warning("No results to save.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare Aurora scopes over Mediterranean")
    parser.add_argument("--generate", action="store_true", help="Run forecast generation")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on saved forecasts")
    
    args = parser.parse_args()
    
    if not args.generate and not args.evaluate:
        logger.info("No flags provided. Running both generation and evaluation sequentially.")
        generate_forecasts()
        evaluate_forecasts()
    else:
        if args.generate:
            generate_forecasts()
        if args.evaluate:
            evaluate_forecasts()
