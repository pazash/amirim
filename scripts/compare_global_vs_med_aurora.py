import os
import re
import random
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

def rmse(forecast_2d: np.ndarray, target_2d: np.ndarray) -> float:
    if np.isnan(target_2d).all() or np.isnan(forecast_2d).all():
        return np.nan
    diff = forecast_2d - target_2d
    return float(np.sqrt(np.nanmean(diff**2)))

def get_available_dates(cache_dir: str = "weather_data") -> list[datetime]:
    """Scans cache_dir for downloaded ERA5 files and returns a list of all valid daily init_times (12:00) available."""
    cache_path = Path(cache_dir)
    available_dates = []
    
    if not cache_path.exists():
        logger.warning(f"Cache dir {cache_dir} not found. Returning empty list.")
        return []

    # Look for files like era5_surf_20210710_00_to_20210815_00.nc
    for file_path in cache_path.glob("era5_surf_*_to_*.nc"):
        match = re.search(r"era5_surf_(\d{8}_\d{2})_to_(\d{8}_\d{2})\.nc", file_path.name)
        if match:
            start_str, end_str = match.groups()
            start_dt = datetime.strptime(start_str, "%Y%m%d_%H")
            end_dt = datetime.strptime(end_str, "%Y%m%d_%H")
            
            # Add days in between (we need at least 6 hours of history, so start from start_dt + 12h)
            # We'll use 12:00 UTC as our standard init time.
            current = start_dt.replace(hour=12, minute=0, second=0)
            if current < start_dt + timedelta(hours=6):
                current += timedelta(days=1)
                
            # We also need to be able to forecast forward (e.g., 5 days = 120 hours).
            # So the end_dt should be at least 120 hours after our init_time if we want to evaluate it.
            while current + timedelta(hours=120) <= end_dt:
                available_dates.append(current)
                current += timedelta(days=1)
                
    return sorted(list(set(available_dates)))

def select_random_dates(dates: list[datetime], num_samples: int = 5) -> list[datetime]:
    """Selects random dates, trying to spread them across different years."""
    if not dates:
        return []
        
    dates_by_year = {}
    for d in dates:
        dates_by_year.setdefault(d.year, []).append(d)
        
    selected = []
    years = list(dates_by_year.keys())
    
    # Try to get at least one from each year if possible
    for year in years:
        if len(selected) < num_samples:
            d = random.choice(dates_by_year[year])
            selected.append(d)
            dates_by_year[year].remove(d)
            if not dates_by_year[year]:
                del dates_by_year[year]
                
    # If we still need more, randomly sample from remaining
    remaining_dates = [d for sublist in dates_by_year.values() for d in sublist]
    if len(selected) < num_samples and remaining_dates:
        needed = num_samples - len(selected)
        selected.extend(random.sample(remaining_dates, min(needed, len(remaining_dates))))
        
    return selected

def align_and_subset_2d(fcst_da: xr.DataArray, tgt_da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    fcst_sub = fcst_da.interp_like(tgt_da, method="nearest", kwargs={"fill_value": "extrapolate"})
    return fcst_sub.values, tgt_da.values

def aurora_batch_to_xarray(predicted_batch, init_time, lead_time_hours):
    t2m_tensor = predicted_batch.surf_vars["2t"].detach().cpu().numpy().squeeze()
    lats = predicted_batch.metadata.lat
    lons = predicted_batch.metadata.lon
    ds = xr.Dataset(
        data_vars={
            "2t_aurora": (["latitude", "longitude"], t2m_tensor),
        },
        coords={
            "latitude": lats,
            "longitude": lons,
        }
    )
    return ds

def main():
    logger.info("Starting Global vs Med Aurora comparison...")
    
    cache_dir = Path("weather_data")
    
    # Define bounding box (ICON-MED enlarged by ~2.5 degrees)
    # Original: Lon 4.0 to 45.5, Lat 25.5 to 53.0
    med_bbox = {
        "lon_min": 1.5,
        "lon_max": 48.0,
        "lat_max": 55.5,
        "lat_min": 23.0
    }
    logger.info(f"Using Bounding Box: {med_bbox}")
    
    # Find available dates
    available_dates = get_available_dates(cache_dir)
    if not available_dates:
        logger.warning(f"No valid dates found in {cache_dir}. Assuming this runs on remote with missing files.")
        logger.info("Will attempt to use a default date: 2021-07-25 12:00:00")
        selected_dates = [datetime(2021, 7, 25, 12)]
    else:
        selected_dates = select_random_dates(available_dates, num_samples=3)
        
    logger.info(f"Selected dates for evaluation: {[d.strftime('%Y-%m-%d') for d in selected_dates]}")
    
    # Load Model
    logger.info("Loading Aurora Model...")
    aurora_model = AuroraV1p5()
    aurora_model.load_checkpoint("microsoft/aurora", "aurora-0.25-v1.5.ckpt", revision="main")
    forecaster = AuroraForecaster(model=aurora_model)
    data_pipeline = AuroraDataLoader(cache_dir=cache_dir)
    
    # Define lead times to evaluate
    lead_times_h = [24, 72, 120]
    results = []
    
    for init_dt in selected_dates:
        logger.info(f"--- Processing Init Time: {init_dt} ---")
        
        # 1. Global Run
        logger.info("Running Global Aurora...")
        try:
            global_input, _ = data_pipeline.get_batches(init_dt, history_steps=1, forecast_steps=0, bbox=None)
            global_forecast_batches = forecaster.predict_rollout(
                initial_batch=global_input,
                steps=int(max(lead_times_h) / 6),
                fine_lead_times=[6.0]
            )
            del global_input
        except Exception as e:
            logger.error(f"Global run failed for {init_dt}: {e}")
            continue
            
        # 2. Regional Run (Med bbox)
        logger.info("Running Regional Aurora (Med bbox)...")
        try:
            med_input, _ = data_pipeline.get_batches(init_dt, history_steps=1, forecast_steps=0, bbox=med_bbox)
            med_forecast_batches = forecaster.predict_rollout(
                initial_batch=med_input,
                steps=int(max(lead_times_h) / 6),
                fine_lead_times=[6.0]
            )
            del med_input
        except Exception as e:
            logger.error(f"Regional run failed for {init_dt}: {e}")
            continue
            
        # 3. Ground Truth (ERA5 for Med)
        target_dts = [init_dt + timedelta(hours=lt) for lt in lead_times_h]
        try:
            target_ds = data_pipeline._fetch_era5_combined(target_dts, bbox=med_bbox)
        except Exception as e:
            logger.error(f"Failed to load target data for {init_dt}: {e}")
            continue

        # Evaluate
        for lt_h in lead_times_h:
            valid_dt = init_dt + timedelta(hours=lt_h)
            step_idx = int(lt_h / 6) - 1
            
            # Ground truth
            valid_dt_str = valid_dt.strftime("%Y-%m-%dT%H:00:00")
            try:
                tgt_2d = target_ds["2m_temperature"].sel(time=valid_dt_str)
            except KeyError:
                logger.warning(f"Target data missing for {valid_dt_str}, skipping.")
                continue

            # Global Model Output cropped to bbox
            global_batch = global_forecast_batches[step_idx]
            global_ds = aurora_batch_to_xarray(global_batch, init_dt, lt_h)
            
            # Crop global ds to Med bbox
            global_cropped = global_ds.sel(
                latitude=slice(med_bbox["lat_max"], med_bbox["lat_min"]),
                longitude=slice(med_bbox["lon_min"], med_bbox["lon_max"])
            )
            fcst_arr_global, tgt_arr = align_and_subset_2d(global_cropped["2t_aurora"], tgt_2d)
            rmse_global = rmse(fcst_arr_global, tgt_arr)
            
            # Regional Model Output (already in bbox)
            med_batch = med_forecast_batches[step_idx]
            med_ds = aurora_batch_to_xarray(med_batch, init_dt, lt_h)
            fcst_arr_med, tgt_arr_med = align_and_subset_2d(med_ds["2t_aurora"], tgt_2d)
            rmse_med = rmse(fcst_arr_med, tgt_arr_med)
            
            logger.info(f"Lead time {lt_h}h | RMSE Global: {rmse_global:.3f} | RMSE Regional: {rmse_med:.3f}")
            
            results.append({
                "init_time": init_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "valid_time": valid_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "lead_time_hours": lt_h,
                "metric": "RMSE",
                "model": "Aurora_Global",
                "value": rmse_global
            })
            
            results.append({
                "init_time": init_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "valid_time": valid_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "lead_time_hours": lt_h,
                "metric": "RMSE",
                "model": "Aurora_Regional",
                "value": rmse_med
            })
            
        # Cleanup memory for next date
        del global_forecast_batches
        del med_forecast_batches
        del target_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if results:
        df = pd.DataFrame(results)
        df.to_csv("aurora_global_vs_regional_rmse.csv", index=False)
        logger.info("Saved results to aurora_global_vs_regional_rmse.csv")
    else:
        logger.warning("No results to save.")

if __name__ == "__main__":
    main()
