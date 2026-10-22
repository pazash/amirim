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
from src.utils import rmse, aurora_batch_to_xarray, align_and_subset_2d
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

# Variables to evaluate across scopes
EVAL_VARIABLES = [
    {
        "id": "2t",
        "name": "2m Surface Temperature",
        "unit": "K",
        "type": "surface",
        "fcst_key": "2t_aurora",
        "target_key": "t2m",
    },
    {
        "id": "t_850",
        "name": "Temperature (850 hPa)",
        "unit": "K",
        "type": "atmos",
        "fcst_key": "t",
        "target_key": "t",
        "level": 850,
    },
    {
        "id": "t_250",
        "name": "Temperature (250 hPa)",
        "unit": "K",
        "type": "atmos",
        "fcst_key": "t",
        "target_key": "t",
        "level": 250,
    },
    {
        "id": "wind_speed_850",
        "name": "Wind Speed (850 hPa)",
        "unit": "m/s",
        "type": "wind_speed",
        "level": 850,
    },
    {
        "id": "wind_speed_250",
        "name": "Wind Speed (250 hPa)",
        "unit": "m/s",
        "type": "wind_speed",
        "level": 250,
    },
    {
        "id": "z_850",
        "name": "Geopotential (850 hPa)",
        "unit": "m²/s²",
        "type": "atmos",
        "fcst_key": "z",
        "target_key": "z",
        "level": 850,
    },
    {
        "id": "z_250",
        "name": "Geopotential (250 hPa)",
        "unit": "m²/s²",
        "type": "atmos",
        "fcst_key": "z",
        "target_key": "z",
        "level": 250,
    },
    {
        "id": "q_850",
        "name": "Specific Humidity (850 hPa)",
        "unit": "kg/kg",
        "type": "atmos",
        "fcst_key": "q",
        "target_key": "q",
        "level": 850,
    },
    {
        "id": "q_250",
        "name": "Specific Humidity (250 hPa)",
        "unit": "kg/kg",
        "type": "atmos",
        "fcst_key": "q",
        "target_key": "q",
        "level": 250,
    },
]


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
            
            # Check if file exists and has atmospheric variables
            file_valid = False
            if out_file.exists():
                try:
                    with xr.open_dataset(out_file) as test_ds:
                        if "t" in test_ds and "u" in test_ds:
                            file_valid = True
                except Exception:
                    file_valid = False

            if not file_valid:
                logger.info(f"Running Aurora for scope: {scope_name}...")
                try:
                    input_batch, _ = data_pipeline.get_batch(init_dt, history_steps=1, forecast_steps=0, bbox=bbox)
                    forecast_batches = forecaster.predict_rollout(
                        initial_batch=input_batch,
                        steps=int(max(lead_times_h) / 6),
                        fine_lead_times=[6.0]
                    )
                    del input_batch
                    
                    # Combine and save with atmospheric variables at 250 & 850 hPa
                    ds_list = []
                    for lt_h in lead_times_h:
                        step_idx = int(lt_h / 6) - 1
                        batch = forecast_batches[step_idx]
                        ds_list.append(aurora_batch_to_xarray(
                            batch,
                            init_dt,
                            lt_h,
                            include_atmos=True,
                            atmos_levels=[250, 850],
                            atmos_vars=["t", "u", "v", "z", "q"],
                        ))
                    
                    xr.concat(ds_list, dim="lead_time").to_netcdf(out_file, engine='h5netcdf')
                    del forecast_batches
                    del ds_list
                except Exception as e:
                    logger.error(f"{scope_name} run failed for {init_dt}: {e}")
            else:
                logger.info(f"{scope_name} forecast for {init_dt} already exists with atmospheric variables.")
                
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def evaluate_forecasts(
    forecast_dir: str = "med_comparison_forecasts",
    output_csv: str = "aurora_context_comparison_rmse.csv"
):
    """Evaluates the saved forecasts against ERA5 strictly on the ORIGINAL bounding box across all variables."""
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
            target_ds = data_pipeline.fetch_era5(target_dts, bbox=ORIGINAL_MED_BBOX)
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
                lt_delta = pd.to_timedelta(f"{lt_h}h")

                for var_cfg in EVAL_VARIABLES:
                    var_id = var_cfg["id"]
                    try:
                        # 1. Target DataArray
                        if var_cfg["type"] == "surface":
                            if var_cfg["target_key"] not in target_ds:
                                logger.warning(f"Target variable {var_cfg['target_key']} missing for {valid_dt_str}")
                                continue
                            tgt_2d = target_ds[var_cfg["target_key"]].sel(time=valid_dt_str)
                        elif var_cfg["type"] == "atmos":
                            if var_cfg["target_key"] not in target_ds:
                                logger.warning(f"Target variable {var_cfg['target_key']} missing for {valid_dt_str}")
                                continue
                            tgt_2d = target_ds[var_cfg["target_key"]].sel(
                                time=valid_dt_str, level=var_cfg["level"], method="nearest"
                            ).squeeze()
                        elif var_cfg["type"] == "wind_speed":
                            if "u" not in target_ds or "v" not in target_ds:
                                logger.warning(f"Target u/v missing for {valid_dt_str}")
                                continue
                            tgt_u = target_ds["u"].sel(
                                time=valid_dt_str, level=var_cfg["level"], method="nearest"
                            ).squeeze()
                            tgt_v = target_ds["v"].sel(
                                time=valid_dt_str, level=var_cfg["level"], method="nearest"
                            ).squeeze()
                            tgt_2d = np.sqrt(tgt_u**2 + tgt_v**2)

                        # 2. Forecast DataArray
                        if var_cfg["type"] == "surface":
                            fcst_key = var_cfg["fcst_key"] if var_cfg["fcst_key"] in scope_ds else "2t"
                            if fcst_key not in scope_ds:
                                logger.warning(f"Forecast key {fcst_key} missing in {scope_file.name}")
                                continue
                            fcst_2d = scope_ds[fcst_key].sel(lead_time=lt_delta).squeeze()
                        elif var_cfg["type"] == "atmos":
                            fcst_key = var_cfg["fcst_key"]
                            if fcst_key not in scope_ds:
                                logger.warning(f"Forecast key {fcst_key} missing in {scope_file.name}")
                                continue
                            fcst_2d = scope_ds[fcst_key].sel(
                                lead_time=lt_delta, level=var_cfg["level"], method="nearest"
                            ).squeeze()
                        elif var_cfg["type"] == "wind_speed":
                            if "u" not in scope_ds or "v" not in scope_ds:
                                logger.warning(f"Forecast u/v missing in {scope_file.name}")
                                continue
                            fcst_u = scope_ds["u"].sel(
                                lead_time=lt_delta, level=var_cfg["level"], method="nearest"
                            ).squeeze()
                            fcst_v = scope_ds["v"].sel(
                                lead_time=lt_delta, level=var_cfg["level"], method="nearest"
                            ).squeeze()
                            fcst_2d = np.sqrt(fcst_u**2 + fcst_v**2)

                        # 3. Crop to Original Med Box and compute RMSE
                        fcst_cropped = fcst_2d.sel(
                            latitude=slice(ORIGINAL_MED_BBOX["lat_max"], ORIGINAL_MED_BBOX["lat_min"]),
                            longitude=slice(ORIGINAL_MED_BBOX["lon_min"], ORIGINAL_MED_BBOX["lon_max"])
                        )
                        fcst_arr, tgt_arr = align_and_subset_2d(fcst_cropped, tgt_2d)
                        rmse_val = rmse(fcst_arr, tgt_arr)

                        logger.info(
                            f"Lead time {lt_h:3d}h | Variable: {var_id:<15} | Model: {scope_name:<15} | RMSE: {rmse_val:.4f} {var_cfg['unit']}"
                        )

                        results.append({
                            "init_time": init_dt.strftime("%Y-%m-%d %H:%M:%S"),
                            "valid_time": valid_dt.strftime("%Y-%m-%d %H:%M:%S"),
                            "lead_time_hours": lt_h,
                            "variable": var_id,
                            "metric": "RMSE",
                            "model": f"Aurora_{scope_name}",
                            "value": rmse_val
                        })
                    except Exception as e:
                        logger.error(f"Error evaluating {var_id} for {scope_name} at {lt_h}h ({valid_dt_str}): {e}")

            scope_ds.close()

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved results to {output_csv}")
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
