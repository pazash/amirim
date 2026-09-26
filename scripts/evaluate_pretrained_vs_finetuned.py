import os
import re
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import gc
import dataclasses

import torch
import numpy as np
import pandas as pd
import xarray as xr

# Import classes from src
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dataloader import AuroraDataLoader
from src.forecaster import AuroraForecaster
from aurora import AuroraPretrained

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Evaluation bounding box — all scopes are evaluated on this region.
ORIGINAL_MED_BBOX = {
    "lon_min": 18.0,
    "lon_max": 61.75,
    "lat_max": 53.25,
    "lat_min": 25.5
}

# Prediction scopes — all centered on (39.875°E, 39.375°N).
PREDICTION_SCOPES = {
    "Original": {  # 176x112 points — same as evaluation box
        "lon_min": 18.0,
        "lon_max": 61.75,
        "lat_max": 53.25,
        "lat_min": 25.5
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

def filter_batch_for_pretrained(batch):
    allowed_surf_vars = ("2t", "10u", "10v", "msl")
    allowed_static_vars = ("lsm", "z", "slt")
    allowed_atmos_vars = ("z", "u", "v", "t", "q")

    return dataclasses.replace(
        batch,
        surf_vars={k: v for k, v in batch.surf_vars.items() if k in allowed_surf_vars},
        static_vars={k: v for k, v in batch.static_vars.items() if k in allowed_static_vars},
        atmos_vars={k: v for k, v in batch.atmos_vars.items() if k in allowed_atmos_vars}
    )

def load_model(model_type, lora_weights_path=None):
    if model_type == "pretrained":
        logger.info("Loading AuroraPretrained base model...")
        model = AuroraPretrained(use_lora=False, autocast=True)
        model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", revision="main", strict=False)
    elif model_type == "finetuned":
        logger.info(f"Loading finetuned Aurora model from {lora_weights_path}...")
        model = AuroraPretrained(
            use_lora=True, 
            lora_steps=1,
            lora_mode="single",
            autocast=True 
        )
        model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", revision="main", strict=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lora_state_dict = torch.load(lora_weights_path, map_location=device)
        model.load_state_dict(lora_state_dict, strict=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model.eval()
    return model

def generate_forecasts(model_type, lora_weights_path=None, output_dir: str = "finetune_comparison_forecasts"):
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
    
    aurora_model = load_model(model_type, lora_weights_path)
    forecaster = AuroraForecaster(model=aurora_model)
    
    # Create separate DataLoaders per scope to isolate their caches
    data_pipelines = {
        scope_name: AuroraDataLoader(cache_dir=cache_dir / scope_name)
        for scope_name in PREDICTION_SCOPES.keys()
    }
    
    for init_dt in selected_dates:
        logger.info(f"--- Generating Forecasts for Init Time: {init_dt} | Model: {model_type} ---")
        init_str = init_dt.strftime("%Y%m%d_%H")
        
        for scope_name, bbox in PREDICTION_SCOPES.items():
            data_pipeline = data_pipelines[scope_name]
            out_file = Path(output_dir) / f"aurora_{model_type}_{scope_name}_{init_str}.nc"
            
            if not out_file.exists():
                logger.info(f"Running Aurora for scope: {scope_name}...")
                try:
                    input_batch, _ = data_pipeline.get_batches(init_dt, history_steps=1, forecast_steps=0, bbox=bbox)
                    input_batch = filter_batch_for_pretrained(input_batch)
                    
                    forecast_batches = forecaster.predict_rollout(
                        initial_batch=input_batch,
                        steps=int(max(lead_times_h) / 6)
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
                    logger.error(f"{model_type} {scope_name} run failed for {init_dt}: {e}")
            else:
                logger.info(f"{model_type} {scope_name} forecast for {init_dt} already exists.")
                
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def evaluate_forecasts(output_dir: str = "finetune_comparison_forecasts"):
    """Evaluates the saved forecasts against ERA5 strictly on the ORIGINAL bounding box."""
    logger.info("Starting Evaluation...")
    cache_dir = Path("weather_data")
    forecast_path = Path(output_dir)
    
    # Use the "Original" cache dir to fetch ground truth matching the original bounding box
    data_pipeline = AuroraDataLoader(cache_dir=cache_dir / "Original")
    
    lead_times_h = [6, 24, 72, 120]
    results = []
    
    # Get all unique init times from generated files
    all_files = list(forecast_path.glob("aurora_*_Original_*.nc"))
    
    if not all_files:
        logger.warning(f"No forecast files found in {output_dir}. Please run generation first.")
        return
        
    unique_inits = set()
    for f in all_files:
        match = re.search(r"(\d{8}_\d{2})\.nc", f.name)
        if match:
            unique_inits.add(match.group(1))
            
    for init_str in unique_inits:
        init_dt = datetime.strptime(init_str, "%Y%m%d_%H")
        logger.info(f"--- Evaluating Forecasts for Init Time: {init_dt} ---")
        
        # Ground Truth (ERA5 strictly on Original Med bbox)
        target_dts = [init_dt + timedelta(hours=lt) for lt in lead_times_h]
        try:
            target_ds = data_pipeline._fetch_era5_combined(target_dts, bbox=ORIGINAL_MED_BBOX)
        except Exception as e:
            logger.error(f"Failed to load target data for {init_dt}: {e}")
            continue

        for model_type in ["pretrained", "finetuned"]:
            for scope_name in PREDICTION_SCOPES.keys():
                scope_file = forecast_path / f"aurora_{model_type}_{scope_name}_{init_str}.nc"
                if not scope_file.exists():
                    logger.warning(f"Missing {model_type} {scope_name} forecast for {init_dt}, skipping.")
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
                    
                    logger.info(f"Lead time {lt_h}h | Model: {model_type}_{scope_name} | RMSE: {rmse_val:.3f}")
                    
                    results.append({
                        "init_time": init_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "valid_time": valid_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "lead_time_hours": lt_h,
                        "metric": "RMSE",
                        "model": f"{model_type}_{scope_name}",
                        "value": rmse_val
                    })
                    
                scope_ds.close()

    if results:
        df = pd.DataFrame(results)
        out_csv = "finetune_comparison_rmse.csv"
        df.to_csv(out_csv, index=False)
        logger.info(f"Saved results to {out_csv}")
    else:
        logger.warning("No results to save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Aurora Pretrained vs Finetuned")
    parser.add_argument("--generate-pretrained", action="store_true", help="Run forecast generation for pretrained model")
    parser.add_argument("--generate-finetuned", action="store_true", help="Run forecast generation for finetuned model")
    parser.add_argument("--lora-weights", type=str, default="aurora-regional-lora-6h/lora_weights.pt", help="Path to LoRA weights")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on saved forecasts")
    
    args = parser.parse_args()
    
    if not (args.generate_pretrained or args.generate_finetuned or args.evaluate):
        logger.info("No flags provided. Running full generation and evaluation pipeline.")
        generate_forecasts("pretrained")
        if not Path(args.lora_weights).exists():
            logger.error(f"LoRA weights not found at {args.lora_weights}. Please provide a valid path or run on the remote machine.")
        else:
            generate_forecasts("finetuned", args.lora_weights)
        evaluate_forecasts()
    else:
        if args.generate_pretrained:
            generate_forecasts("pretrained")
        if args.generate_finetuned:
            if not Path(args.lora_weights).exists():
                logger.error(f"LoRA weights not found at {args.lora_weights}. Please provide a valid path or run on the remote machine.")
            else:
                generate_forecasts("finetuned", args.lora_weights)
        if args.evaluate:
            evaluate_forecasts()
