"""
Compare Aurora 0.1° Global vs. Aurora 0.1° Regional with 0.25° Global Boundary Conditions.

Workflow:
  1. Uses the evaluation process, 10 seasonal dates, 4 lead times (6h, 24h, 72h, 120h),
     and evaluation variables from `compare_global_vs_med_aurora.py`.
  2. For each date, computes:
       • Global_0.1: Aurora 0.1° (AuroraHighRes) rolled out globally via `AuroraForecaster.predict_rollout`.
       • Regional_0.1_BC: Aurora 0.25° (AuroraV1p5) rolled out globally via `AuroraForecaster.predict_rollout`
         + Aurora 0.1° (AuroraHighRes) stepped regionally on `ORIGINAL_MED_BBOX` + ~1° padding
         via `AuroraForecaster.predict_step`, overriding the ~1° boundary after each step.
  3. Measures and saves the 4 requested resource metrics on the finished rollout:
       • time_sec (wall-clock time)
       • ram_usage_MiB (CPU RAM usage)
       • gpu_usage_pct (GPU utilization %)
       • gpu_memory_MiB (peak GPU memory allocated)
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Prevent CUDA caching allocator fragmentation (15+ GiB reserved-but-unallocated on L40S)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import psutil
import torch
import xarray as xr

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))
from aurora import AuroraHighRes, AuroraV1p5, Batch
from src.dataloader import AuroraDataLoader
from src.forecaster import AuroraForecaster
from src.utils import (
    align_and_subset_2d,
    aurora_batch_to_xarray,
    filter_batch_for_pretrained,
    rmse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Domain & Experiment Configuration
# ---------------------------------------------------------------------------

# Evaluation bounding box — identical to compare_global_vs_med_aurora.py
ORIGINAL_MED_BBOX = {
    "lon_min": 18.0,
    "lon_max": 61.75,
    "lat_max": 53.25,
    "lat_min": 25.5,
}

# Regional prediction bounding box: Original + ~1° padding on each side.
# Chosen so that:
#   • Padding is ~1° on all 4 sides:
#       North: 53.25° -> 54.25° (+1.00°)
#       South: 25.50° -> 24.25° (-1.25°)
#       West:  18.00° -> 16.75° (-1.25°)
#       East:  61.75° -> 62.75° (+1.00°)
#   • On the 0.1° grid, lat spans [54.2°, 24.3°] = 300 points and lon spans [16.8°, 62.7°] = 460 points,
#     both exact multiples of AuroraHighRes's patch_size = 10.
REGIONAL_PADDED_BBOX = {
    "lon_min": 16.75,
    "lon_max": 62.75,
    "lat_max": 54.25,
    "lat_min": 24.25,
}

MODEL_SCOPES = ("Global_0.1", "Regional_0.1_BC")

# 10 diverse dates across seasons from compare_global_vs_med_aurora.py
SELECTED_DATES = [
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

LEAD_TIMES_H = [6, 24, 72, 120]

# Variables to evaluate across scopes from compare_global_vs_med_aurora.py
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


# ---------------------------------------------------------------------------
# Resource Monitoring Helpers (Time, RAM, GPU %, GPU Memory)
# ---------------------------------------------------------------------------
def _get_ram_mb() -> float:
    """Return current process RAM usage (RSS) in MiB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def _get_gpu_peak_mem_mb() -> float:
    """Return peak GPU memory allocated in MiB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def _get_gpu_util_pct() -> float | None:
    """Query nvidia-smi for GPU utilization %."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
        )
        return float(out.decode().strip().split(",")[0])
    except Exception:
        return None


def _clear_caches():
    """Free CPU and GPU caches and reset peak GPU memory stats."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Boundary Condition Helpers for Regional 0.1° Rollout
# ---------------------------------------------------------------------------
class RegionalBoundaryApplicator:
    """
    Precomputes bilinear interpolation weights from the 0.25° global grid onto the
    0.1° regional grid (`REGIONAL_PADDED_BBOX`) and a boolean mask for the ~1° border
    outside `ORIGINAL_MED_BBOX`.
    """

    def __init__(
        self,
        lat_025_global: torch.Tensor,
        lon_025_global: torch.Tensor,
        lat_01_reg: torch.Tensor,
        lon_01_reg: torch.Tensor,
        inner_bbox: dict,
        device: torch.device,
        eps: float = 1e-4,
    ):
        lat_01_np = lat_01_reg.detach().cpu().numpy().astype(np.float64)
        lon_01_np = lon_01_reg.detach().cpu().numpy().astype(np.float64)

        lat_025_np = lat_025_global.detach().cpu().numpy().astype(np.float64)
        lon_025_np = lon_025_global.detach().cpu().numpy().astype(np.float64)

        # Precompute 1D linear interpolation indices & weights along latitude (decreasing)
        i1 = np.clip(
            np.searchsorted(-lat_025_np, -lat_01_np, side="right"),
            1,
            len(lat_025_np) - 1,
        )
        i0 = i1 - 1
        wy_np = (lat_025_np[i0] - lat_01_np) / (lat_025_np[i0] - lat_025_np[i1])

        # Precompute 1D linear interpolation indices & weights along longitude (increasing)
        j1 = np.clip(
            np.searchsorted(lon_025_np, lon_01_np, side="right"),
            1,
            len(lon_025_np) - 1,
        )
        j0 = j1 - 1
        wx_np = (lon_01_np - lon_025_np[j0]) / (lon_025_np[j1] - lon_025_np[j0])

        self.i0 = torch.from_numpy(i0).long().to(device)
        self.i1 = torch.from_numpy(i1).long().to(device)
        self.j0 = torch.from_numpy(j0).long().to(device)
        self.j1 = torch.from_numpy(j1).long().to(device)

        self.wy = torch.from_numpy(wy_np.astype(np.float32)).to(device).view(-1, 1)
        self.wx = torch.from_numpy(wx_np.astype(np.float32)).to(device).view(1, -1)

        # Build 2D boolean mask: True in the ~1° boundary ring, False inside ORIGINAL_MED_BBOX
        in_lat = (lat_01_np <= inner_bbox["lat_max"] + eps) & (
            lat_01_np >= inner_bbox["lat_min"] - eps
        )
        in_lon = (lon_01_np >= inner_bbox["lon_min"] - eps) & (
            lon_01_np <= inner_bbox["lon_max"] + eps
        )
        interior_mask_2d = in_lat[:, None] & in_lon[None, :]
        self.boundary_mask_2d = torch.from_numpy(~interior_mask_2d).to(
            device=device, dtype=torch.bool
        )

    def _interp(self, tensor_025: torch.Tensor) -> torch.Tensor:
        """Bilinearly interpolate a global 0.25° tensor onto the regional 0.1° grid."""
        wy = self.wy.to(dtype=tensor_025.dtype)
        wx = self.wx.to(dtype=tensor_025.dtype)
        v_lat = (1.0 - wy) * tensor_025[..., self.i0, :] + wy * tensor_025[..., self.i1, :]
        return (1.0 - wx) * v_lat[..., :, self.j0] + wx * v_lat[..., :, self.j1]

    def apply_boundary(self, pred_01: Batch, pred_025: Batch) -> Batch:
        """
        Override the ~1° extra boundary ring of `pred_01` using `pred_025`
        while keeping the interior (`ORIGINAL_MED_BBOX`) untouched.
        """
        new_surf = {}
        for k, v_01 in pred_01.surf_vars.items():
            v_025 = pred_025.surf_vars[k].to(device=v_01.device, dtype=v_01.dtype)
            v_025_interp = self._interp(v_025)
            new_surf[k] = torch.where(self.boundary_mask_2d, v_025_interp, v_01)

        new_atmos = {}
        for k, v_01 in pred_01.atmos_vars.items():
            v_025 = pred_025.atmos_vars[k].to(device=v_01.device, dtype=v_01.dtype)
            v_025_interp = self._interp(v_025)
            new_atmos[k] = torch.where(self.boundary_mask_2d, v_025_interp, v_01)

        return dataclasses.replace(pred_01, surf_vars=new_surf, atmos_vars=new_atmos)


def _advance_history_window(current_batch: Batch, pred_batch: Batch) -> Batch:
    """
    Construct the next two-step `[t-6h, t]` input batch from `current_batch` and `pred_batch`.
    """
    new_surf = {
        k: torch.cat([current_batch.surf_vars[k][:, 1:].to(v.device), v], dim=1)
        for k, v in pred_batch.surf_vars.items()
        if k in current_batch.surf_vars
    }
    new_atmos = {
        k: torch.cat([current_batch.atmos_vars[k][:, 1:].to(v.device), v], dim=1)
        for k, v in pred_batch.atmos_vars.items()
        if k in current_batch.atmos_vars
    }
    return dataclasses.replace(pred_batch, surf_vars=new_surf, atmos_vars=new_atmos)


def predict_regional_rollout_with_bc(
    forecaster_01: AuroraForecaster,
    initial_batch_01_regional: Batch,
    global_025_batches: list[Batch],
    steps: int,
) -> list[Batch]:
    """
    Roll out the regional 0.1° model step-by-step using `AuroraForecaster.predict_step`,
    overriding the ~1° boundary after each step with the corresponding 0.25° global prediction.
    """
    current_batch = initial_batch_01_regional.crop(forecaster_01.model.patch_size)

    bc_applicator = RegionalBoundaryApplicator(
        lat_025_global=global_025_batches[0].metadata.lat,
        lon_025_global=global_025_batches[0].metadata.lon,
        lat_01_reg=current_batch.metadata.lat,
        lon_01_reg=current_batch.metadata.lon,
        inner_bbox=ORIGINAL_MED_BBOX,
        device=forecaster_01.device,
    )

    predictions: list[Batch] = []
    for step_idx in range(steps):
        pred_01 = forecaster_01.predict_step(current_batch)
        pred_01 = bc_applicator.apply_boundary(pred_01, global_025_batches[step_idx])
        predictions.append(pred_01.to("cpu"))
        current_batch = _advance_history_window(current_batch, pred_01)
        del pred_01

    return predictions


def _save_forecast_netcdf(
    forecast_batches: list[Batch],
    init_dt: datetime,
    lead_times_h: list[int],
    out_file: Path,
) -> None:
    """Save forecast batches at `lead_times_h` to NetCDF, matching compare_global_vs_med_aurora.py."""
    ds_list = []
    for lt_h in lead_times_h:
        step_idx = int(lt_h / 6) - 1
        batch = forecast_batches[step_idx]
        ds_list.append(
            aurora_batch_to_xarray(
                batch,
                init_dt,
                lt_h,
                include_atmos=True,
                atmos_levels=[250, 850],
                atmos_vars=["t", "u", "v", "z", "q"],
            )
        )
    xr.concat(ds_list, dim="lead_time").to_netcdf(out_file, engine="h5netcdf")


def _is_forecast_file_valid(out_file: Path) -> bool:
    """Check whether a saved forecast NetCDF exists and contains surface & atmospheric variables."""
    if not out_file.exists():
        return False
    try:
        with xr.open_dataset(out_file) as test_ds:
            return "t" in test_ds and "u" in test_ds
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main Pipeline Functions
# ---------------------------------------------------------------------------
def generate_forecasts(
    output_dir: str = "med_01_bc_comparison_forecasts",
    resource_csv: str = "benchmark_01_global_vs_regional_bc.csv",
) -> None:
    """
    Runs Aurora predictions for each date and lead time for:
      1. Global_0.1 (Aurora 0.1° globally)
      2. Regional_0.1_BC (Aurora 0.25° globally + Aurora 0.1° regionally with BC)
    Saves NetCDF forecasts and records time, RAM usage, GPU usage %, and GPU memory usage.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path("weather_data")

    # Separate cache directories per resolution & scope to prevent any cache collisions
    loader_01_global = AuroraDataLoader(cache_dir=cache_dir / "Global_0.1", res=0.1)
    loader_025_global = AuroraDataLoader(cache_dir=cache_dir / "Global_0.25", res=0.25)
    loader_01_regional = AuroraDataLoader(cache_dir=cache_dir / "Regional_0.1", res=0.1)

    logger.info("Loading Aurora 0.1° HighRes model...")
    model_01 = AuroraHighRes(autocast=True)
    # Aurora.__init__ only sets autocast_backbone=True by default (unlike AuroraV1p5).
    # Enabling autocast on the Perceiver encoder & decoder halves the 6.43 GiB decoder
    # activations to 3.21 GiB (bfloat16) while unnormalisation still runs in float32.
    model_01.autocast_encoder = True
    model_01.autocast_decoder = True
    model_01.load_checkpoint("microsoft/aurora", "aurora-0.1-finetuned.ckpt")

    logger.info("Loading Aurora 0.25° V1.5 model...")
    model_025 = AuroraV1p5()
    model_025.load_checkpoint("microsoft/aurora", "aurora-0.25-v1.5.ckpt", revision="main")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resource_rows: list[dict] = []

    for init_dt in SELECTED_DATES:
        logger.info(f"--- Generating Forecasts for Init Time: {init_dt} ---")
        init_str = init_dt.strftime("%Y%m%d_%H")

        # ==================================================================
        # 1. Global 0.1° Model
        # ==================================================================
        out_file_g01 = out_dir / f"aurora_Global_0.1_{init_str}.nc"
        if not _is_forecast_file_valid(out_file_g01):
            logger.info("Running Global_0.1...")
            try:
                model_025.to("cpu")
                forecaster_01 = AuroraForecaster(model=model_01, device=device)

                input_batch_01_global, _ = loader_01_global.get_batch(
                    init_dt, history_steps=1, forecast_steps=0, bbox=None
                )
                input_batch_01_global = filter_batch_for_pretrained(input_batch_01_global)

                for lt_h in LEAD_TIMES_H:
                    steps = int(lt_h / 6)
                    _clear_caches()

                    t_start = time.perf_counter()
                    forecast_batches = forecaster_01.predict_rollout(
                        initial_batch=input_batch_01_global,
                        steps=steps,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed_sec = time.perf_counter() - t_start

                    ram_mb = _get_ram_mb()
                    gpu_mem_mb = _get_gpu_peak_mem_mb()
                    gpu_util = _get_gpu_util_pct()

                    resource_rows.append({
                        "init_time": init_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "lead_time_hours": lt_h,
                        "model": "Global_0.1",
                        "time_sec": round(elapsed_sec, 2),
                        "ram_usage_MiB": round(ram_mb, 1),
                        "gpu_usage_pct": gpu_util,
                        "gpu_memory_MiB": round(gpu_mem_mb, 1),
                    })
                    logger.info(
                        f"  [Global_0.1 | {lt_h}h] Time: {elapsed_sec:.2f}s | "
                        f"RAM: {ram_mb:.1f} MiB | GPU Util: {gpu_util}% | GPU Mem: {gpu_mem_mb:.1f} MiB"
                    )

                    if lt_h == max(LEAD_TIMES_H):
                        _save_forecast_netcdf(forecast_batches, init_dt, LEAD_TIMES_H, out_file_g01)

                    del forecast_batches
                    _clear_caches()

                del input_batch_01_global
            except Exception as e:
                logger.error(f"Global_0.1 run failed for {init_dt}: {e}", exc_info=True)
        else:
            logger.info(f"Global_0.1 forecast for {init_dt} already exists.")

        _clear_caches()

        # ==================================================================
        # 2. Global 0.25° + Regional 0.1° with Boundary Conditions
        # ==================================================================
        out_file_r01 = out_dir / f"aurora_Regional_0.1_BC_{init_str}.nc"
        if not _is_forecast_file_valid(out_file_r01):
            logger.info("Running Regional_0.1_BC (0.25° Global + 0.1° Regional with BC)...")
            try:
                input_batch_025_global, _ = loader_025_global.get_batch(
                    init_dt, history_steps=1, forecast_steps=0, bbox=None
                )
                input_batch_01_regional, _ = loader_01_regional.get_batch(
                    init_dt,
                    history_steps=1,
                    forecast_steps=0,
                    bbox=REGIONAL_PADDED_BBOX,
                )
                input_batch_01_regional = filter_batch_for_pretrained(input_batch_01_regional)

                for lt_h in LEAD_TIMES_H:
                    steps = int(lt_h / 6)

                    # Step A: Roll out 0.25° Global
                    model_01.to("cpu")
                    forecaster_025 = AuroraForecaster(model=model_025, device=device)
                    _clear_caches()

                    t_start_025 = time.perf_counter()
                    global_025_batches = forecaster_025.predict_rollout(
                        initial_batch=input_batch_025_global,
                        steps=steps,
                        fine_lead_times=[6.0],
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    time_025 = time.perf_counter() - t_start_025
                    ram_025 = _get_ram_mb()
                    gpu_mem_025 = _get_gpu_peak_mem_mb()
                    gpu_util_025 = _get_gpu_util_pct()

                    # Step B: Roll out 0.1° Regional with BC from 0.25° Global
                    model_025.to("cpu")
                    forecaster_01 = AuroraForecaster(model=model_01, device=device)
                    _clear_caches()

                    t_start_01 = time.perf_counter()
                    regional_01_batches = predict_regional_rollout_with_bc(
                        forecaster_01=forecaster_01,
                        initial_batch_01_regional=input_batch_01_regional,
                        global_025_batches=global_025_batches,
                        steps=steps,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    time_01 = time.perf_counter() - t_start_01
                    ram_01 = _get_ram_mb()
                    gpu_mem_01 = _get_gpu_peak_mem_mb()
                    gpu_util_01 = _get_gpu_util_pct()

                    total_time_sec = time_025 + time_01
                    peak_ram_mb = max(ram_025, ram_01)
                    peak_gpu_mem_mb = max(gpu_mem_025, gpu_mem_01)
                    gpu_utils = [u for u in (gpu_util_025, gpu_util_01) if u is not None]
                    avg_gpu_util = round(float(np.mean(gpu_utils)), 1) if gpu_utils else None

                    resource_rows.append({
                        "init_time": init_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "lead_time_hours": lt_h,
                        "model": "Regional_0.1_BC",
                        "time_sec": round(total_time_sec, 2),
                        "ram_usage_MiB": round(peak_ram_mb, 1),
                        "gpu_usage_pct": avg_gpu_util,
                        "gpu_memory_MiB": round(peak_gpu_mem_mb, 1),
                    })
                    logger.info(
                        f"  [Regional_0.1_BC | {lt_h}h] Time: {total_time_sec:.2f}s | "
                        f"RAM: {peak_ram_mb:.1f} MiB | GPU Util: {avg_gpu_util}% | "
                        f"GPU Mem: {peak_gpu_mem_mb:.1f} MiB"
                    )

                    if lt_h == max(LEAD_TIMES_H):
                        _save_forecast_netcdf(regional_01_batches, init_dt, LEAD_TIMES_H, out_file_r01)

                    del global_025_batches, regional_01_batches
                    _clear_caches()

                del input_batch_025_global, input_batch_01_regional
            except Exception as e:
                logger.error(f"Regional_0.1_BC run failed for {init_dt}: {e}", exc_info=True)
        else:
            logger.info(f"Regional_0.1_BC forecast for {init_dt} already exists.")

        _clear_caches()

        if resource_rows:
            pd.DataFrame(resource_rows).to_csv(resource_csv, index=False)

    if resource_rows:
        df_res = pd.DataFrame(resource_rows)
        df_res.to_csv(resource_csv, index=False)
        logger.info(f"Saved resource benchmark results to {resource_csv}")


def evaluate_forecasts(
    forecast_dir: str = "med_01_bc_comparison_forecasts",
    output_csv: str = "aurora_01_global_vs_regional_bc_rmse.csv",
) -> None:
    """
    Evaluates the saved forecasts against ERA5 strictly on `ORIGINAL_MED_BBOX`
    across all variables, matching `compare_global_vs_med_aurora.py`.
    """
    logger.info("Starting Evaluation...")
    cache_dir = Path("weather_data")
    forecast_path = Path(forecast_dir)

    # Use the "Original" cache dir to fetch ground truth matching the original bounding box
    data_pipeline = AuroraDataLoader(cache_dir=cache_dir / "Original")

    results: list[dict] = []

    all_files = sorted(forecast_path.glob("aurora_Global_0.1_*.nc"))
    if not all_files:
        logger.warning(f"No forecast files found in {forecast_dir}. Please run generation first.")
        return

    for global_file in all_files:
        match = re.search(r"aurora_Global_0\.1_(\d{8}_\d{2})\.nc", global_file.name)
        if not match:
            continue

        init_str = match.group(1)
        init_dt = datetime.strptime(init_str, "%Y%m%d_%H")

        logger.info(f"--- Evaluating Forecasts for Init Time: {init_dt} ---")

        # Ground Truth (ERA5 strictly on Original Med bbox)
        target_dts = [init_dt + timedelta(hours=lt) for lt in LEAD_TIMES_H]
        try:
            target_ds = data_pipeline.fetch_era5(target_dts, bbox=ORIGINAL_MED_BBOX)
        except Exception as e:
            logger.error(f"Failed to load target data for {init_dt}: {e}")
            continue

        for scope_name in MODEL_SCOPES:
            scope_file = forecast_path / f"aurora_{scope_name}_{init_str}.nc"
            if not scope_file.exists():
                logger.warning(f"Missing {scope_name} forecast for {init_dt}, skipping.")
                continue

            try:
                scope_ds = xr.open_dataset(scope_file)
            except Exception as e:
                logger.error(f"Failed to open forecast file {scope_file}: {e}")
                continue

            for lt_h in LEAD_TIMES_H:
                valid_dt = init_dt + timedelta(hours=lt_h)
                valid_dt_str = valid_dt.strftime("%Y-%m-%dT%H:00:00")
                lt_delta = pd.to_timedelta(f"{lt_h}h")

                for var_cfg in EVAL_VARIABLES:
                    var_id = var_cfg["id"]

                    # 1. Target DataArray
                    if var_cfg["type"] == "surface":
                        if var_cfg["target_key"] not in target_ds:
                            logger.warning(
                                f"Target variable {var_cfg['target_key']} missing for {valid_dt_str}"
                            )
                            continue
                        tgt_2d = target_ds[var_cfg["target_key"]].sel(time=valid_dt_str)
                    elif var_cfg["type"] == "atmos":
                        if var_cfg["target_key"] not in target_ds:
                            logger.warning(
                                f"Target variable {var_cfg['target_key']} missing for {valid_dt_str}"
                            )
                            continue
                        tgt_2d = (
                            target_ds[var_cfg["target_key"]]
                            .sel(time=valid_dt_str, level=var_cfg["level"], method="nearest")
                            .squeeze()
                        )
                    elif var_cfg["type"] == "wind_speed":
                        if "u" not in target_ds or "v" not in target_ds:
                            logger.warning(f"Target u/v missing for {valid_dt_str}")
                            continue
                        tgt_u = (
                            target_ds["u"]
                            .sel(time=valid_dt_str, level=var_cfg["level"], method="nearest")
                            .squeeze()
                        )
                        tgt_v = (
                            target_ds["v"]
                            .sel(time=valid_dt_str, level=var_cfg["level"], method="nearest")
                            .squeeze()
                        )
                        tgt_2d = np.sqrt(tgt_u ** 2 + tgt_v ** 2)

                    # 2. Forecast DataArray
                    if var_cfg["type"] == "surface":
                        fcst_key = (
                            var_cfg["fcst_key"] if var_cfg["fcst_key"] in scope_ds else "2t"
                        )
                        if fcst_key not in scope_ds:
                            logger.warning(
                                f"Forecast key {fcst_key} missing in {scope_file.name}"
                            )
                            continue
                        fcst_2d = scope_ds[fcst_key].sel(lead_time=lt_delta).squeeze()
                    elif var_cfg["type"] == "atmos":
                        fcst_key = var_cfg["fcst_key"]
                        if fcst_key not in scope_ds:
                            logger.warning(
                                f"Forecast key {fcst_key} missing in {scope_file.name}"
                            )
                            continue
                        fcst_2d = (
                            scope_ds[fcst_key]
                            .sel(lead_time=lt_delta, level=var_cfg["level"], method="nearest")
                            .squeeze()
                        )
                    elif var_cfg["type"] == "wind_speed":
                        if "u" not in scope_ds or "v" not in scope_ds:
                            logger.warning(f"Forecast u/v missing in {scope_file.name}")
                            continue
                        fcst_u = (
                            scope_ds["u"]
                            .sel(lead_time=lt_delta, level=var_cfg["level"], method="nearest")
                            .squeeze()
                        )
                        fcst_v = (
                            scope_ds["v"]
                            .sel(lead_time=lt_delta, level=var_cfg["level"], method="nearest")
                            .squeeze()
                        )
                        fcst_2d = np.sqrt(fcst_u ** 2 + fcst_v ** 2)

                    # 3. Crop to Original Med Box and compute RMSE
                    fcst_cropped = fcst_2d.sel(
                        latitude=slice(
                            ORIGINAL_MED_BBOX["lat_max"], ORIGINAL_MED_BBOX["lat_min"]
                        ),
                        longitude=slice(
                            ORIGINAL_MED_BBOX["lon_min"], ORIGINAL_MED_BBOX["lon_max"]
                        ),
                    )
                    fcst_arr, tgt_arr = align_and_subset_2d(fcst_cropped, tgt_2d)
                    rmse_val = rmse(fcst_arr, tgt_arr)

                    logger.info(
                        f"Lead time {lt_h:3d}h | Variable: {var_id:<15} | "
                        f"Model: {scope_name:<15} | RMSE: {rmse_val:.4f} {var_cfg['unit']}"
                    )

                    results.append({
                        "init_time": init_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "valid_time": valid_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "lead_time_hours": lt_h,
                        "variable": var_id,
                        "metric": "RMSE",
                        "model": f"Aurora_{scope_name}",
                        "value": rmse_val,
                    })

            scope_ds.close()

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved results to {output_csv}")
    else:
        logger.warning("No results to save.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Aurora 0.1° Global vs. Aurora 0.1° Regional with 0.25° Global BC"
    )
    parser.add_argument("--generate", action="store_true", help="Run forecast generation")
    parser.add_argument(
        "--evaluate", action="store_true", help="Run evaluation on saved forecasts"
    )

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
