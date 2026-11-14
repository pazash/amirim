"""
Compute Gradient-Based Saliency Maps on the Global 0.25° Aurora Model for Weather Prediction in Israel.

For each of the 10 evaluation dates from `compare_global_vs_med_aurora.py` and each lead time
in [6, 24, 72, 120] hours:
  1. Runs a global 0.25° AuroraV1p5 forward rollout in no_grad mode while caching detached
     intermediate step inputs on CPU.
  2. Defines the scalar target Y_Israel at the desired lead time by slicing all 5 predicted
     atmospheric variables (t, q, u, v, z across all 13 pressure levels) over the Israel
     bounding box (lat: 29.5°N–33.5°N, lon: 34.25°E–35.75°E), standardized by Aurora's
     normalization scales so all variables contribute equally.
  3. Backpropagates Y_Israel step-by-step through the autoregressive rollout back to the
     initial global input batch X_0 (exact step-checkpointed BPTT, keeping only 1 step in
     GPU memory at a time).
  4. Collapses the 4D input gradient tensor (Variables × History × Pressure Levels × Lat × Lon)
     into a 2D global spatial saliency map via the L2 norm (and Mean Absolute Gradient) across
     the atmospheric variable, pressure-level, and history dimensions.
  5. Saves per-date NetCDF saliency files and the multi-date mean saliency NetCDF.

Usage:
    python scripts/compute_israel_saliency.py [--output-dir saliency_outputs] [--num-dates 10]
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import xarray as xr

# Append project root to sys.path before importing from src/
sys.path.append(str(Path(__file__).resolve().parent.parent))
from aurora import AuroraV1p5, Batch
from aurora.normalisation import level_to_str, scales
from aurora.rollout import _advance_batch, _make_lead_time_tensor
from src.dataloader import AuroraDataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Target bounding box covering Israel on the 0.25° grid (longitudes in [0, 360))
ISRAEL_BBOX = {
    "lat_min": 29.5,
    "lat_max": 33.5,
    "lon_min": 34.25,
    "lon_max": 35.75,
}

# Same 10 seasonal evaluation dates as scripts/compare_global_vs_med_aurora.py
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

# All 5 atmospheric variables used by Aurora
ATMOS_VARS = ("t", "q", "u", "v", "z")


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _clear_caches() -> None:
    """Free Python and CUDA caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _detach_batch_to_cpu(batch: Batch) -> Batch:
    """Detach all tensors in a Batch and move them to CPU RAM."""
    return Batch(
        surf_vars={k: v.detach().cpu() for k, v in batch.surf_vars.items()},
        static_vars={k: v.detach().cpu() for k, v in batch.static_vars.items()},
        atmos_vars={k: v.detach().cpu() for k, v in batch.atmos_vars.items()},
        metadata=dataclasses.replace(
            batch.metadata,
            lat=batch.metadata.lat.detach().cpu(),
            lon=batch.metadata.lon.detach().cpu(),
        ),
    )


def _clone_batch_with_grad(batch_cpu: Batch, device: torch.device, dtype: torch.dtype) -> Batch:
    """
    Move a CPU Batch to `device` and enable `requires_grad=True` on all
    surface and atmospheric input variable tensors.
    """
    surf_vars = {
        k: v.to(device=device, dtype=dtype).detach().requires_grad_(True)
        for k, v in batch_cpu.surf_vars.items()
    }
    static_vars = {
        k: v.to(device=device, dtype=dtype).detach()
        for k, v in batch_cpu.static_vars.items()
    }
    atmos_vars = {
        k: v.to(device=device, dtype=dtype).detach().requires_grad_(True)
        for k, v in batch_cpu.atmos_vars.items()
    }
    metadata = dataclasses.replace(
        batch_cpu.metadata,
        lat=batch_cpu.metadata.lat.to(device=device),
        lon=batch_cpu.metadata.lon.to(device=device),
    )
    return Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=metadata,
    )


def _get_atmos_scale_tensor(
    var_name: str,
    atmos_levels: tuple[int | float, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return a tensor of shape `(1, 1, C, 1, 1)` containing Aurora's normalization
    scale `sigma_{v, l}` for each pressure level `l` of atmospheric variable `var_name`.
    """
    lvl_scales = [
        float(scales[f"{var_name}_{level_to_str(lvl)}"]) for lvl in atmos_levels
    ]
    return torch.tensor(lvl_scales, device=device, dtype=dtype).view(1, 1, -1, 1, 1)


def get_bbox_indices(
    lat: torch.Tensor, lon: torch.Tensor, bbox: dict[str, float], eps: float = 1e-4
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return 1D index tensors `(lat_idx, lon_idx)` for slicing `bbox`."""
    lat_mask = (lat >= bbox["lat_min"] - eps) & (lat <= bbox["lat_max"] + eps)
    lon_mask = (lon >= bbox["lon_min"] - eps) & (lon <= bbox["lon_max"] + eps)

    lat_idx = torch.where(lat_mask)[0]
    lon_idx = torch.where(lon_mask)[0]
    if lat_idx.numel() == 0 or lon_idx.numel() == 0:
        raise ValueError(
            f"Bounding box {bbox} produced an empty slice on grid "
            f"lat=[{lat.min().item()}, {lat.max().item()}], "
            f"lon=[{lon.min().item()}, {lon.max().item()}]."
        )
    return lat_idx, lon_idx


def compute_israel_target(
    pred_batch: Batch,
    israel_bbox: dict[str, float] = ISRAEL_BBOX,
    atmos_vars: tuple[str, ...] = ATMOS_VARS,
) -> torch.Tensor:
    """
    Compute the scalar target Y_Israel from a predicted Aurora Batch by:
      1. Slicing all atmospheric variables `atmos_vars` to `israel_bbox`.
      2. Dividing each variable & pressure level by its Aurora normalization scale
         so all 5 variables (t, q, u, v, z) and 13 levels contribute equally in
         dimensionless standard-deviation units.
      3. Averaging over the Israel spatial box, pressure levels, and variables.
    """
    lat_idx, lon_idx = get_bbox_indices(
        pred_batch.metadata.lat, pred_batch.metadata.lon, israel_bbox
    )
    atmos_levels = pred_batch.metadata.atmos_levels

    total_target = torch.tensor(0.0, device=pred_batch.metadata.lat.device, dtype=torch.float32)
    for var_name in atmos_vars:
        var_tensor = pred_batch.atmos_vars[var_name]  # shape: (B, 1, C, H, W)
        israel_slice = var_tensor[..., lat_idx, :][..., :, lon_idx].float()
        scale_tensor = _get_atmos_scale_tensor(
            var_name, atmos_levels, device=israel_slice.device, dtype=torch.float32
        )
        normalized_slice = israel_slice / scale_tensor
        total_target = total_target + normalized_slice.mean()

    return total_target / len(atmos_vars)


@torch.no_grad()
def collect_rollout_step_inputs(
    model: AuroraV1p5,
    initial_batch: Batch,
    max_steps: int,
) -> list[Batch]:
    """
    Run a `max_steps` forward rollout in `no_grad` mode and return the list of
    detached CPU input batches `[X_0, X_1, ..., X_{max_steps - 1}]` fed into each step.
    """
    p = next(model.parameters())
    batch = model.batch_transform_hook(initial_batch)
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    base_lead_times = (
        _make_lead_time_tensor(batch, model.timestep.total_seconds() / 3600.0)
        if model.variable_lead_time
        else None
    )

    step_inputs_cpu: list[Batch] = []
    for step_idx in range(max_steps):
        step_inputs_cpu.append(_detach_batch_to_cpu(batch))
        if step_idx < max_steps - 1:
            pred = model.forward(batch, lead_times=base_lead_times)
            pred = model.apply_rollout_input_clipping(pred)
            batch = _advance_batch(batch, pred)
            del pred

    del batch
    _clear_caches()
    return step_inputs_cpu


def backpropagate_rollout_to_input(
    model: AuroraV1p5,
    step_inputs_cpu: list[Batch],
    steps: int,
    israel_bbox: dict[str, float] = ISRAEL_BBOX,
    atmos_vars: tuple[str, ...] = ATMOS_VARS,
) -> dict[str, torch.Tensor]:
    """
    Perform exact step-checkpointed backpropagation from step `steps` back to `X_0`.

    Only a single 6-hour step's computation graph resides in GPU memory at any instant.
    Returns a dictionary mapping each atmospheric variable name in `atmos_vars` to its
    gradient tensor `dY_Israel / dX_0[var]` on CPU with shape `(1, 2, C, H, W)`.
    """
    if steps < 1 or steps > len(step_inputs_cpu):
        raise ValueError(
            f"Requested {steps} steps, but step_inputs_cpu has {len(step_inputs_cpu)} steps."
        )

    p = next(model.parameters())
    device = p.device
    dtype = p.dtype

    upstream_surf_grads: dict[str, torch.Tensor] | None = None
    upstream_atmos_grads: dict[str, torch.Tensor] | None = None

    for step_idx in range(steps - 1, -1, -1):
        batch_in = _clone_batch_with_grad(step_inputs_cpu[step_idx], device=device, dtype=dtype)
        lead_times = (
            _make_lead_time_tensor(batch_in, model.timestep.total_seconds() / 3600.0)
            if model.variable_lead_time
            else None
        )

        with torch.enable_grad():
            pred = model.forward(batch_in, lead_times=lead_times)

            if step_idx == steps - 1:
                # Final step of this lead time: compute scalar Y_Israel and backpropagate
                y_israel = compute_israel_target(
                    pred, israel_bbox=israel_bbox, atmos_vars=atmos_vars
                )
                y_israel.backward()
                del y_israel
            else:
                # Intermediate step: advance batch and propagate upstream gradients from step_idx + 1
                pred_clipped = model.apply_rollout_input_clipping(pred)
                next_batch = _advance_batch(batch_in, pred_clipped)

                tensors_to_backprop: list[torch.Tensor] = []
                grad_tensors: list[torch.Tensor] = []

                assert upstream_surf_grads is not None and upstream_atmos_grads is not None
                for k, g in upstream_surf_grads.items():
                    if k in next_batch.surf_vars and next_batch.surf_vars[k].requires_grad:
                        tensors_to_backprop.append(next_batch.surf_vars[k])
                        grad_tensors.append(g.to(device=device, dtype=next_batch.surf_vars[k].dtype))

                for k, g in upstream_atmos_grads.items():
                    if k in next_batch.atmos_vars and next_batch.atmos_vars[k].requires_grad:
                        tensors_to_backprop.append(next_batch.atmos_vars[k])
                        grad_tensors.append(g.to(device=device, dtype=next_batch.atmos_vars[k].dtype))

                torch.autograd.backward(tensors_to_backprop, grad_tensors)
                del pred_clipped, next_batch, tensors_to_backprop, grad_tensors

        del pred

        if step_idx > 0:
            # Save gradients w.r.t. batch_in on CPU to feed into step_idx - 1
            upstream_surf_grads = {
                k: (
                    v.grad.detach().cpu()
                    if v.grad is not None
                    else torch.zeros_like(v, device="cpu")
                )
                for k, v in batch_in.surf_vars.items()
            }
            upstream_atmos_grads = {
                k: (
                    v.grad.detach().cpu()
                    if v.grad is not None
                    else torch.zeros_like(v, device="cpu")
                )
                for k, v in batch_in.atmos_vars.items()
            }
            del batch_in
            _clear_caches()
        else:
            # step_idx == 0: extract final gradients w.r.t. initial atmospheric inputs X_0
            atmos_grads_x0 = {
                k: batch_in.atmos_vars[k].grad.detach().cpu().float()
                for k in atmos_vars
            }
            del batch_in, upstream_surf_grads, upstream_atmos_grads
            _clear_caches()
            return atmos_grads_x0

    raise RuntimeError("Unreachable state in backpropagate_rollout_to_input.")


def aggregate_saliency_2d(
    atmos_grads_x0: dict[str, torch.Tensor],
    atmos_levels: tuple[int | float, ...],
    atmos_vars: tuple[str, ...] = ATMOS_VARS,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Collapse the 4D input gradients `(Variables × History × Pressure Levels × Lat × Lon)`
    into 2D spatial saliency maps `(Lat, Lon)`.

    Each variable/level gradient `dY_Israel / dX_{0, v, tau, l}` is multiplied by its
    Aurora scale `sigma_{v, l}` to make the sensitivity dimensionless (`dY_Israel / dX_hat`).

    Returns:
        saliency_l2: 2D numpy array (H, W) of the L2 norm across all vars, levels, and history steps.
        saliency_mag: 2D numpy array (H, W) of the Mean Absolute Gradient across all vars, levels, and history steps.
        per_var_l2: dict mapping each variable in `atmos_vars` to its 2D L2 norm (H, W) across levels & history.
    """
    sq_sum_total: torch.Tensor | None = None
    abs_sum_total: torch.Tensor | None = None
    total_channels = 0
    per_var_l2: dict[str, np.ndarray] = {}

    for var_name in atmos_vars:
        grad = atmos_grads_x0[var_name].float()  # shape: (1, T=2, C=13, H, W)
        scale_tensor = _get_atmos_scale_tensor(
            var_name, atmos_levels, device=grad.device, dtype=torch.float32
        )
        # Dimensionless gradient w.r.t. standardized input perturbation
        norm_grad = grad * scale_tensor  # shape: (1, 2, 13, H, W)

        var_sq = (norm_grad ** 2).sum(dim=(0, 1, 2))  # shape: (H, W)
        var_abs = norm_grad.abs().sum(dim=(0, 1, 2))  # shape: (H, W)
        per_var_l2[var_name] = torch.sqrt(var_sq).cpu().numpy().astype(np.float32)

        if sq_sum_total is None:
            sq_sum_total = var_sq
            abs_sum_total = var_abs
        else:
            sq_sum_total = sq_sum_total + var_sq
            abs_sum_total = abs_sum_total + var_abs

        total_channels += norm_grad.shape[1] * norm_grad.shape[2]

    assert sq_sum_total is not None and abs_sum_total is not None
    saliency_l2 = torch.sqrt(sq_sum_total).cpu().numpy().astype(np.float32)
    saliency_mag = (abs_sum_total / max(total_channels, 1)).cpu().numpy().astype(np.float32)
    return saliency_l2, saliency_mag, per_var_l2


def build_saliency_dataset(
    init_dt: datetime,
    lead_times_h: list[int],
    lat: np.ndarray,
    lon: np.ndarray,
    l2_maps: list[np.ndarray],
    mag_maps: list[np.ndarray],
    per_var_maps: dict[str, list[np.ndarray]],
) -> xr.Dataset:
    """Create an xarray.Dataset storing the 2D saliency maps for a single init_time."""
    coords = {
        "init_time": [pd.to_datetime(init_dt)],
        "lead_time": pd.to_timedelta([f"{lt}h" for lt in lead_times_h]),
        "latitude": lat.astype(np.float32),
        "longitude": lon.astype(np.float32),
    }

    data_vars = {
        "saliency_l2": (
            ["init_time", "lead_time", "latitude", "longitude"],
            np.stack(l2_maps, axis=0)[np.newaxis, ...],
        ),
        "saliency_mag": (
            ["init_time", "lead_time", "latitude", "longitude"],
            np.stack(mag_maps, axis=0)[np.newaxis, ...],
        ),
    }

    for var_name, maps_list in per_var_maps.items():
        data_vars[f"saliency_l2_{var_name}"] = (
            ["init_time", "lead_time", "latitude", "longitude"],
            np.stack(maps_list, axis=0)[np.newaxis, ...],
        )

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["israel_bbox"] = str(ISRAEL_BBOX)
    ds.attrs["atmos_vars"] = ",".join(ATMOS_VARS)
    ds.attrs["description"] = (
        "Global 0.25-deg AuroraV1p5 gradient-based saliency maps for Israel weather prediction."
    )
    return ds


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------


def run_saliency_evaluation(
    output_dir: str = "saliency_outputs",
    cache_dir: str = "weather_data",
    num_dates: int | None = None,
) -> Path:
    """
    Compute global saliency maps for Israel across the selected evaluation dates and lead times,
    save per-date NetCDF files, and export the multi-date mean saliency dataset.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dates_to_run = SELECTED_DATES[:num_dates] if num_dates is not None else SELECTED_DATES
    max_steps = max(LEAD_TIMES_H) // 6

    # Use isolated Global cache directory per AGENTS.md guardrails
    dataloader = AuroraDataLoader(cache_dir=Path(cache_dir) / "Global", res=0.25)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading AuroraV1p5 on {device}...")
    model = AuroraV1p5()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-v1.5.ckpt", revision="main")
    model.eval()
    model.to(device)

    # Freeze all model parameters so backward() only computes input gradients
    for param in model.parameters():
        param.requires_grad_(False)

    # Enable block-level activation checkpointing to minimize single-step backward VRAM
    model.configure_activation_checkpointing()

    saved_files: list[Path] = []

    for date_idx, init_dt in enumerate(dates_to_run, start=1):
        init_str = init_dt.strftime("%Y%m%d_%H")
        date_nc_path = out_path / f"israel_saliency_{init_str}.nc"

        if date_nc_path.exists():
            logger.info(
                f"[{date_idx}/{len(dates_to_run)}] Saliency file already exists for {init_dt}: {date_nc_path.name}"
            )
            saved_files.append(date_nc_path)
            continue

        logger.info(
            f"[{date_idx}/{len(dates_to_run)}] Computing Israel Saliency for Init Time: {init_dt}..."
        )
        t0 = time.perf_counter()

        try:
            input_batch, _ = dataloader.get_batch(
                init_dt, history_steps=1, forecast_steps=0, bbox=None
            )
        except Exception as e:
            logger.error(f"Failed to load global ERA5 batch for {init_dt}: {e}", exc_info=True)
            continue

        logger.info(f"  Running {max_steps}-step global forward rollout (up to {max(LEAD_TIMES_H)}h)...")
        step_inputs_cpu = collect_rollout_step_inputs(
            model=model,
            initial_batch=input_batch,
            max_steps=max_steps,
        )
        del input_batch
        _clear_caches()

        lat_np = step_inputs_cpu[0].metadata.lat.numpy()
        lon_np = step_inputs_cpu[0].metadata.lon.numpy()
        atmos_levels = step_inputs_cpu[0].metadata.atmos_levels

        l2_maps: list[np.ndarray] = []
        mag_maps: list[np.ndarray] = []
        per_var_maps: dict[str, list[np.ndarray]] = {v: [] for v in ATMOS_VARS}

        # Run backward pass for each lead time in [6, 24, 72, 120] hours
        for lt_h in LEAD_TIMES_H:
            steps = lt_h // 6
            lt_start = time.perf_counter()
            logger.info(f"  Backpropagating lead time {lt_h:3d}h ({steps:2d} AR steps)...")

            atmos_grads_x0 = backpropagate_rollout_to_input(
                model=model,
                step_inputs_cpu=step_inputs_cpu,
                steps=steps,
                israel_bbox=ISRAEL_BBOX,
                atmos_vars=ATMOS_VARS,
            )

            saliency_l2, saliency_mag, per_var_l2 = aggregate_saliency_2d(
                atmos_grads_x0=atmos_grads_x0,
                atmos_levels=atmos_levels,
                atmos_vars=ATMOS_VARS,
            )
            del atmos_grads_x0

            l2_maps.append(saliency_l2)
            mag_maps.append(saliency_mag)
            for v in ATMOS_VARS:
                per_var_maps[v].append(per_var_l2[v])

            lt_elapsed = time.perf_counter() - lt_start
            max_idx = np.unravel_index(np.argmax(saliency_l2), saliency_l2.shape)
            peak_lat = float(lat_np[max_idx[0]])
            peak_lon = float(lon_np[max_idx[1]])
            logger.info(
                f"    ✓ Lead {lt_h:3d}h done in {lt_elapsed:.1f}s | "
                f"Peak L2 Saliency: {saliency_l2.max():.4e} at ({peak_lat:.2f}°N, {peak_lon:.2f}°E)"
            )

        del step_inputs_cpu
        _clear_caches()

        ds_date = build_saliency_dataset(
            init_dt=init_dt,
            lead_times_h=LEAD_TIMES_H,
            lat=lat_np,
            lon=lon_np,
            l2_maps=l2_maps,
            mag_maps=mag_maps,
            per_var_maps=per_var_maps,
        )
        ds_date.to_netcdf(date_nc_path, engine="h5netcdf")
        ds_date.close()
        saved_files.append(date_nc_path)

        total_elapsed = time.perf_counter() - t0
        logger.info(f"  Saved {date_nc_path.name} ({total_elapsed:.1f}s total).")

    if not saved_files:
        raise RuntimeError("No saliency files were generated or found.")

    # Aggregate across all evaluated dates to produce the mean saliency map per lead time
    mean_nc_path = out_path / "israel_saliency_mean.nc"
    logger.info(f"Computing multi-date mean saliency across {len(saved_files)} dates...")
    datasets = [xr.open_dataset(f, engine="h5netcdf") for f in saved_files]
    combined_ds = xr.concat(datasets, dim="init_time")
    mean_ds = combined_ds.mean(dim="init_time", keep_attrs=True)
    mean_ds.attrs["num_dates"] = len(saved_files)
    mean_ds.attrs["init_times"] = ",".join(
        str(pd.to_datetime(t).strftime("%Y-%m-%d %H:%M")) for t in combined_ds.init_time.values
    )
    mean_ds.to_netcdf(mean_nc_path, engine="h5netcdf")

    for ds in datasets:
        ds.close()
    combined_ds.close()
    mean_ds.close()

    logger.info(f"Saved multi-date mean saliency dataset → {mean_nc_path}")
    return mean_nc_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute global gradient-based saliency maps on Aurora for Israel weather prediction."
    )
    parser.add_argument(
        "--output-dir",
        default="saliency_outputs",
        help="Directory to store per-date and mean saliency NetCDF files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="weather_data",
        help="Base cache directory for ERA5 data.",
    )
    parser.add_argument(
        "--num-dates",
        type=int,
        default=None,
        help="Optional number of evaluation dates to process (default: all 10 dates).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Automatically generate saliency map plots after computation completes.",
    )
    args = parser.parse_args()

    mean_file = run_saliency_evaluation(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        num_dates=args.num_dates,
    )

    if args.plot:
        from scripts.plot_israel_saliency_maps import plot_all_saliency_maps

        plot_all_saliency_maps(saliency_nc=str(mean_file))

