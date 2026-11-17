"""
Compute Integrated Gradients (IG) Saliency Maps on the Global 0.25° Aurora Model for Weather Prediction in Israel.

Uses the same 10 seasonal evaluation dates, lead times [6, 24, 72, 120]h, Israel target bounding box,
and step-checkpointed backpropagation logic as `scripts/compute_israel_saliency.py`, while replacing
single-point vanilla gradients with Integrated Gradients (Sundararajan et al., 2017):

  IG(X_0) = (X_0 - X_base) * integral_0^1 grad_X Y_Israel(X_base + alpha * (X_0 - X_base)) d_alpha

Baseline X_base:
  30-year (1991–2020) ERA5 Monthly-Averaged Reanalysis by Hour of Day per pixel on the 0.25° global grid,
  matching both the calendar month and exact diurnal history timestamps (06:00 UTC and 12:00 UTC) of each
  evaluation date. Cached in `weather_data/Global_Climatology/`.

Usage:
    python scripts/compute_israel_ig_saliency.py [--ig-steps 8] [--output-dir saliency_outputs_ig] [--plot]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cdsapi
import numpy as np
import torch
import xarray as xr

# Append project root to sys.path before importing from src/ and scripts/
sys.path.append(str(Path(__file__).resolve().parent.parent))
from aurora import AuroraV1p5, Batch
from scripts.compute_israel_saliency import (
    ATMOS_VARS,
    ISRAEL_BBOX,
    LEAD_TIMES_H,
    SELECTED_DATES,
    _clear_caches,
    backpropagate_rollout_to_input,
    build_saliency_dataset,
    collect_rollout_step_inputs,
)
from src.dataloader import AuroraDataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

CLIM_START_YEAR = 1991
CLIM_END_YEAR = 2020
DEFAULT_IG_STEPS = 8


# ---------------------------------------------------------------------------
# Per-Pixel ERA5 Multi-Year Monthly-by-Hour Climatology Baseline
# ---------------------------------------------------------------------------


class ClimatologyBaselineBuilder:
    """
    Downloads (once per calendar month) and caches the 30-year (1991–2020) ERA5
    monthly-averaged reanalysis by hour of day (`00:00, 06:00, 12:00, 18:00 UTC`)
    at 0.25° global resolution, and constructs an `aurora.Batch` baseline X_base
    matching any target initialization datetime `t0_dt`.
    """

    def __init__(
        self,
        dataloader: AuroraDataLoader,
        clim_cache_dir: Path | str = Path("weather_data") / "Global_Climatology",
        start_year: int = CLIM_START_YEAR,
        end_year: int = CLIM_END_YEAR,
    ) -> None:
        self.dataloader = dataloader
        self.clim_cache_dir = Path(clim_cache_dir).expanduser()
        self.clim_cache_dir.mkdir(parents=True, exist_ok=True)
        self.start_year = start_year
        self.end_year = end_year

    def _month_clim_path(self, month: int) -> Path:
        return (
            self.clim_cache_dir
            / f"era5_clim_{self.start_year}_{self.end_year}_month_{month:02d}.nc"
        )

    @staticmethod
    def _open_cds_netcdf_or_zip(raw_path: Path) -> xr.Dataset:
        """
        Open a NetCDF file returned by CDS API, transparently handling the case where
        the new CDS infrastructure packages NetCDF file(s) inside a ZIP archive.
        """
        if zipfile.is_zipfile(raw_path):
            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(raw_path, "r") as zf:
                    zf.extractall(tmp_dir)
                nc_files = sorted(Path(tmp_dir).rglob("*.nc"))
                if not nc_files:
                    raise RuntimeError(f"No .nc files found inside ZIP archive {raw_path}")
                datasets = []
                for f in nc_files:
                    with xr.open_dataset(f, engine="netcdf4") as ds_part:
                        datasets.append(AuroraDataLoader._normalize_era5(ds_part).load())
                return xr.merge(datasets, compat="override")
        else:
            with xr.open_dataset(raw_path, engine="netcdf4") as ds_raw:
                return AuroraDataLoader._normalize_era5(ds_raw).load()

    @staticmethod
    def _average_over_years_by_hour(ds: xr.Dataset) -> xr.Dataset:
        """
        Collapse the multi-year `time` dimension into a 4-step `hour` dimension
        (0, 6, 12, 18 UTC) by averaging across all years for each hour of day.
        """
        if "expver" in ds.dims:
            ds = ds.reduce(np.nanmean, dim="expver")
        for drop_coord in ("number", "expver"):
            if drop_coord in ds.coords:
                ds = ds.drop_vars(drop_coord)

        clim_by_hour = ds.groupby("time.hour").mean(dim="time", skipna=True)
        return clim_by_hour.load()

    def ensure_month_climatology(self, month: int) -> Path:
        """
        Ensure the 30-year monthly-by-hour per-pixel climatology NetCDF for `month`
        exists on disk; download from CDS and pre-average if missing.
        """
        clim_path = self._month_clim_path(month)
        if clim_path.exists():
            return clim_path

        years = [str(y) for y in range(self.start_year, self.end_year + 1)]
        times = ["00:00", "06:00", "12:00", "18:00"]
        c = cdsapi.Client()

        raw_surf_path = (
            self.clim_cache_dir
            / f"raw_surf_clim_{self.start_year}_{self.end_year}_m{month:02d}.nc"
        )
        raw_atmos_path = (
            self.clim_cache_dir
            / f"raw_atmos_clim_{self.start_year}_{self.end_year}_m{month:02d}.nc"
        )

        surf_req = {
            "product_type": "monthly_averaged_reanalysis_by_hour_of_day",
            "variable": self.dataloader.surf_request_vars,
            "year": years,
            "month": f"{month:02d}",
            "time": times,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        atmos_req = {
            "product_type": "monthly_averaged_reanalysis_by_hour_of_day",
            "variable": self.dataloader.atmos_request_vars,
            "pressure_level": [str(lvl) for lvl in self.dataloader.levels],
            "year": years,
            "month": f"{month:02d}",
            "time": times,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }

        if not raw_surf_path.exists():
            logger.info(
                f"Downloading ERA5 {self.start_year}–{self.end_year} surface monthly-by-hour "
                f"climatology for month {month:02d}..."
            )
            c.retrieve(
                "reanalysis-era5-single-levels-monthly-means",
                surf_req,
                str(raw_surf_path),
            )

        if not raw_atmos_path.exists():
            logger.info(
                f"Downloading ERA5 {self.start_year}–{self.end_year} atmospheric monthly-by-hour "
                f"climatology for month {month:02d}..."
            )
            c.retrieve(
                "reanalysis-era5-pressure-levels-monthly-means",
                atmos_req,
                str(raw_atmos_path),
            )

        logger.info(f"Computing {self.start_year}–{self.end_year} per-pixel mean for month {month:02d}...")
        ds_surf = self._average_over_years_by_hour(self._open_cds_netcdf_or_zip(raw_surf_path))
        ds_atmos = self._average_over_years_by_hour(self._open_cds_netcdf_or_zip(raw_atmos_path))

        ds_clim = xr.merge([ds_surf, ds_atmos], compat="override")
        ds_clim.attrs["climatology_years"] = f"{self.start_year}-{self.end_year}"
        ds_clim.attrs["month"] = int(month)
        ds_clim.to_netcdf(clim_path, engine="h5netcdf")

        del ds_surf, ds_atmos, ds_clim
        # Remove temporary multi-year raw files after saving compact averaged climatology
        raw_surf_path.unlink(missing_ok=True)
        raw_atmos_path.unlink(missing_ok=True)
        logger.info(f"Saved cached per-pixel climatology → {clim_path.name}")
        return clim_path

    def get_baseline_batch(
        self,
        t0_dt: datetime,
        history_steps: int = 1,
        bbox: dict | None = None,
    ) -> Batch:
        """
        Construct the climatological baseline `Batch` X_base for `t0_dt` by loading
        the multi-year monthly-by-hour per-pixel climatology for each input timestamp
        in `[t0_dt - 6h * history_steps, ..., t0_dt]`.
        """
        input_dts = [
            t0_dt - timedelta(hours=6 * i) for i in range(history_steps, -1, -1)
        ]

        time_slices: list[xr.Dataset] = []
        for dt in input_dts:
            clim_path = self.ensure_month_climatology(dt.month)
            with xr.open_dataset(clim_path, engine="h5netcdf") as ds_month:
                ds_hour = ds_month.sel(hour=dt.hour).drop_vars("hour")
                if bbox is not None:
                    ds_hour = AuroraDataLoader._apply_bbox(ds_hour, bbox)
                ds_hour = ds_hour.expand_dims(
                    time=[dt.strftime("%Y-%m-%dT%H:00:00")]
                ).load()
                time_slices.append(ds_hour)

        clim_ds = xr.concat(time_slices, dim="time")
        baseline_batch, _ = self.dataloader._build_batch(
            clim_ds, input_dts=input_dts, target_dts=[], bbox=bbox
        )
        del clim_ds, time_slices
        return baseline_batch


# ---------------------------------------------------------------------------
# Integrated Gradients Interpolation & Aggregation
# ---------------------------------------------------------------------------


def interpolate_batch(
    baseline_batch: Batch,
    target_batch: Batch,
    alpha: float,
) -> Batch:
    """
    Construct the interpolated input batch `X_0^(alpha) = X_base + alpha * (X_0 - X_base)`.

    Prescribed astronomical `insolation` and geographical `static_vars` are kept fixed
    at their true values from `target_batch`.
    """
    interp_surf: dict[str, torch.Tensor] = {}
    for k, v_tgt in target_batch.surf_vars.items():
        if k == "insolation" or k not in baseline_batch.surf_vars:
            interp_surf[k] = v_tgt.clone()
        else:
            v_base = baseline_batch.surf_vars[k]
            interp_surf[k] = v_base + float(alpha) * (v_tgt - v_base)

    interp_atmos: dict[str, torch.Tensor] = {}
    for k, v_tgt in target_batch.atmos_vars.items():
        v_base = baseline_batch.atmos_vars[k]
        interp_atmos[k] = v_base + float(alpha) * (v_tgt - v_base)

    return Batch(
        surf_vars=interp_surf,
        static_vars=target_batch.static_vars,
        atmos_vars=interp_atmos,
        metadata=target_batch.metadata,
    )


def compute_atmos_input_delta(
    model: AuroraV1p5,
    baseline_batch: Batch,
    target_batch: Batch,
    atmos_vars: tuple[str, ...] = ATMOS_VARS,
) -> dict[str, torch.Tensor]:
    """
    Compute the cropped physical input anomaly `Delta X_0[v] = X_0[v] - X_base[v]`
    on CPU with shape `(1, 2, C, H_cropped, W_cropped)` matching `atmos_grads_x0`.
    """
    base_cropped = model.batch_transform_hook(baseline_batch).crop(model.patch_size)
    tgt_cropped = model.batch_transform_hook(target_batch).crop(model.patch_size)

    return {
        v: (tgt_cropped.atmos_vars[v].detach().cpu().float() - base_cropped.atmos_vars[v].detach().cpu().float())
        for v in atmos_vars
    }


def aggregate_ig_saliency_2d(
    avg_atmos_grads: dict[str, torch.Tensor],
    delta_atmos_x0: dict[str, torch.Tensor],
    atmos_vars: tuple[str, ...] = ATMOS_VARS,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Compute 2D Integrated Gradients spatial saliency maps `(Lat, Lon)` from the
    path-averaged gradients `avg_atmos_grads` and physical anomaly `delta_atmos_x0`:

        IG[v] = (X_0[v] - X_base[v]) * avg_grad[v]

    Because `Y_Israel` is standardized by `sigma_out` and `(X_0[v] - X_base[v]) * dY_Israel / dX_0[v]`
    cancels the physical units of `X_0[v]`, `IG[v]` is dimensionless across all variables and levels.

    Returns:
        saliency_l2: 2D numpy array (H, W) of the L2 norm across all vars, levels, and history steps.
        saliency_mag: 2D numpy array (H, W) of the Mean Absolute Attribution across all vars, levels, and history steps.
        per_var_l2: dict mapping each variable in `atmos_vars` to its 2D L2 norm (H, W).
    """
    sq_sum_total: torch.Tensor | None = None
    abs_sum_total: torch.Tensor | None = None
    total_channels = 0
    per_var_l2: dict[str, np.ndarray] = {}

    for var_name in atmos_vars:
        avg_grad = avg_atmos_grads[var_name].float()  # shape: (1, 2, 13, H, W)
        delta_x = delta_atmos_x0[var_name].float()    # shape: (1, 2, 13, H, W)

        # Elementwise Integrated Gradient attribution
        ig_tensor = delta_x * avg_grad  # shape: (1, 2, 13, H, W)

        var_sq = (ig_tensor ** 2).sum(dim=(0, 1, 2))  # shape: (H, W)
        var_abs = ig_tensor.abs().sum(dim=(0, 1, 2))  # shape: (H, W)
        per_var_l2[var_name] = torch.sqrt(var_sq).cpu().numpy().astype(np.float32)

        if sq_sum_total is None:
            sq_sum_total = var_sq
            abs_sum_total = var_abs
        else:
            sq_sum_total = sq_sum_total + var_sq
            abs_sum_total = abs_sum_total + var_abs

        total_channels += ig_tensor.shape[1] * ig_tensor.shape[2]

    assert sq_sum_total is not None and abs_sum_total is not None
    saliency_l2 = torch.sqrt(sq_sum_total).cpu().numpy().astype(np.float32)
    saliency_mag = (abs_sum_total / max(total_channels, 1)).cpu().numpy().astype(np.float32)
    return saliency_l2, saliency_mag, per_var_l2


# ---------------------------------------------------------------------------
# Main Integrated Gradients Execution Pipeline
# ---------------------------------------------------------------------------


def run_ig_saliency_evaluation(
    output_dir: str = "saliency_outputs_ig",
    cache_dir: str = "weather_data",
    ig_steps: int = DEFAULT_IG_STEPS,
    num_dates: int | None = None,
) -> Path:
    """
    Compute global Integrated Gradients saliency maps for Israel across the evaluation dates
    and lead times [6, 24, 72, 120]h using `ig_steps` midpoint quadrature steps.
    """
    if ig_steps < 1:
        raise ValueError(f"ig_steps must be >= 1, got {ig_steps}.")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dates_to_run = SELECTED_DATES[:num_dates] if num_dates is not None else SELECTED_DATES
    max_steps = max(LEAD_TIMES_H) // 6

    # Reuse the same Global ERA5 cache directory as compute_israel_saliency.py
    dataloader = AuroraDataLoader(cache_dir=Path(cache_dir) / "Global", res=0.25)
    clim_builder = ClimatologyBaselineBuilder(
        dataloader=dataloader,
        clim_cache_dir=Path(cache_dir) / "Global_Climatology",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading AuroraV1p5 on {device}...")
    model = AuroraV1p5()
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-v1.5.ckpt", revision="main")
    model.eval()
    model.to(device)

    for param in model.parameters():
        param.requires_grad_(False)
    model.configure_activation_checkpointing()

    # Midpoint Riemann quadrature points alpha_m in (0, 1)
    alphas = [(m + 0.5) / ig_steps for m in range(ig_steps)]
    logger.info(
        f"Integrated Gradients configured with M={ig_steps} midpoint steps: "
        f"alphas={[round(a, 4) for a in alphas]}"
    )

    saved_files: list[Path] = []

    for date_idx, init_dt in enumerate(dates_to_run, start=1):
        init_str = init_dt.strftime("%Y%m%d_%H")
        date_nc_path = out_path / f"israel_ig_saliency_{init_str}.nc"

        if date_nc_path.exists():
            logger.info(
                f"[{date_idx}/{len(dates_to_run)}] IG saliency file already exists for {init_dt}: {date_nc_path.name}"
            )
            saved_files.append(date_nc_path)
            continue

        logger.info(
            f"[{date_idx}/{len(dates_to_run)}] Computing Israel IG Saliency (M={ig_steps}) for Init Time: {init_dt}..."
        )
        t0 = time.perf_counter()

        try:
            target_batch, _ = dataloader.get_batch(
                init_dt, history_steps=1, forecast_steps=0, bbox=None
            )
            baseline_batch = clim_builder.get_baseline_batch(
                init_dt, history_steps=1, bbox=None
            )
        except Exception as e:
            logger.error(
                f"Failed to load target or climatology baseline batch for {init_dt}: {e}",
                exc_info=True,
            )
            continue

        delta_atmos_x0 = compute_atmos_input_delta(
            model=model,
            baseline_batch=baseline_batch,
            target_batch=target_batch,
            atmos_vars=ATMOS_VARS,
        )

        # Accumulate path gradients for each lead time across the M interpolation steps
        accum_grads_by_lt: dict[int, dict[str, torch.Tensor]] = {
            lt_h: {} for lt_h in LEAD_TIMES_H
        }
        lat_np: np.ndarray | None = None
        lon_np: np.ndarray | None = None

        for m_idx, alpha in enumerate(alphas, start=1):
            alpha_start = time.perf_counter()
            logger.info(
                f"  [Path step {m_idx}/{ig_steps} | alpha={alpha:.4f}] "
                f"Running {max_steps}-step forward rollout..."
            )
            interp_batch = interpolate_batch(baseline_batch, target_batch, alpha=alpha)
            step_inputs_cpu = collect_rollout_step_inputs(
                model=model,
                initial_batch=interp_batch,
                max_steps=max_steps,
            )
            del interp_batch
            _clear_caches()

            if lat_np is None or lon_np is None:
                lat_np = step_inputs_cpu[0].metadata.lat.numpy()
                lon_np = step_inputs_cpu[0].metadata.lon.numpy()

            for lt_h in LEAD_TIMES_H:
                steps = lt_h // 6
                grads_m = backpropagate_rollout_to_input(
                    model=model,
                    step_inputs_cpu=step_inputs_cpu,
                    steps=steps,
                    israel_bbox=ISRAEL_BBOX,
                    atmos_vars=ATMOS_VARS,
                )
                for v in ATMOS_VARS:
                    if v not in accum_grads_by_lt[lt_h]:
                        accum_grads_by_lt[lt_h][v] = grads_m[v] / float(ig_steps)
                    else:
                        accum_grads_by_lt[lt_h][v] = (
                            accum_grads_by_lt[lt_h][v] + grads_m[v] / float(ig_steps)
                        )
                del grads_m

            del step_inputs_cpu
            _clear_caches()
            logger.info(
                f"    ✓ Path step {m_idx}/{ig_steps} (alpha={alpha:.4f}) completed in "
                f"{time.perf_counter() - alpha_start:.1f}s"
            )

        del target_batch, baseline_batch
        _clear_caches()

        assert lat_np is not None and lon_np is not None
        l2_maps: list[np.ndarray] = []
        mag_maps: list[np.ndarray] = []
        per_var_maps: dict[str, list[np.ndarray]] = {v: [] for v in ATMOS_VARS}

        for lt_h in LEAD_TIMES_H:
            saliency_l2, saliency_mag, per_var_l2 = aggregate_ig_saliency_2d(
                avg_atmos_grads=accum_grads_by_lt[lt_h],
                delta_atmos_x0=delta_atmos_x0,
                atmos_vars=ATMOS_VARS,
            )
            l2_maps.append(saliency_l2)
            mag_maps.append(saliency_mag)
            for v in ATMOS_VARS:
                per_var_maps[v].append(per_var_l2[v])

            max_idx = np.unravel_index(np.argmax(saliency_l2), saliency_l2.shape)
            peak_lat = float(lat_np[max_idx[0]])
            peak_lon = float(lon_np[max_idx[1]])
            logger.info(
                f"  [IG Summary | Lead {lt_h:3d}h] "
                f"Peak IG L2: {saliency_l2.max():.4e} at ({peak_lat:.2f}°N, {peak_lon:.2f}°E)"
            )

        del accum_grads_by_lt, delta_atmos_x0
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
        ds_date.attrs["saliency_method"] = "integrated_gradients"
        ds_date.attrs["ig_steps"] = int(ig_steps)
        ds_date.attrs["baseline"] = (
            f"ERA5 {CLIM_START_YEAR}-{CLIM_END_YEAR} monthly-by-hour per-pixel climatology"
        )
        ds_date.attrs["description"] = (
            "Global 0.25-deg AuroraV1p5 Integrated Gradients saliency maps for Israel weather prediction."
        )
        ds_date.to_netcdf(date_nc_path, engine="h5netcdf")
        ds_date.close()
        saved_files.append(date_nc_path)

        logger.info(
            f"  Saved {date_nc_path.name} ({time.perf_counter() - t0:.1f}s total)."
        )

    if not saved_files:
        raise RuntimeError("No IG saliency files were generated or found.")

    mean_nc_path = out_path / "israel_ig_saliency_mean.nc"
    logger.info(f"Computing multi-date mean IG saliency across {len(saved_files)} dates...")
    datasets = [xr.open_dataset(f, engine="h5netcdf") for f in saved_files]
    combined_ds = xr.concat(datasets, dim="init_time")
    mean_ds = combined_ds.mean(dim="init_time", keep_attrs=True)
    mean_ds.attrs["num_dates"] = len(saved_files)
    mean_ds.attrs["saliency_method"] = "integrated_gradients"
    mean_ds.attrs["ig_steps"] = int(ig_steps)
    mean_ds.attrs["init_times"] = ",".join(
        str(datetime.utcfromtimestamp(t.astype("datetime64[s]").astype(int)).strftime("%Y-%m-%d %H:%M"))
        if np.issubdtype(type(t), np.datetime64)
        else str(t)
        for t in combined_ds.init_time.values
    )
    mean_ds.to_netcdf(mean_nc_path, engine="h5netcdf")

    for ds in datasets:
        ds.close()
    combined_ds.close()
    mean_ds.close()

    logger.info(f"Saved multi-date mean IG saliency dataset → {mean_nc_path}")
    return mean_nc_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute global Integrated Gradients (IG) saliency maps on Aurora for Israel weather prediction."
    )
    parser.add_argument(
        "--output-dir",
        default="saliency_outputs_ig",
        help="Directory to store per-date and mean IG saliency NetCDF files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="weather_data",
        help="Base cache directory for ERA5 data.",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=DEFAULT_IG_STEPS,
        help=f"Number of midpoint Riemann integration steps M along baseline-to-input path (default: {DEFAULT_IG_STEPS}).",
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
        help="Automatically generate IG saliency map plots (and Vanilla vs. IG comparison if available).",
    )
    args = parser.parse_args()

    mean_ig_file = run_ig_saliency_evaluation(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        ig_steps=args.ig_steps,
        num_dates=args.num_dates,
    )

    if args.plot:
        from scripts.plot_israel_saliency_maps import (
            plot_all_saliency_maps,
            plot_vanilla_vs_ig_comparison,
        )

        plot_all_saliency_maps(
            saliency_nc=str(mean_ig_file),
            plots_dir="plots/israel_ig_saliency",
        )
        vanilla_mean = Path("saliency_outputs") / "israel_saliency_mean.nc"
        if vanilla_mean.exists():
            plot_vanilla_vs_ig_comparison(
                vanilla_nc=str(vanilla_mean),
                ig_nc=str(mean_ig_file),
                plots_dir="plots/israel_ig_saliency",
            )
