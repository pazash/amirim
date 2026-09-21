"""
Plot Aurora 2m-temperature forecasts vs ERA5 ground truth for each
prediction scope and lead time.

For one event (first available init time from med_comparison_forecasts/),
the script creates a figure per lead time with N+1 panels:
    • One panel per prediction scope (Original, Enlarged_Small, Enlarged_Large, Global),
      cropped to the Original Mediterranean bounding box
    • One panel for the ERA5 ground truth on the same box

All panels share the same colour scale so differences are immediately visible.

Usage:
    python scripts/plot_scope_temperature_maps.py [--forecast-dir med_comparison_forecasts]
"""

import os
import re
import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dataloader import AuroraDataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Same definitions as compare_global_vs_med_aurora.py
# ---------------------------------------------------------------------------
ORIGINAL_MED_BBOX = {
    "lon_min": 18.0, "lon_max": 61.75,
    "lat_max": 53.25, "lat_min": 25.5,
}

PREDICTION_SCOPES = {
    "Original": {
        "lon_min": 18.0, "lon_max": 61.75,
        "lat_max": 53.25, "lat_min": 25.5,
    },
    "Enlarged_Small": {
        "lon_min": 12.0, "lon_max": 67.75,
        "lat_max": 61.25, "lat_min": 17.5,
    },
    "Enlarged_Large": {
        "lon_min": 0.0, "lon_max": 79.75,
        "lat_max": 69.25, "lat_min": 9.5,
    },
    "Global": None,
}

LEAD_TIMES_H = [6, 24, 72, 120]
SCOPE_NAMES = list(PREDICTION_SCOPES.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _discover_init_time(forecast_dir: Path) -> datetime | None:
    """Return the first init time for which ALL scopes have a forecast file."""
    pattern = re.compile(r"aurora_Original_(\d{8}_\d{2})\.nc")
    for f in sorted(forecast_dir.glob("aurora_Original_*.nc")):
        m = pattern.search(f.name)
        if not m:
            continue
        init_str = m.group(1)
        # Check that all scopes exist for this init time
        if all(
            (forecast_dir / f"aurora_{scope}_{init_str}.nc").exists()
            for scope in SCOPE_NAMES
        ):
            return datetime.strptime(init_str, "%Y%m%d_%H")
    return None


def _crop_to_original_box(da: xr.DataArray) -> xr.DataArray:
    """Crop a DataArray to the Original Mediterranean bounding box."""
    return da.sel(
        latitude=slice(ORIGINAL_MED_BBOX["lat_max"], ORIGINAL_MED_BBOX["lat_min"]),
        longitude=slice(ORIGINAL_MED_BBOX["lon_min"], ORIGINAL_MED_BBOX["lon_max"]),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(forecast_dir: str = "med_comparison_forecasts"):
    forecast_path = Path(forecast_dir)
    if not forecast_path.exists():
        logger.error(f"Forecast directory not found: {forecast_path.absolute()}")
        return

    plots_dir = Path("plots") / "scope_temperature_maps"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Pick one event ---
    init_dt = _discover_init_time(forecast_path)
    if init_dt is None:
        logger.error("No init time found with forecast files for ALL scopes.")
        return
    init_str = init_dt.strftime("%Y%m%d_%H")
    logger.info(f"Selected event: {init_dt.strftime('%Y-%m-%d %H:%M UTC')}")

    # --- Load ERA5 ground truth on the Original box ---
    cache_dir = Path("weather_data")
    data_pipeline = AuroraDataLoader(cache_dir=cache_dir)

    target_dts = [init_dt + timedelta(hours=lt) for lt in LEAD_TIMES_H]
    logger.info("Loading ERA5 ground truth …")
    target_ds = data_pipeline._fetch_era5_combined(target_dts, bbox=ORIGINAL_MED_BBOX)

    # --- Load all scope forecasts ---
    scope_datasets: dict[str, xr.Dataset] = {}
    for scope_name in SCOPE_NAMES:
        nc_path = forecast_path / f"aurora_{scope_name}_{init_str}.nc"
        scope_datasets[scope_name] = xr.open_dataset(nc_path)
        logger.info(f"Loaded forecast: {nc_path.name}")

    # --- Plot: one figure per lead time ---
    n_scopes = len(SCOPE_NAMES)
    n_cols = n_scopes + 1  # scopes + ERA5

    for lt_h in LEAD_TIMES_H:
        valid_dt = init_dt + timedelta(hours=lt_h)
        valid_str = valid_dt.strftime("%Y-%m-%d %H:%M")
        init_str_pretty = init_dt.strftime("%Y-%m-%d %H:%M")

        logger.info(f"Plotting lead time {lt_h}h …")

        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(5 * n_cols, 5),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        plt.subplots_adjust(wspace=0.25)

        # --- Collect all 2D arrays first to compute a shared colour range ---
        all_vals = []

        # ERA5
        tgt_2d = target_ds["t2m"].sel(time=valid_dt.strftime("%Y-%m-%dT%H:00:00"))
        tgt_celsius = tgt_2d.values - 273.15
        all_vals.append(tgt_celsius)

        # Each scope (cropped to Original box)
        scope_cropped: dict[str, xr.DataArray] = {}
        for scope_name in SCOPE_NAMES:
            ds = scope_datasets[scope_name]
            fcst_da = ds["2t_aurora"].sel(
                lead_time=pd.to_timedelta(f"{lt_h}h")
            ).squeeze()
            cropped = _crop_to_original_box(fcst_da)
            scope_cropped[scope_name] = cropped
            vals = cropped.values - 273.15
            all_vals.append(vals)

        combined = np.concatenate([v.ravel() for v in all_vals])
        vmin = float(np.nanmin(combined))
        vmax = float(np.nanmax(combined))
        if vmin == vmax:
            vmax = vmin + 1.0

        # --- Draw panels ---
        for i, scope_name in enumerate(SCOPE_NAMES):
            ax = axes[i]
            cropped = scope_cropped[scope_name]
            vals = cropped.values - 273.15
            im = ax.pcolormesh(
                cropped.longitude, cropped.latitude, vals,
                cmap="coolwarm", vmin=vmin, vmax=vmax,
                transform=ccrs.PlateCarree(),
            )
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.4)
            ax.set_title(f"Aurora – {scope_name}", fontsize=11)
            ax.set_extent(
                [ORIGINAL_MED_BBOX["lon_min"], ORIGINAL_MED_BBOX["lon_max"],
                 ORIGINAL_MED_BBOX["lat_min"], ORIGINAL_MED_BBOX["lat_max"]],
                crs=ccrs.PlateCarree(),
            )

        # ERA5 panel (last column)
        ax = axes[-1]
        im = ax.pcolormesh(
            tgt_2d.longitude, tgt_2d.latitude, tgt_celsius,
            cmap="coolwarm", vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.4)
        ax.set_title("ERA5 Ground Truth", fontsize=11)
        ax.set_extent(
            [ORIGINAL_MED_BBOX["lon_min"], ORIGINAL_MED_BBOX["lon_max"],
             ORIGINAL_MED_BBOX["lat_min"], ORIGINAL_MED_BBOX["lat_max"]],
            crs=ccrs.PlateCarree(),
        )

        # Shared colour bar
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.03)
        cbar.set_label("2m Temperature (°C)", fontsize=11)

        fig.suptitle(
            f"2m Temperature — Init: {init_str_pretty} UTC  |  "
            f"Valid: {valid_str} UTC  (Lead: {lt_h}h)",
            fontsize=14, y=1.02,
        )

        out_name = f"t2m_scopes_lt{lt_h:03d}h.png"
        fig.savefig(plots_dir / out_name, bbox_inches="tight", dpi=200)
        plt.close(fig)
        logger.info(f"  Saved → {plots_dir / out_name}")

    # --- Clean up ---
    for ds in scope_datasets.values():
        ds.close()

    logger.info(f"All plots saved to {plots_dir.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot Aurora temperature maps vs ERA5 for each scope and lead time"
    )
    parser.add_argument(
        "--forecast-dir",
        default="med_comparison_forecasts",
        help="Directory containing the Aurora NetCDF forecast files",
    )
    args = parser.parse_args()
    main(forecast_dir=args.forecast_dir)
