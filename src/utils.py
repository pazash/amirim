"""
Shared utility functions for the Aurora weather forecasting project.

Consolidates common functions previously duplicated across multiple scripts:
  - Verification metrics (rmse, peak_amplitude_error, etc.)
  - Data conversion (aurora_batch_to_xarray, align_and_subset_2d)
  - Remote data loaders (load_era5_zarr, load_hres_zarr)
  - Batch filtering (filter_batch_for_pretrained)
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


# ============================================================
# Verification Metrics
# ============================================================


def rmse(forecast_2d: np.ndarray, target_2d: np.ndarray) -> float:
    """Root mean squared error over spatial domain."""
    if np.isnan(target_2d).all() or np.isnan(forecast_2d).all():
        return np.nan
    diff = forecast_2d - target_2d
    return float(np.sqrt(np.nanmean(diff**2)))


def peak_amplitude_error(forecast_2d: np.ndarray, target_2d: np.ndarray) -> float:
    """max(forecast) - max(target) over spatial domain."""
    if np.isnan(target_2d).all() or np.isnan(forecast_2d).all():
        return np.nan
    return float(np.nanmax(forecast_2d) - np.nanmax(target_2d))


def conditional_bias_extremes(
    forecast_2d: np.ndarray, target_2d: np.ndarray, percentile: float = 90.0
) -> float:
    """mean(f - o) where o >= Q_percentile."""
    if np.isnan(target_2d).all():
        return np.nan
    threshold = np.nanpercentile(target_2d, percentile)
    mask = target_2d >= threshold
    if not np.any(mask):
        return np.nan
    return float(np.nanmean(forecast_2d[mask] - target_2d[mask]))


def intersection_over_union(
    forecast_2d: np.ndarray, target_2d: np.ndarray, percentile: float = 90.0
) -> float:
    """IoU of areas exceeding Q_percentile threshold."""
    if np.isnan(target_2d).all() or np.isnan(forecast_2d).all():
        return np.nan
    threshold = np.nanpercentile(target_2d, percentile)
    f_mask = forecast_2d >= threshold
    t_mask = target_2d >= threshold

    intersection = (f_mask & t_mask).sum()
    union = (f_mask | t_mask).sum()
    return float(intersection / union) if union > 0 else np.nan


def extreme_area_ratio(
    forecast_2d: np.ndarray, target_2d: np.ndarray, percentile: float = 90.0
) -> float:
    """Ratio of extreme area: count(f >= Q) / count(o >= Q)."""
    if np.isnan(target_2d).all() or np.isnan(forecast_2d).all():
        return np.nan
    threshold = np.nanpercentile(target_2d, percentile)
    t_count = (target_2d >= threshold).sum()
    if t_count == 0:
        return np.nan
    f_count = (forecast_2d >= threshold).sum()
    return float(f_count / t_count)


# ============================================================
# Data Conversion & Alignment
# ============================================================

AR_IVT_LEVELS = (300, 400, 500, 600, 700, 850, 925, 1000)


def compute_ivt_from_dataset(
    ds: xr.Dataset,
    u_var: str = "eastward_wind",
    v_var: str = "northward_wind",
    q_var: str = "specific_humidity",
    levels: Sequence[int] | None = AR_IVT_LEVELS,
) -> xr.DataArray:
    """
    Compute Integrated Vapor Transport (IVT) using ExtremeWeatherBench's standard
    trapezoid integration over pressure levels (300 to 1000 hPa) divided by g0.
    """
    from extremeweatherbench.events.atmospheric_river import integrated_vapor_transport

    if levels is not None and "level" in ds.dims:
        avail_levels = [lvl for lvl in levels if lvl in ds.level.values]
        ds = ds.sel(level=avail_levels)
    if "level" in ds.dims:
        ds = ds.sortby("level")

    ivt = integrated_vapor_transport(
        specific_humidity=ds[q_var],
        eastward_wind=ds[u_var],
        northward_wind=ds[v_var],
    )
    ivt.name = "integrated_vapor_transport"
    return ivt


def aurora_batch_to_ivt_xarray(
    predicted_batch,
    init_time,
    lead_time_hours: int,
    levels: Sequence[int] = AR_IVT_LEVELS,
) -> xr.Dataset:
    """
    Compute 2D Integrated Vapor Transport (IVT) from a single Aurora Batch prediction
    and return an xarray Dataset with dims [init_time, lead_time, latitude, longitude].
    """
    atmos_ds = aurora_batch_to_xarray(
        predicted_batch,
        init_time=init_time,
        lead_time_hours=lead_time_hours,
        include_atmos=True,
        atmos_levels=list(levels),
        atmos_vars=["u", "v", "q"],
    )
    ivt_da = compute_ivt_from_dataset(
        atmos_ds, u_var="u", v_var="v", q_var="q", levels=levels
    ).compute()

    return xr.Dataset(
        data_vars={"integrated_vapor_transport": ivt_da},
        coords={
            "init_time": atmos_ds.coords["init_time"],
            "lead_time": atmos_ds.coords["lead_time"],
            "latitude": atmos_ds.coords["latitude"],
            "longitude": atmos_ds.coords["longitude"],
        },
    )


def aurora_batch_to_xarray(
    predicted_batch,
    init_time,
    lead_time_hours: int,
    var_name: str = "2t",
    out_var_name: str = "2t_aurora",
    include_atmos: bool = False,
    atmos_levels: list[int] | tuple[int, ...] | None = None,
    atmos_vars: list[str] | tuple[str, ...] | None = None,
) -> xr.Dataset:
    """
    Convert a single Aurora Batch prediction to an xarray Dataset.

    Args:
        predicted_batch: Aurora Batch object with surf_vars, atmos_vars, and metadata.
        init_time: Forecast initialization datetime.
        lead_time_hours: Lead time in hours for this prediction.
        var_name: Key in predicted_batch.surf_vars to extract (default "2t").
        out_var_name: Variable name in the output Dataset (default "2t_aurora").
        include_atmos: If True, also extract atmospheric 3D variables.
        atmos_levels: Specific pressure levels to extract (e.g. [250, 850]). Defaults to all available.
        atmos_vars: Specific atmospheric variable keys to extract (e.g. ["t", "u", "v", "z", "q"]).

    Returns:
        xr.Dataset with dims [init_time, lead_time, (level,) latitude, longitude].
    """
    if var_name not in predicted_batch.surf_vars:
        var_name = list(predicted_batch.surf_vars.keys())[0]

    tensor = predicted_batch.surf_vars[var_name].detach().cpu().numpy().squeeze()
    lats = predicted_batch.metadata.lat
    lons = predicted_batch.metadata.lon

    if hasattr(lats, "cpu"):
        lats = lats.cpu().numpy()
    if hasattr(lons, "cpu"):
        lons = lons.cpu().numpy()

    coords = {
        "init_time": [pd.to_datetime(init_time)],
        "lead_time": pd.to_timedelta([f"{lead_time_hours}h"]),
        "latitude": lats,
        "longitude": lons,
    }

    data_vars = {
        out_var_name: (
            ["init_time", "lead_time", "latitude", "longitude"],
            tensor[np.newaxis, np.newaxis, :, :],
        ),
    }

    if out_var_name != var_name and var_name in predicted_batch.surf_vars:
        data_vars[var_name] = data_vars[out_var_name]

    if include_atmos and hasattr(predicted_batch, "atmos_vars") and predicted_batch.atmos_vars:
        sel_levels = atmos_levels

        sel_vars = atmos_vars

        if sel_levels and sel_vars:
            coords["level"] = sel_levels
            lvl_indices = [list(predicted_batch.metadata.atmos_levels).index(lvl) for lvl in sel_levels]
            for v_key in sel_vars:
                raw_atmos = predicted_batch.atmos_vars[v_key].detach().cpu().numpy().squeeze()
                # raw_atmos shape is (len(avail_levels), H, W)
                if raw_atmos.ndim == 3:
                    sliced = raw_atmos[lvl_indices]
                elif raw_atmos.ndim == 2:
                    sliced = raw_atmos[np.newaxis, :, :]
                else:
                    sliced = raw_atmos
                data_vars[v_key] = (
                    ["init_time", "lead_time", "level", "latitude", "longitude"],
                    sliced[np.newaxis, np.newaxis, :, :, :],
                )

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    return ds


def align_and_subset_2d(
    fcst_da: xr.DataArray, tgt_da: xr.DataArray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate forecast onto target grid (nearest-neighbor)."""
    fcst_sub = fcst_da.interp_like(
        tgt_da, method="nearest", kwargs={"fill_value": "extrapolate"}
    )
    return fcst_sub.values, tgt_da.values


def standardize_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Standardize dataset longitude to [0, 360] and sort."""
    if "longitude" in ds.coords:
        new_lon = ds["longitude"] % 360
        ds = ds.assign_coords(longitude=new_lon)
        ds = ds.sortby("longitude")
    return ds


# ============================================================
# Remote Data Sources
# ============================================================


def _ensure_gcs_anon():
    """Set environment variables for anonymous GCS access."""
    os.environ.setdefault("GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT", "false")
    os.environ.setdefault("GCSFS_TOKEN", "anon")


def load_era5_zarr(variable: str = "temperature") -> xr.Dataset:
    """
    Connect to ARCO ERA5 zarr store (lazy, anonymous).

    Args:
        variable: ``"temperature"`` returns ``surface_air_temperature``
                  (renamed from ``2m_temperature``).
                  ``"precipitation"`` returns ``total_precipitation``.
                  ``"atmospheric_river"`` or ``"ivt"`` returns
                  ``eastward_wind``, ``northward_wind``, ``specific_humidity``
                  on the common AR pressure levels.
    """
    _ensure_gcs_anon()
    logger.info("Connecting to ARCO ERA5 zarr store...")
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        storage_options={"token": "anon"},
    )
    if variable == "temperature":
        ds = ds.rename({"2m_temperature": "surface_air_temperature"})
        return ds[["surface_air_temperature"]]
    elif variable == "precipitation":
        return ds[["total_precipitation"]]
    elif variable in ("atmospheric_river", "ivt"):
        ds = ds.rename(
            {
                "u_component_of_wind": "eastward_wind",
                "v_component_of_wind": "northward_wind",
            }
        )
        ds = ds[["eastward_wind", "northward_wind", "specific_humidity"]]
        avail_levels = [lvl for lvl in AR_IVT_LEVELS if lvl in ds.level.values]
        return ds.sel(level=avail_levels)
    else:
        raise ValueError(
            f"Unknown variable: {variable!r}. Use 'temperature', 'precipitation', or 'atmospheric_river'."
        )


def load_hres_zarr(variable: str = "temperature") -> xr.Dataset:
    """
    Connect to HRES WeatherBench2 zarr store (lazy, anonymous).

    Args:
        variable: ``"temperature"`` returns ``surface_air_temperature``.
                  ``"precipitation"`` returns ``total_precipitation_6hr``
                  (or fallback).
                  ``"atmospheric_river"`` or ``"ivt"`` returns
                  ``eastward_wind``, ``northward_wind``, ``specific_humidity``
                  on the common AR pressure levels.
    """
    _ensure_gcs_anon()
    logger.info("Connecting to HRES WeatherBench2 zarr store...")
    ds = xr.open_zarr(
        "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
        storage_options={"token": "anon"},
    )
    ds = ds.rename({"prediction_timedelta": "lead_time", "time": "init_time"})

    if variable == "temperature":
        ds = ds.rename({"2m_temperature": "surface_air_temperature"})
        return ds[["surface_air_temperature"]]
    elif variable == "precipitation":
        if "total_precipitation_6hr" in ds.data_vars:
            return ds[["total_precipitation_6hr"]]
        elif "total_precipitation" in ds.data_vars:
            return ds[["total_precipitation"]]
        else:
            logger.warning(
                f"No obvious precip variable found in HRES. "
                f"Available: {list(ds.data_vars.keys())}"
            )
            return ds
    elif variable in ("atmospheric_river", "ivt"):
        ds = ds.rename(
            {
                "u_component_of_wind": "eastward_wind",
                "v_component_of_wind": "northward_wind",
            }
        )
        ds = ds[["eastward_wind", "northward_wind", "specific_humidity"]]
        avail_levels = [lvl for lvl in AR_IVT_LEVELS if lvl in ds.level.values]
        return ds.sel(level=avail_levels)
    else:
        raise ValueError(
            f"Unknown variable: {variable!r}. Use 'temperature', 'precipitation', or 'atmospheric_river'."
        )


# ============================================================
# Batch Filtering
# ============================================================

PRETRAINED_SURF_VARS = ("2t", "10u", "10v", "msl")
PRETRAINED_STATIC_VARS = ("lsm", "z", "slt")
PRETRAINED_ATMOS_VARS = ("z", "u", "v", "t", "q")


def filter_batch_for_pretrained(batch):
    """
    Filter an Aurora Batch to only the variables that AuroraPretrained expects.

    Removes extended surface variables, extra static fields, etc. that
    Aurora V1.5 adds but the base pretrained model doesn't support.
    """
    return dataclasses.replace(
        batch,
        surf_vars={k: v for k, v in batch.surf_vars.items() if k in PRETRAINED_SURF_VARS},
        static_vars={k: v for k, v in batch.static_vars.items() if k in PRETRAINED_STATIC_VARS},
        atmos_vars={k: v for k, v in batch.atmos_vars.items() if k in PRETRAINED_ATMOS_VARS},
    )
