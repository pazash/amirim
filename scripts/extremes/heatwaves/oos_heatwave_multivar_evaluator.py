"""
Out-of-sample multi-variable heatwave evaluation and plotting pipeline for
Aurora 0.25° Finetuned vs. ECMWF IFS HRES across whole bounding boxes and
top-10% 2m-temperature extreme areas.

Evaluated Variables (9 total)
-----------------------------
- Surface: ``2t`` (2m surface air temperature)
- Low pressure (850 hPa): ``t_850``, ``q_850``, ``z_850``, ``wind_speed_850``
- High pressure (250 hPa): ``t_250``, ``q_250``, ``z_250``, ``wind_speed_250``

Spatial Evaluation Regions
--------------------------
1. ``whole_bbox``: Full ExtremeWeatherBench heatwave ``BoundingBoxRegion``.
2. ``top10_2t_area``: Grid cells within the bounding box where ERA5 ground-truth
   2m temperature is at or above its 90th percentile at each valid timestep
   (the exact spatial mask used by ``conditional_bias_extremes``).

Verification Metrics & Outputs
------------------------------
Computes ``RMSE`` and signed ``Bias`` (``mean(forecast - ERA5)``) for both
regions and produces 3x3 multi-variable comparison grids:
  - ``whole_bbox_all_variables_rmse.png``
  - ``whole_bbox_all_variables_bias.png``
  - ``top10_2t_area_all_variables_rmse.png``
  - ``top10_2t_area_all_variables_bias.png``
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Sequence

# Set anonymous GCS access before importing extremeweatherbench/xarray/gcsfs
os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "false"
os.environ["GCSFS_TOKEN"] = "anon"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import xarray as xr
import yaml

# Suppress benign division/NaN warnings on empty/edge masks during metric computation
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="The 'generic' unit for NumPy timedelta is deprecated",
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aurora import Aurora, AuroraPretrained
from src.dataloader import AuroraDataLoader
from src.forecaster import AuroraForecaster
from src.utils import (
    align_and_subset_2d,
    aurora_batch_to_xarray,
    filter_batch_for_pretrained,
    standardize_longitude,
    timedelta_to_hours,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

sns.set_theme(style="whitegrid")

MODEL_CONFIGS: dict[str, dict[str, object]] = {
    "pretrained": {
        "checkpoint": "aurora-0.25-pretrained.ckpt",
        "cutoff_year": 2020,  # Trained 1979-2020 -> 2021+ is out-of-sample
        "label": "aurora_pretrained",
    },
    "finetuned": {
        "checkpoint": "aurora-0.25-finetuned.ckpt",
        "cutoff_year": 2021,  # Fine-tuned 2016-2021 -> 2022+ is out-of-sample
        "label": "aurora_finetuned",
    },
}

ATMOS_LEVELS: tuple[int, ...] = (250, 850)
ATMOS_VARS: tuple[str, ...] = ("t", "u", "v", "z", "q")

DEFAULT_MODELS: tuple[str, ...] = ("finetuned",)
DEFAULT_YEARS: tuple[int, ...] = (2022,)
DEFAULT_START_LEADS: tuple[int, ...] = (24, 72, 120)
DEFAULT_MAX_LEAD_HOURS: int = 240
DEFAULT_FORECAST_DIR = Path("ewb_oos_hw_multivar_forecasts")
DEFAULT_OUTPUT_CSV = Path("oos_heatwave_multivar_evaluations.csv")
DEFAULT_PLOTS_DIR = Path("plots") / "oos_heatwave_multivar"
PRETRAINED_CACHE_DIR = Path("weather_data/pretrained_global")

EVAL_VARIABLES: tuple[dict[str, object], ...] = (
    {
        "id": "2t",
        "title": "2m Surface Temperature",
        "unit": "K",
        "kind": "surface",
        "aurora_key": "2t_aurora",
        "era5_key": "2m_temperature",
    },
    {
        "id": "t_850",
        "title": "Temperature (850 hPa - Low)",
        "unit": "K",
        "kind": "atmos",
        "level": 850,
        "aurora_key": "t",
        "era5_key": "temperature",
    },
    {
        "id": "t_250",
        "title": "Temperature (250 hPa - High)",
        "unit": "K",
        "kind": "atmos",
        "level": 250,
        "aurora_key": "t",
        "era5_key": "temperature",
    },
    {
        "id": "wind_speed_850",
        "title": "Wind Speed (850 hPa - Low)",
        "unit": "m/s",
        "kind": "wind_speed",
        "level": 850,
    },
    {
        "id": "wind_speed_250",
        "title": "Wind Speed (250 hPa - High)",
        "unit": "m/s",
        "kind": "wind_speed",
        "level": 250,
    },
    {
        "id": "z_850",
        "title": "Geopotential (850 hPa - Low)",
        "unit": "m²/s²",
        "kind": "atmos",
        "level": 850,
        "aurora_key": "z",
        "era5_key": "geopotential",
    },
    {
        "id": "z_250",
        "title": "Geopotential (250 hPa - High)",
        "unit": "m²/s²",
        "kind": "atmos",
        "level": 250,
        "aurora_key": "z",
        "era5_key": "geopotential",
    },
    {
        "id": "q_850",
        "title": "Specific Humidity (850 hPa - Low)",
        "unit": "kg/kg",
        "kind": "atmos",
        "level": 850,
        "aurora_key": "q",
        "era5_key": "specific_humidity",
    },
    {
        "id": "q_250",
        "title": "Specific Humidity (250 hPa - High)",
        "unit": "kg/kg",
        "kind": "atmos",
        "level": 250,
        "aurora_key": "q",
        "era5_key": "specific_humidity",
    },
)

DEFAULT_VAR_ORDER: tuple[str, ...] = tuple(str(v["id"]) for v in EVAL_VARIABLES)
VARIABLE_METADATA: dict[str, dict[str, str]] = {
    str(v["id"]): {"title": str(v["title"]), "unit": str(v["unit"])}
    for v in EVAL_VARIABLES
}

REGION_METADATA: dict[str, dict[str, str]] = {
    "whole_bbox": {
        "title": "Whole Bounding Box",
        "bias_label": "Signed Bias (Forecast - ERA5)",
    },
    "top10_2t_area": {
        "title": "Top-10% 2t Temperature Area (Conditional Bias Hotspot)",
        "bias_label": "Conditional Bias (Forecast - ERA5)",
    },
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "aurora_pretrained": "Aurora 0.25 Pretrained",
    "aurora_finetuned": "Aurora 0.25 Finetuned",
    "HRES": "ECMWF HRES",
    "Persistence": "ERA5 Persistence",
}

MODEL_ORDER: tuple[str, ...] = (
    "Aurora 0.25 Pretrained",
    "Aurora 0.25 Finetuned",
    "ECMWF HRES",
    "ERA5 Persistence",
)

MODEL_PALETTE: dict[str, str] = {
    "Aurora 0.25 Pretrained": "#1f77b4",
    "Aurora 0.25 Finetuned": "#2ca02c",
    "ECMWF HRES": "#d62728",
    "ERA5 Persistence": "#7f7f7f",
}


class FallbackBoundingBoxRegion:
    """
    Offline-safe BoundingBoxRegion matching ``ewb.regions.BoundingBoxRegion.mask``
    so case loading works without triggering remote GCS connections on import.
    """

    def __init__(
        self,
        latitude_min: float,
        latitude_max: float,
        longitude_min: float,
        longitude_max: float,
    ) -> None:
        self.latitude_min = float(latitude_min)
        self.latitude_max = float(latitude_max)
        self.longitude_min = float(longitude_min)
        self.longitude_max = float(longitude_max)

    def mask(self, dataset: xr.Dataset) -> xr.Dataset:
        """Subset *dataset* to [latitude_min, latitude_max] and [longitude_min, longitude_max]."""
        ds_uses_360 = bool(np.any(dataset["longitude"].values > 180))
        lon_min = self.longitude_min % 360 if ds_uses_360 else self.longitude_min
        lon_max = self.longitude_max % 360 if ds_uses_360 else self.longitude_max

        lat_da = dataset.latitude.where(
            (dataset.latitude >= self.latitude_min)
            & (dataset.latitude <= self.latitude_max),
            drop=True,
        )
        if lon_min > lon_max:
            lon_da = dataset.longitude.where(
                (dataset.longitude >= lon_min) | (dataset.longitude <= lon_max),
                drop=True,
            )
        else:
            lon_da = dataset.longitude.where(
                (dataset.longitude >= lon_min) & (dataset.longitude <= lon_max),
                drop=True,
            )
        return dataset.sel(latitude=lat_da, longitude=lon_da)


@dataclass
class HeatwaveCaseRecord:
    """Case metadata container matching ExtremeWeatherBench IndividualCase attributes."""

    case_id_number: int
    title: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    location: object
    event_type: str = "heat_wave"


def align_to_synoptic_12h(dt: str | pd.Timestamp) -> pd.Timestamp:
    """
    Floor a timestamp to the nearest 00:00 or 12:00 UTC boundary.

    WeatherBench 2 HRES only initializes at 00:00 and 12:00 UTC, so snapping
    case start timestamps ensures Aurora and HRES share identical init/lead times.
    """
    return pd.to_datetime(dt).floor("12h")


def is_model_out_of_sample(model_family: str, year: int) -> bool:
    """Return True if *year* is strictly after *model_family*'s training cutoff."""
    if model_family == "aurora_pretrained":
        return year > int(MODEL_CONFIGS["pretrained"]["cutoff_year"])
    if model_family == "aurora_finetuned":
        return year > int(MODEL_CONFIGS["finetuned"]["cutoff_year"])
    if model_family in ("HRES", "Persistence"):
        return True
    raise ValueError(f"Unknown model_family: {model_family!r}")


def forecast_filename(
    output_dir: Path,
    model_key: str,
    init_time: pd.Timestamp,
    start_lead: int,
    case_id: int,
) -> Path:
    """Build an unambiguous per-model, per-case multi-variable NetCDF path."""
    return (
        Path(output_dir)
        / f"aurora_{model_key}_hw_multivar_{init_time.strftime('%Y%m%d_%H')}_L{start_lead}_case{case_id}.nc"
    )


def _find_ewb_events_yaml() -> Path | None:
    """Locate ExtremeWeatherBench's bundled events.yaml without importing ewb.defaults."""
    spec = importlib.util.find_spec("extremeweatherbench")
    if spec is not None and spec.origin is not None:
        candidate = Path(spec.origin).resolve().parent / "data" / "events.yaml"
        if candidate.exists():
            return candidate
    return None


def get_oos_heatwave_cases(
    years: Sequence[int] = DEFAULT_YEARS,
    max_cases: int | None = None,
) -> list[HeatwaveCaseRecord]:
    """
    Load ExtremeWeatherBench BoundingBoxRegion heatwave cases in *years*.

    Reads EWB's bundled ``events.yaml`` directly to avoid triggering eager GCS
    connections on import, falling back to ``ewb.cases.load_cases()`` if needed.
    """
    target_years = {int(y) for y in years}
    yaml_path = _find_ewb_events_yaml()
    cases: list[HeatwaveCaseRecord] = []

    if yaml_path is not None:
        with yaml_path.open("r", encoding="utf-8") as f:
            raw_events = yaml.safe_load(f) or []
        for entry in raw_events:
            if entry.get("event_type") != "heat_wave":
                continue
            loc = entry.get("location", {})
            if loc.get("type") != "bounded_region":
                continue
            start_dt = pd.to_datetime(entry["start_date"])
            if start_dt.year not in target_years:
                continue
            params = loc["parameters"]
            lat_min = float(params["latitude_min"])
            lat_max = float(params["latitude_max"])
            lon_min = float(params["longitude_min"])
            lon_max = float(params["longitude_max"])
            cases.append(
                HeatwaveCaseRecord(
                    case_id_number=int(entry["case_id_number"]),
                    title=str(entry["title"]),
                    start_date=start_dt,
                    end_date=pd.to_datetime(entry["end_date"]),
                    lat_min=lat_min,
                    lat_max=lat_max,
                    lon_min=lon_min,
                    lon_max=lon_max,
                    location=FallbackBoundingBoxRegion(
                        latitude_min=lat_min,
                        latitude_max=lat_max,
                        longitude_min=lon_min,
                        longitude_max=lon_max,
                    ),
                )
            )
    else:
        import extremeweatherbench as ewb

        for c in ewb.cases.load_cases():
            if (
                c.event_type == "heat_wave"
                and isinstance(c.location, ewb.regions.BoundingBoxRegion)
                and pd.to_datetime(c.start_date).year in target_years
            ):
                bbox = c.location.as_geopandas().total_bounds
                cases.append(
                    HeatwaveCaseRecord(
                        case_id_number=int(c.case_id_number),
                        title=str(c.title),
                        start_date=pd.to_datetime(c.start_date),
                        end_date=pd.to_datetime(c.end_date),
                        lat_min=float(bbox[1]),
                        lat_max=float(bbox[3]),
                        lon_min=float(bbox[0]),
                        lon_max=float(bbox[2]),
                        location=c.location,
                    )
                )

    cases.sort(key=lambda c: (pd.to_datetime(c.start_date), c.case_id_number))
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


def load_aurora_025_model(model_key: str):
    """Instantiate and load the requested 0.25-degree Aurora checkpoint."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_key {model_key!r}. Expected one of {tuple(MODEL_CONFIGS)}."
        )
    ckpt_name = str(MODEL_CONFIGS[model_key]["checkpoint"])
    if model_key == "pretrained":
        model = AuroraPretrained(use_lora=False, autocast=True)
        model.load_checkpoint(
            "microsoft/aurora", ckpt_name, revision="main", strict=False
        )
    else:
        model = Aurora(use_lora=True, autocast=True)
        model.load_checkpoint(
            "microsoft/aurora", ckpt_name, revision="main", strict=True
        )
    model.eval()
    return model


def _is_valid_multivar_forecast(nc_path: Path) -> bool:
    """Return True if *nc_path* exists and holds surface 2t plus 250/850 hPa vars."""
    if not nc_path.exists():
        return False
    try:
        with xr.open_dataset(nc_path) as ds:
            has_2t = ("2t_aurora" in ds.data_vars) or ("2t" in ds.data_vars)
            has_atmos = all(v in ds.data_vars for v in ATMOS_VARS)
            has_levels = "level" in ds.coords and all(
                lvl in ds.level.values for lvl in ATMOS_LEVELS
            )
            has_leads = "lead_time" in ds.dims and ds.sizes.get("lead_time", 0) > 0
            return bool(has_2t and has_atmos and has_levels and has_leads)
    except Exception:
        return False


# =============================================================================
# Cloud Zarr Loaders & Field Helpers
# =============================================================================

REANALYSIS_VARS: tuple[str, ...] = (
    "2m_temperature",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
)


def load_multivar_era5_zarr(levels: Sequence[int] = ATMOS_LEVELS) -> xr.Dataset:
    """Connect lazily to ARCO ERA5 Zarr for 2m temperature and upper-air fields."""
    logger.info("Connecting to ARCO ERA5 zarr store (multi-variable)...")
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        storage_options={"token": "anon"},
        chunks=None,
    )
    ds = ds[list(REANALYSIS_VARS)]
    avail_levels = [lvl for lvl in levels if lvl in ds.level.values]
    return ds.sel(level=avail_levels)


def load_multivar_hres_zarr(levels: Sequence[int] = ATMOS_LEVELS) -> xr.Dataset:
    """Connect lazily to WeatherBench2 HRES Zarr for 2m temperature and upper-air fields."""
    logger.info("Connecting to HRES WeatherBench2 zarr store (multi-variable)...")
    ds = xr.open_zarr(
        "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
        storage_options={"token": "anon"},
        chunks=None,
    )
    ds = ds.rename({"prediction_timedelta": "lead_time", "time": "init_time"})
    ds = ds[list(REANALYSIS_VARS)]
    avail_levels = [lvl for lvl in levels if lvl in ds.level.values]
    return ds.sel(level=avail_levels)


def _select_hres_at_lead(hres_fcst: xr.Dataset, lt_hours: int) -> xr.Dataset:
    """Select a single lead-time slice from HRES handling int or timedelta64 coords."""
    if np.issubdtype(hres_fcst.lead_time.dtype, np.timedelta64):
        return hres_fcst.sel(lead_time=pd.to_timedelta(lt_hours, unit="h"))
    return hres_fcst.sel(lead_time=lt_hours)


def _extract_variable_2d(
    ds: xr.Dataset,
    var_cfg: dict[str, object],
    source_kind: str,
) -> xr.DataArray:
    """
    Extract a 2D spatial DataArray (ordered ``latitude, longitude``) for
    *var_cfg* from a time/lead-sliced Dataset.

    ``source_kind`` must be ``"aurora"`` or ``"reanalysis"`` (ERA5 / HRES).
    """
    kind = str(var_cfg["kind"])

    if kind == "surface":
        if source_kind == "aurora":
            key = "2t_aurora" if "2t_aurora" in ds.data_vars else "2t"
        else:
            key = (
                "2m_temperature"
                if "2m_temperature" in ds.data_vars
                else "surface_air_temperature"
            )
        da = ds[key].squeeze(drop=True)
    elif kind == "atmos":
        level = int(var_cfg["level"])
        key = str(
            var_cfg["aurora_key"] if source_kind == "aurora" else var_cfg["era5_key"]
        )
        da = ds[key].sel(level=level).squeeze(drop=True)
    elif kind == "wind_speed":
        level = int(var_cfg["level"])
        u_key, v_key = (
            ("u", "v")
            if source_kind == "aurora"
            else ("u_component_of_wind", "v_component_of_wind")
        )
        u_da = ds[u_key].sel(level=level).squeeze(drop=True)
        v_da = ds[v_key].sel(level=level).squeeze(drop=True)
        da = np.hypot(u_da, v_da)
    else:
        raise ValueError(f"Unsupported variable kind: {kind!r}")

    return da.transpose("latitude", "longitude")


def compute_region_metrics(
    fcst_2d: np.ndarray,
    tgt_2d: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute RMSE and signed Bias (forecast - target) on full or masked grid cells."""
    if mask is not None:
        f = fcst_2d[mask]
        o = tgt_2d[mask]
    else:
        f = fcst_2d
        o = tgt_2d

    if f.size == 0 or np.isnan(f).all() or np.isnan(o).all():
        return {"RMSE": np.nan, "Bias": np.nan}

    diff = f - o
    if np.isnan(diff).all():
        return {"RMSE": np.nan, "Bias": np.nan}

    return {
        "RMSE": float(np.sqrt(np.nanmean(diff**2))),
        "Bias": float(np.nanmean(diff)),
    }


# =============================================================================
# Section 1: Multi-Variable Forecast Generation
# =============================================================================


def generate_oos_heatwave_multivar_forecasts(
    output_dir: str | Path = DEFAULT_FORECAST_DIR,
    models: Sequence[str] = DEFAULT_MODELS,
    years: Sequence[int] = DEFAULT_YEARS,
    start_leads: Sequence[int] = DEFAULT_START_LEADS,
    cache_dir: str | Path = PRETRAINED_CACHE_DIR,
    max_cases: int | None = None,
) -> None:
    """
    Generate Aurora 0.25° forecasts containing 2t and 250/850 hPa (t, q, z, u, v)
    for EWB heatwaves in *years* at *start_leads*.

    Groups pending forecasts by ``(model_key, init_time)`` to share GPU rollouts
    while masking each case output to its bounding box to minimize RAM and disk usage.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cases = get_oos_heatwave_cases(years=years, max_cases=max_cases)
    logger.info(
        f"Loaded {len(cases)} EWB heatwave cases for years={list(years)}."
    )
    if not cases:
        logger.warning("No matching heatwave cases found. Exiting generation.")
        return

    pending_by_model: dict[str, dict[pd.Timestamp, list[dict]]] = {}
    for model_key in models:
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model key: {model_key!r}")
        grouped_by_init: dict[pd.Timestamp, list[dict]] = {}
        for case in cases:
            aligned_start = align_to_synoptic_12h(case.start_date)
            event_end = pd.to_datetime(case.end_date)
            for start_lead in start_leads:
                init_time = aligned_start - timedelta(hours=int(start_lead))
                out_file = forecast_filename(
                    output_path,
                    model_key,
                    init_time,
                    int(start_lead),
                    case.case_id_number,
                )
                if _is_valid_multivar_forecast(out_file):
                    logger.info(
                        f"[{model_key}] Skipping Case {case.case_id_number} L{start_lead} "
                        f"({out_file.name} exists with atmos vars)"
                    )
                    continue
                grouped_by_init.setdefault(init_time, []).append(
                    {
                        "case": case,
                        "aligned_start": aligned_start,
                        "event_end": event_end,
                        "start_lead": int(start_lead),
                        "filename": out_file,
                    }
                )
        if grouped_by_init:
            pending_by_model[model_key] = grouped_by_init

    if not pending_by_model:
        logger.info("All requested multi-variable Aurora forecasts already exist.")
        return

    data_pipeline = AuroraDataLoader(
        cache_dir=Path(cache_dir), pretrained_only=True, res=0.25
    )

    for model_key, grouped_by_init in pending_by_model.items():
        logger.info("=" * 60)
        logger.info(
            f"Generating multi-var forecasts for model={model_key} "
            f"({MODEL_CONFIGS[model_key]['checkpoint']}) across "
            f"{len(grouped_by_init)} unique init times"
        )
        logger.info("=" * 60)

        aurora_model = load_aurora_025_model(model_key)
        forecaster = AuroraForecaster(model=aurora_model)

        for init_time in sorted(grouped_by_init.keys()):
            jobs = grouped_by_init[init_time]
            max_end = max(job["event_end"] for job in jobs)
            total_steps_needed = max(
                1, int(math.ceil((max_end - init_time).total_seconds() / 21600.0))
            )
            case_desc = ", ".join(
                f"Case {j['case'].case_id_number}(L{j['start_lead']})" for j in jobs
            )
            logger.info(
                f"[{model_key}] Init={init_time} | Steps={total_steps_needed} "
                f"({total_steps_needed * 6}h) | Targets: {case_desc}"
            )

            try:
                input_batch, _ = data_pipeline.get_batch(
                    init_time.to_pydatetime(),
                    history_steps=1,
                    forecast_steps=0,
                    bbox=None,
                )
                input_batch = filter_batch_for_pretrained(input_batch)

                active_buffers: dict[int, list[xr.Dataset]] = {
                    idx: [] for idx in range(len(jobs))
                }
                completed_jobs: set[int] = set()

                for step_idx, batch in enumerate(
                    forecaster.stream_rollout(
                        initial_batch=input_batch,
                        steps=total_steps_needed,
                        fine_lead_times=None,
                    )
                ):
                    lead_hours = (step_idx + 1) * 6
                    valid_time = init_time + timedelta(hours=lead_hours)

                    matching_indices = [
                        idx
                        for idx, job in enumerate(jobs)
                        if idx not in completed_jobs
                        and job["aligned_start"] <= valid_time <= job["event_end"]
                    ]
                    if matching_indices:
                        step_ds = aurora_batch_to_xarray(
                            batch,
                            init_time=init_time.to_pydatetime(),
                            lead_time_hours=lead_hours,
                            var_name="2t",
                            out_var_name="2t_aurora",
                            include_atmos=True,
                            atmos_levels=list(ATMOS_LEVELS),
                            atmos_vars=list(ATMOS_VARS),
                        )
                        for idx in matching_indices:
                            # Crop per case before buffering to keep RAM and NetCDF files small
                            cropped_ds = standardize_longitude(
                                jobs[idx]["case"].location.mask(step_ds)
                            )
                            active_buffers[idx].append(cropped_ds)
                        del step_ds

                    for idx, job in enumerate(jobs):
                        if idx not in completed_jobs and valid_time >= job["event_end"]:
                            buf = active_buffers.pop(idx, [])
                            if buf:
                                full_ds = xr.concat(buf, dim="lead_time")
                                full_ds.to_netcdf(job["filename"], engine="h5netcdf")
                                logger.info(f"Saved {job['filename']}")
                                del full_ds
                            completed_jobs.add(idx)

                    del batch

                for idx, buf in active_buffers.items():
                    if buf:
                        job = jobs[idx]
                        full_ds = xr.concat(buf, dim="lead_time")
                        full_ds.to_netcdf(job["filename"], engine="h5netcdf")
                        logger.info(f"Saved {job['filename']}")
                        del full_ds

                del input_batch, active_buffers
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as exc:
                logger.error(
                    f"[{model_key}] Failed rollout for init_time={init_time}: {exc}"
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info(f"Releasing GPU resources for model={model_key}...")
        del aurora_model
        del forecaster
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# =============================================================================
# Section 2: Multi-Variable Region Evaluation
# =============================================================================


def _append_multivar_rows(
    results: list[dict],
    metrics: dict[str, float],
    *,
    case_id: int,
    variable_id: str,
    region: str,
    forecast_source: str,
    model_family: str,
    init_time: pd.Timestamp,
    valid_time: pd.Timestamp,
    lead_hours: int,
    year: int,
) -> None:
    """Append RMSE and Bias metric rows for one (variable, region, model) slice."""
    oos_flag = is_model_out_of_sample(model_family, year)
    for metric_name, val in metrics.items():
        results.append(
            {
                "value": val,
                "lead_time": lead_hours,
                "init_time": init_time,
                "valid_time": valid_time,
                "variable": variable_id,
                "region": region,
                "metric": metric_name,
                "forecast_source": forecast_source,
                "target_source": "ERA5",
                "case_id_number": case_id,
                "event_type": "heat_wave",
                "model_family": model_family,
                "year": year,
                "is_out_of_sample": oos_flag,
            }
        )


def _score_dataset_step(
    results: list[dict],
    fcst_step: xr.Dataset,
    era5_step: xr.Dataset,
    top10_2t_mask: np.ndarray,
    *,
    case_id: int,
    forecast_source: str,
    model_family: str,
    init_time: pd.Timestamp,
    valid_time: pd.Timestamp,
    lead_hours: int,
    year: int,
    fcst_source_kind: str,
) -> None:
    """Score all 9 variables on whole_bbox and top10_2t_area for a single step."""
    for var_cfg in EVAL_VARIABLES:
        var_id = str(var_cfg["id"])
        tgt_da = _extract_variable_2d(era5_step, var_cfg, source_kind="reanalysis")
        fcst_da = _extract_variable_2d(fcst_step, var_cfg, source_kind=fcst_source_kind)
        fcst_arr, tgt_arr = align_and_subset_2d(fcst_da, tgt_da)

        for region_name, mask in (
            ("whole_bbox", None),
            ("top10_2t_area", top10_2t_mask),
        ):
            metrics = compute_region_metrics(fcst_arr, tgt_arr, mask=mask)
            _append_multivar_rows(
                results,
                metrics,
                case_id=case_id,
                variable_id=var_id,
                region=region_name,
                forecast_source=forecast_source,
                model_family=model_family,
                init_time=init_time,
                valid_time=valid_time,
                lead_hours=lead_hours,
                year=year,
            )


def print_evaluation_summaries(df: pd.DataFrame) -> None:
    """Print common-target out-of-sample summaries by region, metric, and variable."""
    if df.empty:
        return

    df_valid = df.dropna(subset=["value"]).copy()
    df_valid = df_valid[df_valid["lead_time"] <= DEFAULT_MAX_LEAD_HOURS]
    if "is_out_of_sample" in df_valid.columns:
        df_valid = df_valid[df_valid["is_out_of_sample"] == True]  # noqa: E712
    if df_valid.empty:
        return

    families = sorted(df_valid["model_family"].unique())
    key_cols = ["case_id_number", "init_time", "valid_time"]
    common = None
    for fam in families:
        keys = df_valid[df_valid["model_family"] == fam][key_cols].drop_duplicates()
        common = (
            keys
            if common is None
            else pd.merge(common, keys, on=key_cols, how="inner")
        )

    if common is not None and not common.empty:
        df_common = pd.merge(df_valid, common, on=key_cols, how="inner")
        summary = (
            df_common.groupby(["region", "metric", "variable", "model_family"])["value"]
            .agg(["count", "mean", "std"])
        )
        logger.info(
            "\n=== Out-of-Sample Multi-Variable Summary (<= 240h, Common Targets) ===\n"
            + summary.to_string()
        )


def evaluate_oos_heatwave_multivar(
    forecast_dir: str | Path = DEFAULT_FORECAST_DIR,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    models: Sequence[str] = DEFAULT_MODELS,
    years: Sequence[int] = DEFAULT_YEARS,
    start_leads: Sequence[int] = DEFAULT_START_LEADS,
    max_cases: int | None = None,
    include_persistence: bool = False,
) -> pd.DataFrame:
    """
    Evaluate saved Aurora 0.25° multi-variable forecasts against ECMWF HRES
    (and optionally ERA5 Persistence) using ARCO ERA5 as ground truth on both
    ``whole_bbox`` and ``top10_2t_area``.
    """
    forecast_path = Path(forecast_dir)
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"Forecast directory {forecast_path.resolve()} does not exist. "
            "Run with --generate first."
        )

    logger.info("=" * 60)
    logger.info("Starting Multi-Variable Out-of-Sample Heatwave Evaluation")
    logger.info("=" * 60)

    era5_ds = load_multivar_era5_zarr()
    hres_ds = load_multivar_hres_zarr()
    cases = get_oos_heatwave_cases(years=years, max_cases=max_cases)
    logger.info(
        f"Evaluating {len(cases)} EWB heatwave cases for years={list(years)}."
    )

    results: list[dict] = []
    evaluated_hres_keys: set[tuple[int, pd.Timestamp, pd.Timestamp]] = set()
    evaluated_persist_keys: set[tuple[int, pd.Timestamp, pd.Timestamp]] = set()

    for case_idx, case in enumerate(cases):
        aligned_start = align_to_synoptic_12h(case.start_date)
        end_dt = pd.to_datetime(case.end_date)
        case_year = int(aligned_start.year)

        logger.info(
            f"\nEvaluating Case {case.case_id_number}: {case.title} "
            f"({case_idx + 1}/{len(cases)}) | {aligned_start} -> {end_dt}"
        )

        era5_case_raw = era5_ds.sel(
            time=slice(aligned_start.to_pydatetime(), end_dt.to_pydatetime())
        )
        six_hr_mask = era5_case_raw.time.dt.hour.isin([0, 6, 12, 18])
        era5_case_raw = era5_case_raw.isel(time=np.where(six_hr_mask.values)[0])
        era5_case = standardize_longitude(case.location.mask(era5_case_raw).compute())
        del era5_case_raw

        for start_lead in start_leads:
            start_lead = int(start_lead)
            init_t = aligned_start - timedelta(hours=start_lead)

            model_files: list[tuple[str, Path]] = []
            for model_key in models:
                nc_path = forecast_filename(
                    forecast_path, model_key, init_t, start_lead, case.case_id_number
                )
                if _is_valid_multivar_forecast(nc_path):
                    model_files.append((model_key, nc_path))
                else:
                    logger.warning(
                        f"  Missing or incomplete multivar forecast for Case {case.case_id_number} "
                        f"({model_key}, L{start_lead}): {nc_path.name}"
                    )

            if not model_files:
                continue

            persist_ds = None
            if include_persistence:
                try:
                    era5_init_raw = era5_ds.sel(time=init_t.to_pydatetime())
                    persist_ds = standardize_longitude(
                        case.location.mask(era5_init_raw).compute()
                    )
                    del era5_init_raw
                except Exception as exc:
                    logger.warning(
                        f"  Failed loading ERA5 persistence at init_t={init_t}: {exc}"
                    )

            hres_fcst = None
            try:
                hres_raw = hres_ds.sel(init_time=init_t.to_datetime64())
                if np.issubdtype(hres_raw.lead_time.dtype, np.timedelta64):
                    hres_lead_td = pd.to_timedelta(hres_raw.lead_time.values)
                else:
                    hres_lead_td = pd.to_timedelta(hres_raw.lead_time.values, unit="h")
                hres_valid_times = init_t + hres_lead_td
                in_window = np.asarray(
                    (hres_valid_times >= aligned_start) & (hres_valid_times <= end_dt)
                )
                if np.any(in_window):
                    hres_slice = hres_raw.isel(lead_time=np.where(in_window)[0])
                    hres_fcst = standardize_longitude(
                        case.location.mask(hres_slice).compute()
                    )
                else:
                    logger.warning(
                        f"  HRES has no lead times in [{aligned_start}, {end_dt}] "
                        f"for init_time={init_t}"
                    )
            except (KeyError, IndexError) as exc:
                logger.warning(
                    f"  No exact HRES init_time match for {init_t}: {exc}"
                )

            for model_key, nc_path in model_files:
                model_family = str(MODEL_CONFIGS[model_key]["label"])
                forecast_source = f"{model_family}_L{start_lead}"

                with xr.open_dataset(nc_path) as aurora_raw:
                    aurora_fcst = standardize_longitude(
                        case.location.mask(aurora_raw).compute()
                    )
                file_init_t = pd.to_datetime(aurora_fcst.init_time.values[0])

                for lt in aurora_fcst.lead_time.values:
                    lt_hours = timedelta_to_hours(lt)
                    if lt_hours > DEFAULT_MAX_LEAD_HOURS:
                        continue
                    valid_time = file_init_t + timedelta(hours=lt_hours)
                    if not (aligned_start <= valid_time <= end_dt):
                        continue

                    try:
                        era5_step = era5_case.sel(time=valid_time.to_pydatetime())
                    except KeyError:
                        continue

                    tgt_2t_vals = (
                        era5_step["2m_temperature"]
                        .squeeze(drop=True)
                        .transpose("latitude", "longitude")
                        .values
                    )
                    if np.isnan(tgt_2t_vals).all():
                        continue
                    q90_2t = float(np.nanpercentile(tgt_2t_vals, 90.0))
                    top10_2t_mask = (tgt_2t_vals >= q90_2t) & np.isfinite(tgt_2t_vals)
                    if not np.any(top10_2t_mask):
                        continue

                    aurora_step = aurora_fcst.sel(
                        init_time=aurora_fcst.init_time.values[0], lead_time=lt
                    )
                    _score_dataset_step(
                        results,
                        aurora_step,
                        era5_step,
                        top10_2t_mask,
                        case_id=case.case_id_number,
                        forecast_source=forecast_source,
                        model_family=model_family,
                        init_time=file_init_t,
                        valid_time=valid_time,
                        lead_hours=lt_hours,
                        year=case_year,
                        fcst_source_kind="aurora",
                    )

                    baseline_key = (case.case_id_number, init_t, valid_time)
                    if hres_fcst is not None and baseline_key not in evaluated_hres_keys:
                        try:
                            hres_step = _select_hres_at_lead(hres_fcst, lt_hours)
                        except KeyError:
                            pass
                        else:
                            _score_dataset_step(
                                results,
                                hres_step,
                                era5_step,
                                top10_2t_mask,
                                case_id=case.case_id_number,
                                forecast_source="HRES",
                                model_family="HRES",
                                init_time=init_t,
                                valid_time=valid_time,
                                lead_hours=lt_hours,
                                year=case_year,
                                fcst_source_kind="reanalysis",
                            )
                            evaluated_hres_keys.add(baseline_key)

                    if (
                        persist_ds is not None
                        and baseline_key not in evaluated_persist_keys
                    ):
                        _score_dataset_step(
                            results,
                            persist_ds,
                            era5_step,
                            top10_2t_mask,
                            case_id=case.case_id_number,
                            forecast_source="Persistence",
                            model_family="Persistence",
                            init_time=init_t,
                            valid_time=valid_time,
                            lead_hours=lt_hours,
                            year=case_year,
                            fcst_source_kind="reanalysis",
                        )
                        evaluated_persist_keys.add(baseline_key)

                del aurora_fcst

            if hres_fcst is not None:
                del hres_fcst
            if persist_ds is not None:
                del persist_ds
            gc.collect()

        del era5_case
        gc.collect()

    logger.info(f"\nEvaluation complete. Generated {len(results)} metric rows.")
    if not results:
        logger.warning("No evaluation results generated. Verify forecast files exist.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    col_order = [
        "value",
        "lead_time",
        "init_time",
        "valid_time",
        "variable",
        "region",
        "metric",
        "forecast_source",
        "target_source",
        "case_id_number",
        "event_type",
        "model_family",
        "year",
        "is_out_of_sample",
    ]
    df = df[col_order]

    out_path = Path(output_csv)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved evaluation results to {out_path.resolve()}")
    print_evaluation_summaries(df)
    return df


# =============================================================================
# Section 3: 3x3 Multi-Variable Comparison Plots
# =============================================================================


def load_and_clean_multivar_results(
    csv_path: str | Path,
    include_persistence: bool = False,
) -> pd.DataFrame:
    """
    Load multi-variable evaluation CSV, restrict to out-of-sample rows <= 240h,
    and intersect onto common (case_id_number, init_time, valid_time) targets.
    """
    df = pd.read_csv(csv_path)
    df["lead_time"] = pd.to_numeric(df["lead_time"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value", "lead_time"]).copy()
    df = df[df["lead_time"] <= DEFAULT_MAX_LEAD_HOURS].copy()

    if "model_family" in df.columns:
        df["model_group"] = df["model_family"].map(
            lambda x: MODEL_DISPLAY_NAMES.get(str(x), str(x))
        )
    else:
        df["model_group"] = df["forecast_source"].astype(str)

    if "is_out_of_sample" in df.columns:
        aurora_mask = df["model_group"].str.contains("aurora", case=False, na=False)
        if bool((df.loc[aurora_mask, "is_out_of_sample"] == True).any()):  # noqa: E712
            df = df[df["is_out_of_sample"] == True].copy()  # noqa: E712

    if not include_persistence:
        df = df[df["model_group"] != MODEL_DISPLAY_NAMES["Persistence"]].copy()

    active_groups = [m for m in MODEL_ORDER if m in set(df["model_group"].unique())]
    if not active_groups:
        raise ValueError("No valid model groups found in evaluation CSV.")

    key_cols = ["case_id_number", "init_time", "valid_time"]
    common_targets: pd.DataFrame | None = None
    for group in active_groups:
        group_keys = df[df["model_group"] == group][key_cols].drop_duplicates()
        common_targets = (
            group_keys
            if common_targets is None
            else pd.merge(common_targets, group_keys, on=key_cols, how="inner")
        )

    if common_targets is None or common_targets.empty:
        raise ValueError(
            "No common (case_id_number, init_time, valid_time) targets shared across models."
        )

    df = pd.merge(df, common_targets, on=key_cols, how="inner")
    logger.info(
        f"Aligned {len(common_targets)} common evaluation targets across {active_groups}."
    )
    return df


def _plot_3x3_region_metric(
    df_subset: pd.DataFrame,
    region: str,
    metric: str,
    output_path: Path,
) -> Path | None:
    """Render a 3x3 lead-time comparison grid across all 9 variables for one (region, metric)."""
    if df_subset.empty:
        logger.warning(f"No rows for region={region}, metric={metric}; skipping plot.")
        return None

    present_set = set(df_subset["variable"].unique())
    present_vars = [v for v in DEFAULT_VAR_ORDER if v in present_set]
    for v in sorted(present_set):
        if v not in present_vars:
            present_vars.append(v)

    n_vars = len(present_vars)
    if n_vars == 0:
        return None

    ncols = 3 if n_vars >= 3 else n_vars
    nrows = math.ceil(n_vars / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        sharex=False,
        sharey=False,
    )
    axes_flat = np.array([axes]).flatten() if n_vars == 1 else np.array(axes).flatten()

    present_models = set(df_subset["model_group"].unique())
    hue_order = [m for m in MODEL_ORDER if m in present_models]
    palette = {m: MODEL_PALETTE[m] for m in hue_order if m in MODEL_PALETTE}

    region_meta = REGION_METADATA.get(
        region, {"title": region, "bias_label": "Signed Bias (Forecast - ERA5)"}
    )
    metric_label = "RMSE" if metric == "RMSE" else region_meta["bias_label"]
    all_lead_times = sorted(df_subset["lead_time"].unique())

    handles, labels = None, None

    for idx, var_id in enumerate(present_vars):
        ax = axes_flat[idx]
        var_df = df_subset[df_subset["variable"] == var_id]
        meta = VARIABLE_METADATA.get(var_id, {"title": var_id, "unit": ""})
        title_str = meta["title"]
        unit_str = meta["unit"]

        sns.lineplot(
            data=var_df,
            x="lead_time",
            y="value",
            hue="model_group",
            hue_order=hue_order,
            palette=palette,
            style="model_group",
            style_order=hue_order,
            markers=True,
            dashes=False,
            linewidth=2.0,
            markersize=4.5,
            err_style="band",
            ax=ax,
        )

        if metric == "Bias":
            ax.axhline(0.0, color="black", linestyle="--", alpha=0.55, linewidth=1.0)

        ax.set_title(title_str, fontsize=12.5, fontweight="bold", pad=8)
        ax.set_xlabel("Lead Time (Hours)", fontsize=10.5)
        ax.set_ylabel(
            f"{metric_label} ({unit_str})" if unit_str else metric_label,
            fontsize=10.5,
        )

        if len(all_lead_times) <= 10:
            ax.set_xticks(all_lead_times)
        elif max(all_lead_times) > 72:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(24))

        if unit_str == "kg/kg":
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, -2))
            ax.yaxis.set_major_formatter(formatter)

        curr_handles, curr_labels = ax.get_legend_handles_labels()
        if curr_handles and (handles is None or len(curr_handles) > len(handles)):
            handles, labels = curr_handles, curr_labels
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    for unused_idx in range(n_vars, len(axes_flat)):
        axes_flat[unused_idx].set_visible(False)

    if handles and labels:
        fig.legend(
            handles,
            labels,
            title="Model",
            title_fontsize="12",
            fontsize="11",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=len(labels),
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            framealpha=0.95,
        )

    fig.suptitle(
        f"Out-of-Sample Heatwaves (2022) — {region_meta['title']}: {metric_label}",
        fontsize=15.5,
        fontweight="bold",
        y=0.995,
    )

    plt.subplots_adjust(
        top=0.89,
        bottom=0.06,
        left=0.07,
        right=0.97,
        hspace=0.36,
        wspace=0.28,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {output_path.resolve()}")
    return output_path


def plot_multivar_heatwave_results(
    csv_path: str | Path = DEFAULT_OUTPUT_CSV,
    output_dir: str | Path = DEFAULT_PLOTS_DIR,
    include_persistence: bool = False,
) -> list[Path]:
    """
    Generate the 4 multi-variable 3x3 grid comparison plots (RMSE and Signed Bias
    for both ``whole_bbox`` and ``top10_2t_area``).
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(
            f"Evaluation CSV {csv_file.resolve()} not found. Run with --evaluate first."
        )

    df = load_and_clean_multivar_results(
        csv_file, include_persistence=include_persistence
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = (
        ("whole_bbox", "RMSE", "whole_bbox_all_variables_rmse.png"),
        ("whole_bbox", "Bias", "whole_bbox_all_variables_bias.png"),
        ("top10_2t_area", "RMSE", "top10_2t_area_all_variables_rmse.png"),
        ("top10_2t_area", "Bias", "top10_2t_area_all_variables_bias.png"),
    )

    saved: list[Path] = []
    for region, metric, filename in plot_specs:
        sub = df[(df["region"] == region) & (df["metric"] == metric)]
        saved_path = _plot_3x3_region_metric(
            sub, region=region, metric=metric, output_path=out_dir / filename
        )
        if saved_path is not None:
            saved.append(saved_path)

    return saved


# =============================================================================
# Section 4: CLI Entrypoint
# =============================================================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for forecast generation, evaluation, and plotting."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Aurora 0.25° Finetuned vs. ECMWF HRES across 2t and 850/250 hPa "
            "(t, q, z, wind speed) on strictly out-of-sample 2022 EWB heatwaves, "
            "comparing the whole bounding box and the top-10% 2t area."
        )
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run multi-variable Aurora forecast generation.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run multi-variable evaluation against ERA5 and ECMWF HRES.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate 3x3 multi-variable RMSE and Bias comparison plots.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        default=list(DEFAULT_MODELS),
        help="Aurora 0.25° checkpoints to include (default: finetuned).",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(DEFAULT_YEARS),
        help="Out-of-sample heatwave event years to include (default: 2022).",
    )
    parser.add_argument(
        "--start-leads",
        type=int,
        nargs="+",
        default=list(DEFAULT_START_LEADS),
        help="Onset lead-time buckets in hours (default: 24 72 120).",
    )
    parser.add_argument(
        "--forecast-dir",
        type=Path,
        default=DEFAULT_FORECAST_DIR,
        help="Directory for generated multi-variable NetCDF forecasts.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path for multi-variable evaluation results.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Output directory for generated 3x3 comparison plots.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PRETRAINED_CACHE_DIR,
        help="Isolated ERA5 cache directory for pretrained_only=True inputs.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap on the number of heatwave cases (for smoke testing).",
    )
    parser.add_argument(
        "--include-persistence",
        action="store_true",
        help="Also evaluate and plot the ERA5 Persistence baseline.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Execute forecast generation, evaluation, and/or plotting based on CLI flags."""
    args = parse_args(argv)
    run_all = not (args.generate or args.evaluate or args.plot)
    run_generate = args.generate or run_all
    run_evaluate = args.evaluate or run_all
    run_plot = args.plot or run_all

    if run_generate:
        generate_oos_heatwave_multivar_forecasts(
            output_dir=args.forecast_dir,
            models=args.models,
            years=args.years,
            start_leads=args.start_leads,
            cache_dir=args.cache_dir,
            max_cases=args.max_cases,
        )
    if run_evaluate:
        evaluate_oos_heatwave_multivar(
            forecast_dir=args.forecast_dir,
            output_csv=args.output_csv,
            models=args.models,
            years=args.years,
            start_leads=args.start_leads,
            max_cases=args.max_cases,
            include_persistence=args.include_persistence,
        )
    if run_plot:
        plot_multivar_heatwave_results(
            csv_path=args.output_csv,
            output_dir=args.plots_dir,
            include_persistence=args.include_persistence,
        )


if __name__ == "__main__":
    main()
