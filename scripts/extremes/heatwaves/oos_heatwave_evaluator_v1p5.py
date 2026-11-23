"""
Out-of-sample (2024+) heatwave evaluation pipeline for Aurora 1.5 (0.25°) vs.
ECMWF IFS 0.25° Control Forecast.

Why this script exists
----------------------
``AuroraV1p5`` (``aurora-0.25-v1.5.ckpt``) was trained and fine-tuned on
1981-2023 ERA5 and IFS analyses (``cutoff_year = 2023``). Evaluating it on
2020-2023 heatwaves causes in-sample training data leakage, while WeatherBench 2's
static HRES archive (``gs://weatherbench2/datasets/hres/2016-2022-...``) ends in
2022.

This script provides a strictly out-of-sample evaluation pipeline for Aurora 1.5
on **2024+ ExtremeWeatherBench heatwaves** (16 official 2024 cases, Cases 70-85):

1. **Strict Out-of-Sample Evaluation (2024+)**:
   - All 16 EWB 2024 heatwave cases (May-August 2024, including -120h lead
     initializations) fall strictly after Aurora 1.5's 2023 training cutoff
     (``is_out_of_sample = True``).
   - Autoregressive rollouts are capped at ``max_rollout_hours = 240`` (10 days,
     40 steps) to match the ``<= 240h`` evaluation horizon and avoid redundant
     multi-week GPU rollouts on long events (e.g., Case 76, 30-day Antarctica).
2. **Ground Truth with Automatic Fallback**:
   - Uses Google Cloud **ARCO ERA5** (``load_era5_zarr``) as primary 0.25°
     ground truth, and automatically falls back to **Copernicus CDS ERA5** via
     ``AuroraDataLoader.fetch_era5()`` if a timestamp is unavailable in the Zarr
     store or if GCS is unreachable.
3. **2024+ Physical NWP Baseline (ECMWF IFS 0.25° Control Forecast)**:
   - Replaces the 2016-2022 WeatherBench 2 HRES archive with the **ECMWF IFS
     0.25° Control Forecast** (unperturbed member 0 from the cloud-optimized
     Dynamical.org / AWS Open Data ``ecmwf-ifs-ens-forecast-15-day-0-25-degree``
     archive, available April 2024+). Since ECMWF Cycle 48r1, the IFS control
     member runs at the exact same TCo1279 (~9 km / 0.25°) resolution and
     unperturbed physical core as operational IFS HRES.
   - Automatically converts IFS ``temperature_2m`` from degrees Celsius to
     Kelvin (``+ 273.15``) to match ERA5 and Aurora 1.5 units.
4. **Methodological Parity with ``oos_heatwave_evaluator.py``**:
   - Exact 12-hourly synoptic alignment (00:00 / 12:00 UTC).
   - Isolated per-case, per-model NetCDF forecast files (no cross-case leakage).
   - Single-pass IFS baseline scoring per ``(case_id_number, init_time, valid_time)``.
   - Isolated 18-variable ERA5 input cache (``weather_data/v1.5``).
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

# Set anonymous GCS access before importing extremeweatherbench/gcsfs
os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "false"
os.environ["GCSFS_TOKEN"] = "anon"

import numpy as np
import pandas as pd
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

from aurora import AuroraV1p5
from src.dataloader import AuroraDataLoader
from src.forecaster import AuroraForecaster
from src.utils import (
    align_and_subset_2d,
    aurora_batch_to_xarray,
    conditional_bias_extremes,
    extreme_area_ratio,
    intersection_over_union,
    load_era5_zarr,
    peak_amplitude_error,
    rmse,
    standardize_longitude,
    timedelta_to_hours,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

MODEL_CONFIGS: dict[str, dict[str, object]] = {
    "v1p5": {
        "checkpoint": "aurora-0.25-v1.5.ckpt",
        "cutoff_year": 2023,  # Trained 1981-2023 -> 2024+ is strictly out-of-sample
        "label": "aurora_v1p5",
    },
}

DEFAULT_MODELS: tuple[str, ...] = ("v1p5",)
DEFAULT_YEARS: tuple[int, ...] = (2024,)
DEFAULT_START_LEADS: tuple[int, ...] = (24, 72, 120)
DEFAULT_MAX_ROLLOUT_HOURS: int = 240
DEFAULT_FORECAST_DIR = Path("ewb_oos_hw_forecasts_v1p5")
DEFAULT_OUTPUT_CSV = Path("oos_heatwave_evaluations_v1p5.csv")
V1P5_CACHE_DIR = Path("weather_data/v1.5")

# Dynamical.org / AWS Open Data / Source Cooperative identifiers for ECMWF IFS 0.25°
IFS_CATALOG_ID = "ecmwf-ifs-ens-forecast-15-day-0-25-degree"
IFS_S3_BUCKET = "dynamical-ecmwf-ifs-ens"
IFS_S3_REGION = "us-west-2"
IFS_S3_PREFIX_CANDIDATES: tuple[str, ...] = (
    "ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.1.0.icechunk",
    "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
)
IFS_HTTP_ICECHUNK_URL = (
    "https://data.source.coop/dynamical/ecmwf-ifs-ens-forecast-15-day-0-25-degree/"
    "v0.1.0.icechunk"
)
IFS_LEGACY_ZARR_URL = (
    "https://data.dynamical.org/ecmwf/ifs-ens/forecast-15-day-0-25-degree/latest.zarr"
)

# Recorded as "HRES" in model_family / forecast_source so plot_oos_heatwave_eval.py
# works out-of-the-box without modifying plotting scripts.
IFS_BASELINE_FAMILY = "HRES"


class FallbackBoundingBoxRegion:
    """
    Lightweight BoundingBoxRegion compatible with ``ewb.regions.BoundingBoxRegion.mask``
    so case loading works offline without triggering remote GCS connections on import.
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
    """Case container matching ExtremeWeatherBench IndividualCase attributes."""

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

    ECMWF IFS 0.25° forecasts initialize at 00:00 and 12:00 UTC. Snapping case
    start timestamps down to 12-hourly synoptic boundaries guarantees that
    Aurora 1.5 and IFS share identical initialization and lead times.
    """
    return pd.to_datetime(dt).floor("12h")


def is_model_out_of_sample(model_family: str, year: int) -> bool:
    """Return True if *year* is strictly after *model_family*'s training cutoff."""
    if model_family == "aurora_v1p5":
        return year > int(MODEL_CONFIGS["v1p5"]["cutoff_year"])
    if model_family in (IFS_BASELINE_FAMILY, "IFS_Control", "Persistence"):
        return True
    raise ValueError(f"Unknown model_family: {model_family!r}")


def forecast_filename(
    output_dir: Path,
    model_key: str,
    init_time: pd.Timestamp,
    start_lead: int,
    case_id: int,
) -> Path:
    """Build an unambiguous per-model, per-case NetCDF forecast path."""
    return (
        Path(output_dir)
        / f"aurora_{model_key}_hw_{init_time.strftime('%Y%m%d_%H')}_L{start_lead}_case{case_id}.nc"
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

    Parses EWB's bundled ``events.yaml`` directly when available (avoiding
    unnecessary remote GCS connections during package initialization) and falls
    back to ``ewb.cases.load_cases()``. Sorted deterministically by
    ``(start_date, case_id_number)``.
    """
    target_years = set(int(y) for y in years)
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

        all_cases = ewb.cases.load_cases()
        for c in all_cases:
            if (
                c.event_type == "heat_wave"
                and isinstance(c.location, ewb.regions.BoundingBoxRegion)
                and pd.to_datetime(c.start_date).year in target_years
            ):
                bbox = c.location.as_geopandas().total_bounds  # (minx, miny, maxx, maxy)
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


def load_aurora_025_model(model_key: str) -> AuroraV1p5:
    """Instantiate and load the 0.25-degree Aurora 1.5 checkpoint."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_key {model_key!r}. Expected one of {tuple(MODEL_CONFIGS)}."
        )
    ckpt_name = str(MODEL_CONFIGS[model_key]["checkpoint"])
    model = AuroraV1p5()
    model.load_checkpoint("microsoft/aurora", ckpt_name, revision="main")
    model.eval()
    return model


def compute_all_metrics(
    fcst_2d: np.ndarray, tgt_2d: np.ndarray
) -> dict[str, float]:
    """Compute the 5 standard heatwave verification metrics on aligned 2D arrays."""
    return {
        "Peak_Amplitude_Error": peak_amplitude_error(fcst_2d, tgt_2d),
        "Conditional_Bias_Extremes": conditional_bias_extremes(fcst_2d, tgt_2d),
        "RootMeanSquaredError": rmse(fcst_2d, tgt_2d),
        "Spatial_IOU": intersection_over_union(fcst_2d, tgt_2d),
        "Extreme_Area_Ratio": extreme_area_ratio(fcst_2d, tgt_2d),
    }


# =============================================================================
# Ground Truth & ECMWF IFS 0.25° Control Forecast Loaders
# =============================================================================


def _ensure_kelvin(da: xr.DataArray) -> xr.DataArray:
    """
    Convert a 2m temperature DataArray from degrees Celsius to Kelvin if needed.

    Dynamical.org's ``ecmwf-ifs-ens-forecast-15-day-0-25-degree`` stores
    ``temperature_2m`` in ``degree_Celsius`` (values typically in [-90, 60]),
    whereas ERA5 and Aurora 1.5 use Kelvin (values typically in [180, 335]).
    """
    units = str(da.attrs.get("units", "")).lower()
    is_celsius_attr = "celsius" in units or units in ("c", "degc", "deg_c", "°c")
    sample_vals = da.values
    finite_vals = sample_vals[np.isfinite(sample_vals)]
    is_celsius_val = bool(finite_vals.size > 0 and float(np.nanmedian(finite_vals)) < 150.0)

    if is_celsius_attr or is_celsius_val:
        da = da + np.float32(273.15)
        da.attrs["units"] = "K"
    return da


def _normalize_ifs_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Normalize ECMWF IFS ENS dataset to the unperturbed Control Forecast
    (member 0) with ``init_time``, ``lead_time``, and ``surface_air_temperature``.
    """
    # Select unperturbed control member (member 0) if an ensemble dimension exists
    for ens_dim in ("ensemble_member", "number", "member"):
        if ens_dim in ds.dims:
            ds = ds.isel({ens_dim: 0}, drop=True)
            break

    rename_map: dict[str, str] = {}
    if "time" in ds.dims and "init_time" not in ds.dims:
        rename_map["time"] = "init_time"
    for lt_cand in ("prediction_timedelta", "step"):
        if lt_cand in ds.dims and "lead_time" not in ds.dims:
            rename_map[lt_cand] = "lead_time"
            break
    for temp_cand in ("temperature_2m", "2m_temperature", "t2m"):
        if temp_cand in ds.data_vars:
            rename_map[temp_cand] = "surface_air_temperature"
            break

    if rename_map:
        ds = ds.rename(rename_map)

    if "surface_air_temperature" not in ds.data_vars:
        raise KeyError(
            "Could not find 2m temperature variable in IFS dataset. "
            f"Available variables: {list(ds.data_vars.keys())}"
        )
    return ds[["surface_air_temperature"]]


def load_ifs_control_zarr() -> xr.Dataset:
    """
    Connect lazily to the 2024+ ECMWF IFS 0.25° Control Forecast archive
    (``ecmwf-ifs-ens-forecast-15-day-0-25-degree``, member 0).

    Tries in order:
      1. ``dynamical_catalog.open("ecmwf-ifs-ens-forecast-15-day-0-25-degree")``
      2. ``icechunk`` anonymous S3/HTTP repository on AWS Open Data / Source Cooperative
      3. Direct Zarr endpoint on ``data.dynamical.org``
    """
    errors: list[str] = []

    # 1. Try dynamical_catalog if installed
    try:
        import dynamical_catalog  # type: ignore[import-not-found]

        logger.info(
            f"Connecting to ECMWF IFS 0.25° archive via dynamical_catalog ({IFS_CATALOG_ID})..."
        )
        ds = dynamical_catalog.open(IFS_CATALOG_ID, chunks=None)
        return _normalize_ifs_dataset(ds)
    except Exception as exc:
        errors.append(f"dynamical_catalog: {exc}")

    # 2. Try Icechunk S3 / HTTP repositories directly (icechunk is bundled with EWB)
    try:
        import icechunk

        for prefix in IFS_S3_PREFIX_CANDIDATES:
            try:
                logger.info(
                    f"Connecting to ECMWF IFS 0.25° Icechunk store s3://{IFS_S3_BUCKET}/{prefix}..."
                )
                storage = icechunk.s3_storage(
                    bucket=IFS_S3_BUCKET,
                    prefix=prefix,
                    region=IFS_S3_REGION,
                    anonymous=True,
                )
                repo = icechunk.Repository.open(storage)
                session = repo.readonly_session(branch="main")
                ds = xr.open_zarr(session.store, consolidated=False, chunks=None)
                return _normalize_ifs_dataset(ds)
            except Exception as s3_exc:
                errors.append(f"s3://{IFS_S3_BUCKET}/{prefix}: {s3_exc}")

        if hasattr(icechunk, "http_storage"):
            try:
                logger.info(
                    f"Connecting to ECMWF IFS 0.25° Icechunk HTTP store ({IFS_HTTP_ICECHUNK_URL})..."
                )
                storage = icechunk.http_storage(IFS_HTTP_ICECHUNK_URL)
                repo = icechunk.Repository.open(storage)
                session = repo.readonly_session(branch="main")
                ds = xr.open_zarr(session.store, consolidated=False, chunks=None)
                return _normalize_ifs_dataset(ds)
            except Exception as http_exc:
                errors.append(f"http_storage({IFS_HTTP_ICECHUNK_URL}): {http_exc}")
    except Exception as ic_exc:
        errors.append(f"icechunk: {ic_exc}")

    # 3. Try direct Zarr URL fallback
    try:
        logger.info(
            f"Connecting to ECMWF IFS 0.25° Zarr endpoint ({IFS_LEGACY_ZARR_URL})..."
        )
        ds = xr.open_zarr(IFS_LEGACY_ZARR_URL, chunks=None)
        return _normalize_ifs_dataset(ds)
    except Exception as zarr_exc:
        errors.append(f"open_zarr({IFS_LEGACY_ZARR_URL}): {zarr_exc}")

    joined_errors = "\n  - ".join(errors)
    raise RuntimeError(
        "Failed to open ECMWF IFS 0.25° Control Forecast archive via all configured backends:\n"
        f"  - {joined_errors}"
    )


def fetch_case_era5_ground_truth(
    case: HeatwaveCaseRecord,
    aligned_start: pd.Timestamp,
    end_dt: pd.Timestamp,
    era5_zarr_ds: xr.Dataset | None,
    cds_loader: AuroraDataLoader | None,
    cache_dir: Path = V1P5_CACHE_DIR,
) -> tuple[xr.Dataset, AuroraDataLoader | None]:
    """
    Fetch 6-hourly 0.25° ERA5 2m temperature ground truth for *case* over
    ``[aligned_start, end_dt]``.

    Uses ARCO ERA5 Zarr (``load_era5_zarr``) as primary source and automatically
    falls back to Copernicus CDS ERA5 via ``AuroraDataLoader.fetch_era5()`` if
    ARCO ERA5 does not cover the requested 2024+ dates or fails to stream.
    """
    required_times = pd.date_range(
        start=aligned_start, end=end_dt, freq="6h"
    )
    required_dts: list[datetime] = [ts.to_pydatetime() for ts in required_times]

    if era5_zarr_ds is not None:
        try:
            era5_case_raw = era5_zarr_ds.sel(
                time=slice(aligned_start.to_pydatetime(), end_dt.to_pydatetime())
            )
            six_hr_mask = era5_case_raw.time.dt.hour.isin([0, 6, 12, 18])
            era5_case_raw = era5_case_raw.isel(time=np.where(six_hr_mask.values)[0])
            if era5_case_raw.sizes.get("time", 0) >= len(required_dts):
                era5_case = case.location.mask(era5_case_raw).compute()
                era5_case = standardize_longitude(era5_case)
                vals = era5_case["surface_air_temperature"].values
                if np.isfinite(vals).any():
                    return era5_case, cds_loader
            logger.warning(
                f"  ARCO ERA5 Zarr incomplete for Case {case.case_id_number} "
                f"({aligned_start} -> {end_dt}); falling back to CDS ERA5."
            )
        except Exception as exc:
            logger.warning(
                f"  ARCO ERA5 Zarr fetch failed for Case {case.case_id_number} ({exc}); "
                "falling back to CDS ERA5 via AuroraDataLoader."
            )

    if cds_loader is None:
        cds_loader = AuroraDataLoader(
            cache_dir=Path(cache_dir), pretrained_only=False, res=0.25
        )

    # Download/load global 6-hourly ERA5 from the shared v1.5 cache and mask to case bbox
    cds_raw = cds_loader.fetch_era5(required_dts, bbox=None)
    if "t2m" in cds_raw.data_vars:
        cds_temp = cds_raw[["t2m"]].rename({"t2m": "surface_air_temperature"})
    elif "2m_temperature" in cds_raw.data_vars:
        cds_temp = cds_raw[["2m_temperature"]].rename(
            {"2m_temperature": "surface_air_temperature"}
        )
    else:
        raise KeyError(
            f"Expected 't2m' or '2m_temperature' in CDS ERA5 dataset, got {list(cds_raw.data_vars)}"
        )

    era5_case = case.location.mask(cds_temp).compute()
    era5_case = standardize_longitude(era5_case)
    del cds_raw, cds_temp
    return era5_case, cds_loader


# =============================================================================
# Section 1: Forecast Generation
# =============================================================================


def generate_oos_heatwave_forecasts(
    output_dir: str | Path = DEFAULT_FORECAST_DIR,
    models: Sequence[str] = DEFAULT_MODELS,
    years: Sequence[int] = DEFAULT_YEARS,
    start_leads: Sequence[int] = DEFAULT_START_LEADS,
    cache_dir: str | Path = V1P5_CACHE_DIR,
    max_rollout_hours: int = DEFAULT_MAX_ROLLOUT_HOURS,
    max_cases: int | None = None,
) -> None:
    """
    Generate Aurora 1.5 (0.25°) forecasts for 2024+ EWB heatwaves at *start_leads*.

    Groups pending forecasts by ``(model_key, init_time)`` so concurrent cases
    sharing an initialization time execute a single GPU rollout while writing
    isolated per-case NetCDF files capped at *max_rollout_hours*.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cases = get_oos_heatwave_cases(years=years, max_cases=max_cases)
    logger.info(
        f"Loaded {len(cases)} EWB out-of-sample heatwave cases for years={list(years)}."
    )
    if not cases:
        logger.warning("No matching heatwave cases found. Exiting generation.")
        return

    max_lead_td = timedelta(hours=int(max_rollout_hours))
    pending_by_model: dict[str, dict[pd.Timestamp, list[dict]]] = {}

    for model_key in models:
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model key: {model_key!r}")
        grouped_by_init: dict[pd.Timestamp, list[dict]] = {}
        for case in cases:
            aligned_start = align_to_synoptic_12h(case.start_date)
            raw_event_end = pd.to_datetime(case.end_date)
            for start_lead in start_leads:
                init_time = aligned_start - timedelta(hours=int(start_lead))
                effective_end = min(raw_event_end, init_time + max_lead_td)
                if effective_end < aligned_start:
                    continue

                out_file = forecast_filename(
                    output_path,
                    model_key,
                    init_time,
                    int(start_lead),
                    case.case_id_number,
                )
                if out_file.exists():
                    logger.info(
                        f"[{model_key}] Skipping Case {case.case_id_number} L{start_lead} "
                        f"({out_file.name} exists)"
                    )
                    continue
                grouped_by_init.setdefault(init_time, []).append(
                    {
                        "case": case,
                        "aligned_start": aligned_start,
                        "event_end": effective_end,
                        "start_lead": int(start_lead),
                        "filename": out_file,
                    }
                )
        if grouped_by_init:
            pending_by_model[model_key] = grouped_by_init

    if not pending_by_model:
        logger.info("All requested out-of-sample Aurora 1.5 forecasts already exist.")
        return

    data_pipeline = AuroraDataLoader(
        cache_dir=Path(cache_dir), pretrained_only=False, res=0.25
    )

    for model_key, grouped_by_init in pending_by_model.items():
        logger.info("=" * 60)
        logger.info(
            f"Generating forecasts for model={model_key} "
            f"({MODEL_CONFIGS[model_key]['checkpoint']}) across "
            f"{len(grouped_by_init)} unique init times (cap={max_rollout_hours}h)"
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
                        )
                        for idx in matching_indices:
                            active_buffers[idx].append(step_ds)

                    # Flush finished jobs early to minimize RAM accumulation
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
# Section 2: Direct Evaluation (Aurora 1.5 vs. ECMWF IFS 0.25° Control vs. ERA5)
# =============================================================================


def _append_metric_rows(
    results: list[dict],
    metrics: dict[str, float],
    *,
    case_id: int,
    forecast_source: str,
    model_family: str,
    init_time: pd.Timestamp,
    valid_time: pd.Timestamp,
    lead_hours: int,
    year: int,
) -> None:
    """Append one row per verification metric in standard EWB schema."""
    oos_flag = is_model_out_of_sample(model_family, year)
    for metric_name, val in metrics.items():
        results.append(
            {
                "value": val,
                "lead_time": lead_hours,
                "init_time": init_time,
                "valid_time": valid_time,
                "target_variable": "surface_air_temperature",
                "metric": metric_name,
                "forecast_source": forecast_source,
                "target_source": "ERA5",
                "case_id_number": case_id,
                "event_type": "heat_wave",
                "forecast_variable": "surface_air_temperature",
                "model_family": model_family,
                "year": year,
                "is_out_of_sample": oos_flag,
            }
        )


def _select_ifs_lead(ifs_fcst: xr.Dataset, lt_hours: int) -> xr.DataArray:
    """Select a 2D IFS surface_air_temperature field in Kelvin handling int or timedelta64 coords."""
    if np.issubdtype(ifs_fcst.lead_time.dtype, np.timedelta64):
        da = ifs_fcst["surface_air_temperature"].sel(
            lead_time=pd.to_timedelta(lt_hours, unit="h")
        )
    else:
        da = ifs_fcst["surface_air_temperature"].sel(lead_time=lt_hours)
    return _ensure_kelvin(da)


def print_evaluation_summaries(df: pd.DataFrame) -> None:
    """Print common-target out-of-sample summary (<= 240h) and per-year breakdown."""
    if df.empty:
        return

    df_valid = df.dropna(subset=["value"]).copy()
    df_valid = df_valid[df_valid["lead_time"] <= 240]
    if df_valid.empty:
        return

    families = sorted(df_valid["model_family"].unique())
    common = None
    for fam in families:
        keys = df_valid[df_valid["model_family"] == fam][
            ["case_id_number", "init_time", "valid_time"]
        ].drop_duplicates()
        common = (
            keys
            if common is None
            else pd.merge(
                common,
                keys,
                on=["case_id_number", "init_time", "valid_time"],
                how="inner",
            )
        )

    if common is not None and not common.empty:
        df_common = pd.merge(
            df_valid,
            common,
            on=["case_id_number", "init_time", "valid_time"],
            how="inner",
        )
        summary_common = (
            df_common.groupby(["metric", "model_family"])["value"]
            .agg(["count", "mean", "std"])
        )
        logger.info(
            "\n=== Strict Out-of-Sample Common-Target Summary (<= 240h) ===\n"
            + summary_common.to_string()
        )

    df_oos_only = df_valid[df_valid["is_out_of_sample"] == True]  # noqa: E712
    if not df_oos_only.empty:
        summary_oos = (
            df_oos_only.groupby(["year", "metric", "model_family"])["value"]
            .agg(["count", "mean", "std"])
        )
        logger.info(
            "\n=== Out-of-Sample Breakdown by Year (is_out_of_sample == True, <= 240h) ===\n"
            + summary_oos.to_string()
        )


def evaluate_oos_heatwaves(
    forecast_dir: str | Path = DEFAULT_FORECAST_DIR,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    models: Sequence[str] = DEFAULT_MODELS,
    years: Sequence[int] = DEFAULT_YEARS,
    start_leads: Sequence[int] = DEFAULT_START_LEADS,
    cache_dir: str | Path = V1P5_CACHE_DIR,
    max_rollout_hours: int = DEFAULT_MAX_ROLLOUT_HOURS,
    max_cases: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate saved Aurora 1.5 (0.25°) forecasts against the ECMWF IFS 0.25°
    Control Forecast using ARCO ERA5 (with automatic CDS ERA5 fallback) as
    ground truth.
    """
    forecast_path = Path(forecast_dir)
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"Forecast directory {forecast_path.resolve()} does not exist. "
            "Run with --generate first."
        )

    logger.info("=" * 60)
    logger.info("Starting Aurora 1.5 Out-of-Sample Heatwave Evaluation Pipeline")
    logger.info("=" * 60)

    era5_zarr_ds: xr.Dataset | None = None
    try:
        era5_zarr_ds = load_era5_zarr(variable="temperature")
    except Exception as exc:
        logger.warning(
            f"Could not connect to ARCO ERA5 Zarr ({exc}); "
            "will use Copernicus CDS ERA5 via AuroraDataLoader."
        )

    ifs_ds = load_ifs_control_zarr()
    cds_loader: AuroraDataLoader | None = None
    max_lead_td = timedelta(hours=int(max_rollout_hours))

    cases = get_oos_heatwave_cases(years=years, max_cases=max_cases)
    logger.info(
        f"Evaluating {len(cases)} EWB heatwave cases for years={list(years)}."
    )

    results: list[dict] = []
    evaluated_ifs_keys: set[tuple[int, pd.Timestamp, pd.Timestamp]] = set()

    for case_idx, case in enumerate(cases):
        aligned_start = align_to_synoptic_12h(case.start_date)
        raw_end_dt = pd.to_datetime(case.end_date)
        # Cap case evaluation window at max_init + max_rollout_hours
        latest_init = aligned_start - timedelta(hours=min(int(l) for l in start_leads))
        end_dt = min(raw_end_dt, latest_init + max_lead_td)
        case_year = int(aligned_start.year)

        logger.info(
            f"\nEvaluating Case {case.case_id_number}: {case.title} "
            f"({case_idx + 1}/{len(cases)}) | {aligned_start} -> {end_dt}"
        )

        era5_case, cds_loader = fetch_case_era5_ground_truth(
            case=case,
            aligned_start=aligned_start,
            end_dt=end_dt,
            era5_zarr_ds=era5_zarr_ds,
            cds_loader=cds_loader,
            cache_dir=Path(cache_dir),
        )

        for start_lead in start_leads:
            start_lead = int(start_lead)
            init_t = aligned_start - timedelta(hours=start_lead)
            init_window_end = min(raw_end_dt, init_t + max_lead_td)

            model_files: list[tuple[str, Path]] = []
            for model_key in models:
                nc_path = forecast_filename(
                    forecast_path, model_key, init_t, start_lead, case.case_id_number
                )
                if nc_path.exists():
                    model_files.append((model_key, nc_path))
                else:
                    logger.warning(
                        f"  Missing forecast file for Case {case.case_id_number} "
                        f"({model_key}, L{start_lead}): {nc_path.name}"
                    )

            if not model_files:
                continue

            # Fetch ECMWF IFS 0.25° Control Forecast once for this (case, init_t)
            ifs_fcst = None
            try:
                ifs_raw = ifs_ds.sel(init_time=init_t.to_datetime64())
                if np.issubdtype(ifs_raw.lead_time.dtype, np.timedelta64):
                    ifs_lead_td = pd.to_timedelta(ifs_raw.lead_time.values)
                else:
                    ifs_lead_td = pd.to_timedelta(ifs_raw.lead_time.values, unit="h")
                ifs_valid_times = init_t + ifs_lead_td
                in_window = np.asarray(
                    (ifs_valid_times >= aligned_start)
                    & (ifs_valid_times <= init_window_end)
                    & (ifs_valid_times.hour.isin([0, 6, 12, 18]))
                )
                if np.any(in_window):
                    ifs_slice = ifs_raw.isel(lead_time=np.where(in_window)[0])
                    ifs_fcst = case.location.mask(ifs_slice).compute()
                    ifs_fcst = standardize_longitude(ifs_fcst)
                else:
                    logger.warning(
                        f"  ECMWF IFS has no lead times in [{aligned_start}, {init_window_end}] "
                        f"for init_time={init_t}"
                    )
            except (KeyError, IndexError) as exc:
                logger.warning(
                    f"  No exact ECMWF IFS init_time match for {init_t}: {exc}"
                )

            # Evaluate Aurora 1.5 & record matched ECMWF IFS Control once per (case, init_t, valid_t)
            for model_key, nc_path in model_files:
                model_family = str(MODEL_CONFIGS[model_key]["label"])
                forecast_source = f"{model_family}_L{start_lead}"

                with xr.open_dataset(nc_path) as aurora_raw:
                    aurora_fcst = case.location.mask(aurora_raw).compute()
                aurora_fcst = standardize_longitude(aurora_fcst)
                aurora_var = (
                    "2t_aurora"
                    if "2t_aurora" in aurora_fcst.data_vars
                    else [
                        v
                        for v in aurora_fcst.data_vars
                        if v not in ("latitude", "longitude")
                    ][0]
                )
                file_init_t = pd.to_datetime(aurora_fcst.init_time.values[0])

                for lt in aurora_fcst.lead_time.values:
                    lt_hours = timedelta_to_hours(lt)
                    if lt_hours > max_rollout_hours:
                        continue
                    valid_time = file_init_t + timedelta(hours=lt_hours)

                    if not (aligned_start <= valid_time <= init_window_end):
                        continue

                    try:
                        tgt_2d = era5_case["surface_air_temperature"].sel(
                            time=valid_time.to_pydatetime()
                        )
                    except KeyError:
                        continue

                    # 1. Aurora 1.5 evaluation
                    fcst_2d_aurora = aurora_fcst[aurora_var].sel(
                        init_time=aurora_fcst.init_time.values[0], lead_time=lt
                    )
                    fcst_arr, tgt_arr = align_and_subset_2d(fcst_2d_aurora, tgt_2d)
                    _append_metric_rows(
                        results,
                        compute_all_metrics(fcst_arr, tgt_arr),
                        case_id=case.case_id_number,
                        forecast_source=forecast_source,
                        model_family=model_family,
                        init_time=file_init_t,
                        valid_time=valid_time,
                        lead_hours=lt_hours,
                        year=case_year,
                    )

                    # 2. Matched ECMWF IFS 0.25° Control Forecast evaluation
                    baseline_key = (case.case_id_number, init_t, valid_time)
                    if ifs_fcst is not None and baseline_key not in evaluated_ifs_keys:
                        try:
                            fcst_2d_ifs = _select_ifs_lead(ifs_fcst, lt_hours)
                        except KeyError:
                            pass
                        else:
                            ifs_arr, ifs_tgt_arr = align_and_subset_2d(
                                fcst_2d_ifs, tgt_2d
                            )
                            _append_metric_rows(
                                results,
                                compute_all_metrics(ifs_arr, ifs_tgt_arr),
                                case_id=case.case_id_number,
                                forecast_source=IFS_BASELINE_FAMILY,
                                model_family=IFS_BASELINE_FAMILY,
                                init_time=init_t,
                                valid_time=valid_time,
                                lead_hours=lt_hours,
                                year=case_year,
                            )
                            evaluated_ifs_keys.add(baseline_key)

                del aurora_fcst

            if ifs_fcst is not None:
                del ifs_fcst
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
        "target_variable",
        "metric",
        "forecast_source",
        "target_source",
        "case_id_number",
        "event_type",
        "forecast_variable",
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
# Section 3: CLI Entrypoint
# =============================================================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for generation and evaluation stages."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Aurora 1.5 (0.25°) against ECMWF IFS 0.25° Control Forecast "
            "on strictly out-of-sample (2024+) ExtremeWeatherBench heatwaves."
        )
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run Aurora 1.5 forecast generation. Runs both stages if neither --generate nor --evaluate is set.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation against ERA5 (ARCO Zarr + CDS fallback) and ECMWF IFS 0.25° Control.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        default=list(DEFAULT_MODELS),
        help="Aurora 0.25° checkpoints to include (default: v1p5).",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(DEFAULT_YEARS),
        help="Out-of-sample heatwave event years to include (default: 2024).",
    )
    parser.add_argument(
        "--start-leads",
        type=int,
        nargs="+",
        default=list(DEFAULT_START_LEADS),
        help="Onset lead-time buckets in hours (default: 24 72 120).",
    )
    parser.add_argument(
        "--max-rollout-hours",
        type=int,
        default=DEFAULT_MAX_ROLLOUT_HOURS,
        help="Maximum forecast rollout horizon in hours per initialization (default: 240).",
    )
    parser.add_argument(
        "--forecast-dir",
        type=Path,
        default=DEFAULT_FORECAST_DIR,
        help="Directory for generated NetCDF forecasts (default: ewb_oos_hw_forecasts_v1p5).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path for evaluation results (default: oos_heatwave_evaluations_v1p5.csv).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=V1P5_CACHE_DIR,
        help="ERA5 cache directory for Aurora 1.5 (18 surface vars) inputs (default: weather_data/v1.5).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap on the number of heatwave cases (for smoke testing).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Execute forecast generation and/or evaluation based on CLI flags."""
    args = parse_args(argv)
    run_generate = args.generate or (not args.generate and not args.evaluate)
    run_evaluate = args.evaluate or (not args.generate and not args.evaluate)

    if run_generate:
        generate_oos_heatwave_forecasts(
            output_dir=args.forecast_dir,
            models=args.models,
            years=args.years,
            start_leads=args.start_leads,
            cache_dir=args.cache_dir,
            max_rollout_hours=args.max_rollout_hours,
            max_cases=args.max_cases,
        )
    if run_evaluate:
        evaluate_oos_heatwaves(
            forecast_dir=args.forecast_dir,
            output_csv=args.output_csv,
            models=args.models,
            years=args.years,
            start_leads=args.start_leads,
            cache_dir=args.cache_dir,
            max_rollout_hours=args.max_rollout_hours,
            max_cases=args.max_cases,
        )


if __name__ == "__main__":
    main()
