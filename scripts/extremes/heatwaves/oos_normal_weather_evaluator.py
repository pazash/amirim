"""
Out-of-sample (2022) normal-weather baseline evaluation pipeline for Aurora 0.25°
vs. ECMWF IFS HRES and ERA5 Persistence.

Why this script exists
----------------------
The legacy ``normal_weather_evaluator.py`` selected "normal" baseline dates by
subtracting 1-3 years from each ExtremeWeatherBench (EWB) heatwave date, landing
in 2017-2019. Because ``aurora-0.25-pretrained.ckpt`` (trained 1979-2020) and
``aurora-0.25-finetuned.ckpt`` (fine-tuned 2016-2021) saw 2017-2019 during
training, comparing 2022 out-of-sample heatwaves against 2017-2019 normal dates
confounded extreme-weather degradation with the in-sample vs. out-of-sample
generalization gap.

This script resolves that flaw with strict methodological parity to
``oos_heatwave_evaluator.py``:

1. **100% Out-of-Sample (2022) Normal Windows**:
   - Pairs each of the 12 2022 EWB heatwave cases 1-to-1 with a same-season
     normal window in **2022** of identical duration and bounding box.
   - All initialization times (including L120 = -120h) and valid times fall
     strictly within 2022, so both ``aurora_pretrained`` and ``aurora_finetuned``
     are 100% out-of-sample and WeatherBench 2 HRES (2016-2022) has full coverage.
2. **Conflict-Free & ERA5-Verifiable Normality**:
   - Candidate windows maintain a strict synoptic buffer from all spatially
     overlapping 2022 EWB heatwave cases (e.g., separating the June France,
     July UK, and August UK heatwaves).
   - ``--verify-dates`` queries ARCO ERA5 over a ±45-day seasonal pool around
     each heatwave to select/verify the non-overlapping window closest to the
     seasonal median temperature (excluding uncatalogued heat spikes or cold
     snaps) and caches the verified dates to ``oos_normal_verified_windows.json``.
3. **Exact Synoptic Alignment & Isolated Per-Case Outputs**:
   - Timestamps are aligned to 12-hourly synoptic boundaries (00:00 / 12:00 UTC)
     matching HRES initializations.
   - Forecast NetCDF files are isolated by ``(model_key, init_time, start_lead, case_id)``.
   - Shares the isolated 4-variable ERA5 input cache (``weather_data/pretrained_global``).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Sequence

# Set anonymous GCS access before importing extremeweatherbench/gcsfs
os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "false"
os.environ["GCSFS_TOKEN"] = "anon"

import numpy as np
import pandas as pd
import torch
import xarray as xr

try:
    import extremeweatherbench as ewb

    _HAS_EWB = True
except Exception:
    ewb = None  # type: ignore[assignment]
    _HAS_EWB = False

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
    conditional_bias_extremes,
    extreme_area_ratio,
    filter_batch_for_pretrained,
    intersection_over_union,
    load_era5_zarr,
    load_hres_zarr,
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

DEFAULT_MODELS: tuple[str, ...] = ("pretrained", "finetuned")
DEFAULT_START_LEADS: tuple[int, ...] = (24, 72, 120)
DEFAULT_FORECAST_DIR = Path("ewb_oos_normal_forecasts")
DEFAULT_OUTPUT_CSV = Path("oos_normal_evaluations.csv")
DEFAULT_VERIFIED_JSON = Path("oos_normal_verified_windows.json")
PRETRAINED_CACHE_DIR = Path("weather_data/pretrained_global")


class FallbackBoundingBoxRegion:
    """
    Lightweight BoundingBoxRegion compatible with ``ewb.regions.BoundingBoxRegion.mask``
    when ``extremeweatherbench`` cannot initialize its remote default stores offline.
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


def make_bbox_region(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float
):
    """Create an EWB BoundingBoxRegion if available, else FallbackBoundingBoxRegion."""
    if _HAS_EWB and ewb is not None:
        return ewb.regions.BoundingBoxRegion(
            latitude_min=lat_min,
            latitude_max=lat_max,
            longitude_min=lon_min,
            longitude_max=lon_max,
        )
    return FallbackBoundingBoxRegion(
        latitude_min=lat_min,
        latitude_max=lat_max,
        longitude_min=lon_min,
        longitude_max=lon_max,
    )


@dataclass
class NormalBaselineCase:
    """
    1-to-1 paired 2022 normal-weather baseline case corresponding to an EWB 2022
    heatwave case. Preserves ``case_id_number``, bounding box, and duration.
    """

    case_id_number: int
    title: str
    extreme_start_date: str
    extreme_end_date: str
    start_date: str
    end_date: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    event_type: str = "heat_wave_normal_baseline"

    def __post_init__(self) -> None:
        self.location = make_bbox_region(
            self.lat_min, self.lat_max, self.lon_min, self.lon_max
        )

    @property
    def duration(self) -> pd.Timedelta:
        return pd.to_datetime(self.end_date) - pd.to_datetime(self.start_date)


# 12 paired 2022 normal baseline cases matching the 12 2022 EWB heatwave cases
# in oos_heatwave_evaluations.csv:
# - Exact same case_id_number, bounding box, and event duration
# - Strictly within 2022 (including -120h initialization offset) -> 100% OOS for
#   both aurora-0.25-pretrained (<=2020) and aurora-0.25-finetuned (<=2021),
#   and 100% covered by WeatherBench 2 HRES (2016-2022)
# - Snapped to 12h synoptic boundaries (00:00 / 12:00 UTC)
# - Zero temporal overlap (with >=6-day buffer) with any spatially overlapping
#   2022 EWB heatwave case
DEFAULT_2022_NORMAL_CASES: tuple[NormalBaselineCase, ...] = (
    NormalBaselineCase(
        case_id_number=26,
        title="2022 West Australia (Normal Baseline)",
        extreme_start_date="2022-01-15 00:00:00",
        extreme_end_date="2022-01-25 00:00:00",
        start_date="2022-02-05 00:00:00",  # +21d (keeps -120h init inside 2022)
        end_date="2022-02-15 00:00:00",
        lat_min=-35.5,
        lat_max=-28.5,
        lon_min=113.25,
        lon_max=119.25,
    ),
    NormalBaselineCase(
        case_id_number=25,
        title="2022 East Antarctica (Normal Baseline)",
        extreme_start_date="2022-03-12 00:00:00",
        extreme_end_date="2022-03-26 00:00:00",
        start_date="2022-02-19 00:00:00",  # -21d
        end_date="2022-03-05 00:00:00",
        lat_min=-85.5,
        lat_max=-64.5,
        lon_min=101.0,
        lon_max=145.75,
    ),
    NormalBaselineCase(
        case_id_number=24,
        title="2022 India (Normal Baseline)",
        extreme_start_date="2022-04-24 00:00:00",
        extreme_end_date="2022-05-04 00:00:00",
        start_date="2022-04-03 00:00:00",  # -21d
        end_date="2022-04-13 00:00:00",
        lat_min=20.75,
        lat_max=41.75,
        lon_min=58.0,
        lon_max=76.0,
    ),
    NormalBaselineCase(
        case_id_number=2,
        title="2022 Upper Midwest (Normal Baseline)",
        extreme_start_date="2022-05-07 00:00:00",
        extreme_end_date="2022-05-17 00:00:00",
        start_date="2022-04-16 00:00:00",  # -21d (avoids Case 6 May 17 & Case 4 Jun 30)
        end_date="2022-04-26 00:00:00",
        lat_min=31.5,
        lat_max=47.5,
        lon_min=-102.25,
        lon_max=-82.25,
    ),
    NormalBaselineCase(
        case_id_number=6,
        title="2022 Mid-Atlantic (Normal Baseline)",
        extreme_start_date="2022-05-17 00:00:00",
        extreme_end_date="2022-05-24 00:00:00",
        start_date="2022-06-07 00:00:00",  # +21d
        end_date="2022-06-14 00:00:00",
        lat_min=33.75,
        lat_max=41.75,
        lon_min=-83.0,
        lon_max=-74.0,
    ),
    NormalBaselineCase(
        case_id_number=3,
        title="2022 California (Normal Baseline)",
        extreme_start_date="2022-06-07 00:00:00",
        extreme_end_date="2022-06-15 00:00:00",
        start_date="2022-05-17 00:00:00",  # -21d
        end_date="2022-05-25 00:00:00",
        lat_min=31.5,
        lat_max=42.5,
        lon_min=-123.75,
        lon_max=-104.75,
    ),
    NormalBaselineCase(
        case_id_number=22,
        title="2022 France (Normal Baseline)",
        extreme_start_date="2022-06-09 00:00:00",
        extreme_end_date="2022-06-21 00:00:00",
        start_date="2022-05-19 00:00:00",  # -21d (clear of Jul & Aug UK/Europe HWs)
        end_date="2022-05-31 00:00:00",
        lat_min=31.0,
        lat_max=51.0,
        lon_min=-9.0,
        lon_max=10.0,
    ),
    NormalBaselineCase(
        case_id_number=23,
        title="2022 Japan (Normal Baseline)",
        extreme_start_date="2022-06-20 00:00:00",
        extreme_end_date="2022-07-05 00:00:00",
        start_date="2022-05-27 00:00:00",  # -24d
        end_date="2022-06-11 00:00:00",
        lat_min=33.25,
        lat_max=42.25,
        lon_min=126.25,
        lon_max=142.25,
    ),
    NormalBaselineCase(
        case_id_number=4,
        title="2022 Texas (Normal Baseline)",
        extreme_start_date="2022-06-30 00:00:00",
        extreme_end_date="2022-07-18 00:00:00",
        start_date="2022-07-28 00:00:00",  # +28d (clear of May Case 2 & Jul Case 4)
        end_date="2022-08-15 00:00:00",
        lat_min=25.25,
        lat_max=37.25,
        lon_min=-103.25,
        lon_max=-90.25,
    ),
    NormalBaselineCase(
        case_id_number=21,
        title="2022 UK (July) (Normal Baseline)",
        extreme_start_date="2022-07-15 00:00:00",
        extreme_end_date="2022-07-23 00:00:00",
        start_date="2022-06-27 00:00:00",  # -18d (in calm gap between Jun 21 France & Jul 15 UK)
        end_date="2022-07-05 00:00:00",
        lat_min=37.0,
        lat_max=54.0,
        lon_min=-7.75,
        lon_max=2.25,
    ),
    NormalBaselineCase(
        case_id_number=20,
        title="2022 UK (August) (Normal Baseline)",
        extreme_start_date="2022-08-08 00:00:00",
        extreme_end_date="2022-08-16 00:00:00",
        start_date="2022-08-29 00:00:00",  # +21d (clear of Jul 15-23 & Aug 8-16 UK HWs)
        end_date="2022-09-06 00:00:00",
        lat_min=41.0,
        lat_max=56.0,
        lon_min=-6.75,
        lon_max=8.25,
    ),
    NormalBaselineCase(
        case_id_number=19,
        title="2022 Germany (Normal Baseline)",
        extreme_start_date="2022-12-23 00:00:00",
        extreme_end_date="2022-12-31 00:00:00",
        start_date="2022-12-02 00:00:00",  # -21d (stays strictly inside 2022)
        end_date="2022-12-10 00:00:00",
        lat_min=46.0,
        lat_max=55.0,
        lon_min=-2.0,
        lon_max=17.0,
    ),
)


def align_to_synoptic_12h(dt: str | pd.Timestamp) -> pd.Timestamp:
    """Floor a timestamp to the nearest 00:00 or 12:00 UTC boundary."""
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
    """Build an unambiguous per-model, per-case NetCDF normal forecast path."""
    return (
        Path(output_dir)
        / f"aurora_{model_key}_norm_{init_time.strftime('%Y%m%d_%H')}_L{start_lead}_case{case_id}.nc"
    )


def _bboxes_overlap(a: NormalBaselineCase, b: NormalBaselineCase) -> bool:
    """Return True if two lat/lon bounding boxes intersect."""
    lat_overlap = (a.lat_min <= b.lat_max) and (b.lat_min <= a.lat_max)
    lon_overlap = (a.lon_min <= b.lon_max) and (b.lon_min <= a.lon_max)
    return bool(lat_overlap and lon_overlap)


def verify_and_select_normal_windows(
    verified_json_path: str | Path = DEFAULT_VERIFIED_JSON,
    search_days: int = 45,
    min_buffer_days: int = 5,
) -> list[NormalBaselineCase]:
    """
    Verify and refine the 12 2022 normal baseline windows against ARCO ERA5.

    For each 2022 heatwave case, downloads the 6-hourly ARCO ERA5 temperature
    field across a ±*search_days* seasonal window in 2022, masks out all
    spatially overlapping 2022 heatwave periods (plus *min_buffer_days*), and
    selects the non-overlapping 12h-aligned window of identical duration whose
    domain-mean and spatial-90th-percentile temperatures are closest to the
    non-extreme seasonal median (while rejecting windows with max temperature
    above the 75th percentile of daily peaks or mean below the 25th percentile).
    """
    logger.info("=" * 60)
    logger.info("Verifying 2022 Same-Season Normal Windows Against ARCO ERA5")
    logger.info("=" * 60)

    era5_ds = load_era5_zarr(variable="temperature")
    base_cases = list(DEFAULT_2022_NORMAL_CASES)
    verified_cases: list[NormalBaselineCase] = []
    diagnostics: list[dict[str, object]] = []

    min_init_bound = pd.Timestamp("2022-01-01 00:00:00")
    max_end_bound = pd.Timestamp("2022-12-31 18:00:00")
    max_lead_td = timedelta(hours=max(DEFAULT_START_LEADS))
    buffer_td = timedelta(days=min_buffer_days)

    for case in base_cases:
        hw_start = align_to_synoptic_12h(case.extreme_start_date)
        hw_end = pd.to_datetime(case.extreme_end_date)
        duration = hw_end - hw_start

        # Identify all spatially overlapping 2022 heatwave windows to avoid
        forbidden_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for other in base_cases:
            if _bboxes_overlap(case, other):
                o_start = pd.to_datetime(other.extreme_start_date) - buffer_td
                o_end = pd.to_datetime(other.extreme_end_date) + buffer_td
                forbidden_intervals.append((o_start, o_end))

        season_start = max(
            min_init_bound + max_lead_td,
            hw_start - timedelta(days=search_days),
        )
        season_end = min(
            max_end_bound,
            hw_end + timedelta(days=search_days),
        )

        logger.info(
            f"Scanning ERA5 seasonal pool for Case {case.case_id_number} ({case.title}): "
            f"{season_start} -> {season_end}"
        )
        era5_raw = era5_ds.sel(
            time=slice(season_start.to_pydatetime(), season_end.to_pydatetime())
        )
        six_hr_mask = era5_raw.time.dt.hour.isin([0, 6, 12, 18])
        era5_raw = era5_raw.isel(time=np.where(six_hr_mask.values)[0])
        era5_box = case.location.mask(era5_raw).compute()
        era5_box = standardize_longitude(era5_box)
        del era5_raw

        da_temp = era5_box["surface_air_temperature"]
        times = pd.to_datetime(da_temp.time.values)

        # Compute per-timestep spatial statistics: domain mean, 90th percentile, and max
        flat_vals = da_temp.values.reshape(len(times), -1)
        ts_mean = np.nanmean(flat_vals, axis=1)
        ts_p90 = np.nanpercentile(flat_vals, 90.0, axis=1)
        ts_max = np.nanmax(flat_vals, axis=1)

        ts_df = pd.DataFrame(
            {"mean": ts_mean, "p90": ts_p90, "max": ts_max}, index=times
        )

        # Reference heatwave stats
        hw_slice = ts_df.loc[
            (ts_df.index >= hw_start) & (ts_df.index <= hw_end)
        ]
        hw_mean_k = float(hw_slice["mean"].mean()) if not hw_slice.empty else float("nan")
        hw_p90_k = float(hw_slice["p90"].mean()) if not hw_slice.empty else float("nan")
        hw_max_k = float(hw_slice["max"].max()) if not hw_slice.empty else float("nan")

        # Non-extreme seasonal distribution (excluding forbidden heatwave intervals)
        non_hw_mask = np.ones(len(ts_df), dtype=bool)
        for f_start, f_end in forbidden_intervals:
            non_hw_mask &= ~((ts_df.index >= f_start) & (ts_df.index <= f_end))

        non_hw_df = ts_df.loc[non_hw_mask]
        if non_hw_df.empty:
            non_hw_df = ts_df

        season_med_mean = float(non_hw_df["mean"].median())
        season_iqr_mean = max(float(non_hw_df["mean"].quantile(0.75) - non_hw_df["mean"].quantile(0.25)), 0.5)
        season_med_p90 = float(non_hw_df["p90"].median())
        season_iqr_p90 = max(float(non_hw_df["p90"].quantile(0.75) - non_hw_df["p90"].quantile(0.25)), 0.5)
        season_p75_max = float(non_hw_df["max"].quantile(0.75))
        season_p25_mean = float(non_hw_df["mean"].quantile(0.25))

        # Enumerate 12h-aligned candidate windows of identical duration
        cand_starts = pd.date_range(
            start=align_to_synoptic_12h(season_start),
            end=align_to_synoptic_12h(season_end - duration),
            freq="12h",
        )

        best_start: pd.Timestamp | None = None
        best_score = float("inf")
        best_stats: dict[str, float] = {}

        for c_start in cand_starts:
            # Enforce exact diurnal phase alignment (same UTC start hour as the extreme heatwave)
            if c_start.hour != hw_start.hour:
                continue

            c_end = c_start + duration
            if c_start - max_lead_td < min_init_bound or c_end > max_end_bound:
                continue

            # Must be at least 14 days away from the case's own heatwave start
            if abs((c_start - hw_start).total_seconds()) < 14 * 86400:
                continue

            # Check zero overlap with any forbidden interval (including buffer)
            overlaps_hw = False
            for f_start, f_end in forbidden_intervals:
                if (c_start <= f_end) and (c_end >= f_start):
                    overlaps_hw = True
                    break
            if overlaps_hw:
                continue

            w_df = ts_df.loc[(ts_df.index >= c_start) & (ts_df.index <= c_end)]
            if len(w_df) < 4:
                continue

            w_mean = float(w_df["mean"].mean())
            w_p90 = float(w_df["p90"].mean())
            w_max = float(w_df["max"].max())

            # Penalty if window has an uncatalogued warm spike or cold snap
            tail_penalty = 0.0
            if w_max > season_p75_max:
                tail_penalty += 5.0 * ((w_max - season_p75_max) / season_iqr_p90)
            if w_mean < season_p25_mean:
                tail_penalty += 5.0 * ((season_p25_mean - w_mean) / season_iqr_mean)

            score = (
                abs(w_mean - season_med_mean) / season_iqr_mean
                + abs(w_p90 - season_med_p90) / season_iqr_p90
                + tail_penalty
            )
            if score < best_score:
                best_score = score
                best_start = c_start
                best_stats = {
                    "normal_mean_k": w_mean,
                    "normal_p90_k": w_p90,
                    "normal_max_k": w_max,
                    "score": score,
                }

        if best_start is None:
            logger.warning(
                f"  Falling back to default normal window for Case {case.case_id_number}"
            )
            best_start = align_to_synoptic_12h(case.start_date)
            c_end = best_start + duration
            w_df = ts_df.loc[(ts_df.index >= best_start) & (ts_df.index <= c_end)]
            best_stats = {
                "normal_mean_k": float(w_df["mean"].mean()) if not w_df.empty else float("nan"),
                "normal_p90_k": float(w_df["p90"].mean()) if not w_df.empty else float("nan"),
                "normal_max_k": float(w_df["max"].max()) if not w_df.empty else float("nan"),
                "score": float("nan"),
            }
        else:
            c_end = best_start + duration

        verified_case = NormalBaselineCase(
            case_id_number=case.case_id_number,
            title=case.title,
            extreme_start_date=case.extreme_start_date,
            extreme_end_date=case.extreme_end_date,
            start_date=best_start.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=c_end.strftime("%Y-%m-%d %H:%M:%S"),
            lat_min=case.lat_min,
            lat_max=case.lat_max,
            lon_min=case.lon_min,
            lon_max=case.lon_max,
        )
        verified_cases.append(verified_case)

        diag_row = {
            "case_id_number": case.case_id_number,
            "title": case.title,
            "extreme_start_date": case.extreme_start_date,
            "extreme_end_date": case.extreme_end_date,
            "normal_start_date": verified_case.start_date,
            "normal_end_date": verified_case.end_date,
            "lat_min": case.lat_min,
            "lat_max": case.lat_max,
            "lon_min": case.lon_min,
            "lon_max": case.lon_max,
            "hw_mean_k": round(hw_mean_k, 3),
            "normal_mean_k": round(best_stats["normal_mean_k"], 3),
            "season_median_mean_k": round(season_med_mean, 3),
            "hw_p90_k": round(hw_p90_k, 3),
            "normal_p90_k": round(best_stats["normal_p90_k"], 3),
            "season_median_p90_k": round(season_med_p90, 3),
            "hw_max_k": round(hw_max_k, 3),
            "normal_max_k": round(best_stats["normal_max_k"], 3),
        }
        diagnostics.append(diag_row)
        logger.info(
            f"  -> Selected Normal Window: {verified_case.start_date} to {verified_case.end_date} | "
            f"ERA5 Mean: Normal={diag_row['normal_mean_k']:.2f}K (Season Med={season_med_mean:.2f}K) vs "
            f"HW={hw_mean_k:.2f}K | Max: Normal={diag_row['normal_max_k']:.2f}K vs HW={hw_max_k:.2f}K"
        )

        del era5_box, da_temp, ts_df
        gc.collect()

    out_json = Path(verified_json_path)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    logger.info(f"Saved ERA5-verified normal windows to {out_json.resolve()}")

    verified_cases.sort(key=lambda c: (pd.to_datetime(c.start_date), c.case_id_number))
    return verified_cases


def get_oos_normal_cases(
    verified_json_path: str | Path = DEFAULT_VERIFIED_JSON,
    max_cases: int | None = None,
) -> list[NormalBaselineCase]:
    """
    Load the 12 2022 OOS normal baseline cases (from *verified_json_path* if
    present, otherwise from ``DEFAULT_2022_NORMAL_CASES``), sorted by
    ``(start_date, case_id_number)``.
    """
    json_path = Path(verified_json_path)
    if json_path.exists():
        logger.info(f"Loading ERA5-verified normal windows from {json_path}")
        with json_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        cases = [
            NormalBaselineCase(
                case_id_number=int(r["case_id_number"]),
                title=str(r["title"]),
                extreme_start_date=str(r["extreme_start_date"]),
                extreme_end_date=str(r["extreme_end_date"]),
                start_date=str(r["normal_start_date"]),
                end_date=str(r["normal_end_date"]),
                lat_min=float(r["lat_min"]),
                lat_max=float(r["lat_max"]),
                lon_min=float(r["lon_min"]),
                lon_max=float(r["lon_max"]),
            )
            for r in records
        ]
    else:
        cases = list(DEFAULT_2022_NORMAL_CASES)

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


def compute_all_metrics(
    fcst_2d: np.ndarray, tgt_2d: np.ndarray
) -> dict[str, float]:
    """Compute the 5 standard verification metrics on aligned 2D arrays."""
    return {
        "Peak_Amplitude_Error": peak_amplitude_error(fcst_2d, tgt_2d),
        "Conditional_Bias_Extremes": conditional_bias_extremes(fcst_2d, tgt_2d),
        "RootMeanSquaredError": rmse(fcst_2d, tgt_2d),
        "Spatial_IOU": intersection_over_union(fcst_2d, tgt_2d),
        "Extreme_Area_Ratio": extreme_area_ratio(fcst_2d, tgt_2d),
    }


# =============================================================================
# Section 1: Forecast Generation
# =============================================================================


def generate_oos_normal_forecasts(
    output_dir: str | Path = DEFAULT_FORECAST_DIR,
    models: Sequence[str] = DEFAULT_MODELS,
    start_leads: Sequence[int] = DEFAULT_START_LEADS,
    cache_dir: str | Path = PRETRAINED_CACHE_DIR,
    verified_json_path: str | Path = DEFAULT_VERIFIED_JSON,
    max_cases: int | None = None,
) -> None:
    """
    Generate Aurora 0.25° forecasts for the 12 2022 OOS normal baseline windows.

    Groups pending forecasts by (model_key, init_time) so concurrent cases share
    a single GPU rollout while saving isolated per-case NetCDF files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cases = get_oos_normal_cases(
        verified_json_path=verified_json_path, max_cases=max_cases
    )
    logger.info(f"Loaded {len(cases)} 2022 OOS normal baseline cases.")
    if not cases:
        logger.warning("No normal baseline cases found. Exiting generation.")
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
                if out_file.exists():
                    logger.info(
                        f"[{model_key}] Skipping Normal Case {case.case_id_number} "
                        f"L{start_lead} ({out_file.name} exists)"
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
        logger.info("All requested 2022 OOS normal Aurora forecasts already exist.")
        return

    data_pipeline = AuroraDataLoader(
        cache_dir=Path(cache_dir), pretrained_only=True, res=0.25
    )

    for model_key, grouped_by_init in pending_by_model.items():
        logger.info("=" * 60)
        logger.info(
            f"Generating 2022 normal forecasts for model={model_key} "
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
                        )
                        for idx in matching_indices:
                            active_buffers[idx].append(step_ds)

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
# Section 2: Direct Evaluation (Aurora 0.25 vs. HRES & Persistence vs. ERA5)
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
    """Append one row per verification metric in standard EWB OOS schema."""
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
                "event_type": "heat_wave_normal_baseline",
                "forecast_variable": "surface_air_temperature",
                "model_family": model_family,
                "year": year,
                "is_out_of_sample": oos_flag,
            }
        )


def _select_hres_lead(hres_fcst: xr.Dataset, lt_hours: int) -> xr.DataArray:
    """Select a 2D HRES surface_air_temperature field handling int or timedelta64 coords."""
    if np.issubdtype(hres_fcst.lead_time.dtype, np.timedelta64):
        return hres_fcst["surface_air_temperature"].sel(
            lead_time=pd.to_timedelta(lt_hours, unit="h")
        )
    return hres_fcst["surface_air_temperature"].sel(lead_time=lt_hours)


def print_evaluation_summaries(df: pd.DataFrame) -> None:
    """Print strict 2022 common OOS normal-weather summary."""
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
        summary = (
            df_common.groupby(["metric", "model_family"])["value"]
            .agg(["count", "mean", "std"])
        )
        logger.info(
            "\n=== Strict Out-of-Sample Normal Baseline Summary: 2022 Common Targets (<= 240h) ===\n"
            + summary.to_string()
        )


def evaluate_oos_normal(
    forecast_dir: str | Path = DEFAULT_FORECAST_DIR,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    models: Sequence[str] = DEFAULT_MODELS,
    start_leads: Sequence[int] = DEFAULT_START_LEADS,
    verified_json_path: str | Path = DEFAULT_VERIFIED_JSON,
    max_cases: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate saved 2022 normal-weather Aurora 0.25° forecasts against
    WeatherBench 2 HRES and ERA5 Persistence using ARCO ERA5 as ground truth.
    """
    forecast_path = Path(forecast_dir)
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"Forecast directory {forecast_path.resolve()} does not exist. "
            "Run with --generate first."
        )

    logger.info("=" * 60)
    logger.info("Starting Out-of-Sample Normal-Weather Evaluation Pipeline (2022)")
    logger.info("=" * 60)

    era5_ds = load_era5_zarr(variable="temperature")
    hres_ds = load_hres_zarr(variable="temperature")
    cases = get_oos_normal_cases(
        verified_json_path=verified_json_path, max_cases=max_cases
    )
    logger.info(f"Evaluating {len(cases)} 2022 OOS normal baseline cases.")

    results: list[dict] = []
    evaluated_hres_keys: set[tuple[int, pd.Timestamp, pd.Timestamp]] = set()
    evaluated_persist_keys: set[tuple[int, pd.Timestamp, pd.Timestamp]] = set()

    for case_idx, case in enumerate(cases):
        aligned_start = align_to_synoptic_12h(case.start_date)
        end_dt = pd.to_datetime(case.end_date)
        case_year = int(aligned_start.year)

        logger.info(
            f"\nEvaluating Normal Baseline for Case {case.case_id_number}: {case.title} "
            f"({case_idx + 1}/{len(cases)})\n"
            f"  Paired Extreme Date: {case.extreme_start_date} -> {case.extreme_end_date}\n"
            f"  Normal Baseline Date: {aligned_start} -> {end_dt}"
        )

        era5_case_raw = era5_ds.sel(
            time=slice(aligned_start.to_pydatetime(), end_dt.to_pydatetime())
        )
        six_hr_mask = era5_case_raw.time.dt.hour.isin([0, 6, 12, 18])
        era5_case_raw = era5_case_raw.isel(time=np.where(six_hr_mask.values)[0])
        era5_case = case.location.mask(era5_case_raw).compute()
        era5_case = standardize_longitude(era5_case)
        del era5_case_raw

        for start_lead in start_leads:
            start_lead = int(start_lead)
            init_t = aligned_start - timedelta(hours=start_lead)

            model_files: list[tuple[str, Path]] = []
            for model_key in models:
                nc_path = forecast_filename(
                    forecast_path,
                    model_key,
                    init_t,
                    start_lead,
                    case.case_id_number,
                )
                if nc_path.exists():
                    model_files.append((model_key, nc_path))
                else:
                    logger.warning(
                        f"  Missing normal forecast file for Case {case.case_id_number} "
                        f"({model_key}, L{start_lead}): {nc_path.name}"
                    )

            if not model_files:
                continue

            # 1. Fetch ERA5 Persistence field once for this (case, init_t)
            persist_2d = None
            try:
                era5_init_raw = era5_ds.sel(time=init_t.to_pydatetime())
                era5_init = case.location.mask(era5_init_raw).compute()
                era5_init = standardize_longitude(era5_init)
                persist_2d = era5_init["surface_air_temperature"]
                del era5_init_raw, era5_init
            except Exception as exc:
                logger.warning(
                    f"  Failed loading ERA5 persistence at init_t={init_t}: {exc}"
                )

            # 2. Fetch WeatherBench 2 HRES forecast once for this (case, init_t)
            hres_fcst = None
            try:
                hres_raw = hres_ds.sel(init_time=init_t.to_datetime64())
                if np.issubdtype(hres_raw.lead_time.dtype, np.timedelta64):
                    hres_lead_td = pd.to_timedelta(hres_raw.lead_time.values)
                else:
                    hres_lead_td = pd.to_timedelta(
                        hres_raw.lead_time.values, unit="h"
                    )
                hres_valid_times = init_t + hres_lead_td
                in_window = np.asarray(
                    (hres_valid_times >= aligned_start) & (hres_valid_times <= end_dt)
                )
                if np.any(in_window):
                    hres_slice = hres_raw.isel(lead_time=np.where(in_window)[0])
                    hres_fcst = case.location.mask(hres_slice).compute()
                    hres_fcst = standardize_longitude(hres_fcst)
                else:
                    logger.warning(
                        f"  HRES has no lead times in [{aligned_start}, {end_dt}] "
                        f"for init_time={init_t}"
                    )
            except (KeyError, IndexError) as exc:
                logger.warning(
                    f"  No exact HRES init_time match for {init_t}: {exc}"
                )

            # 3. Evaluate each Aurora checkpoint & record baselines once per (case, init_t, valid_t)
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
                    valid_time = file_init_t + timedelta(hours=lt_hours)

                    if not (aligned_start <= valid_time <= end_dt):
                        continue

                    try:
                        tgt_2d = era5_case["surface_air_temperature"].sel(
                            time=valid_time.to_pydatetime()
                        )
                    except KeyError:
                        continue

                    # A. Aurora evaluation
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

                    # B. Matched HRES evaluation (once per unique case, init_time, valid_time)
                    baseline_key = (case.case_id_number, init_t, valid_time)
                    if hres_fcst is not None and baseline_key not in evaluated_hres_keys:
                        try:
                            fcst_2d_hres = _select_hres_lead(hres_fcst, lt_hours)
                        except KeyError:
                            pass
                        else:
                            hres_arr, hres_tgt_arr = align_and_subset_2d(
                                fcst_2d_hres, tgt_2d
                            )
                            _append_metric_rows(
                                results,
                                compute_all_metrics(hres_arr, hres_tgt_arr),
                                case_id=case.case_id_number,
                                forecast_source="HRES",
                                model_family="HRES",
                                init_time=init_t,
                                valid_time=valid_time,
                                lead_hours=lt_hours,
                                year=case_year,
                            )
                            evaluated_hres_keys.add(baseline_key)

                    # C. Matched ERA5 Persistence evaluation (once per unique case, init_time, valid_time)
                    if (
                        persist_2d is not None
                        and baseline_key not in evaluated_persist_keys
                    ):
                        per_arr, per_tgt_arr = align_and_subset_2d(persist_2d, tgt_2d)
                        _append_metric_rows(
                            results,
                            compute_all_metrics(per_arr, per_tgt_arr),
                            case_id=case.case_id_number,
                            forecast_source="Persistence",
                            model_family="Persistence",
                            init_time=init_t,
                            valid_time=valid_time,
                            lead_hours=lt_hours,
                            year=case_year,
                        )
                        evaluated_persist_keys.add(baseline_key)

                del aurora_fcst

            if hres_fcst is not None:
                del hres_fcst
            if persist_2d is not None:
                del persist_2d
            gc.collect()

        del era5_case
        gc.collect()

    logger.info(
        f"\nNormal-weather evaluation complete. Generated {len(results)} metric rows."
    )
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
    logger.info(f"Saved normal evaluation results to {out_path.resolve()}")
    print_evaluation_summaries(df)
    return df


# =============================================================================
# Section 3: CLI Entrypoint
# =============================================================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for normal window verification, generation, and evaluation."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Aurora 0.25° (Pretrained and Finetuned) against HRES and "
            "ERA5 Persistence on strictly out-of-sample (2022) same-season normal "
            "weather windows paired 1-to-1 with 2022 EWB heatwave cases."
        )
    )
    parser.add_argument(
        "--verify-dates",
        action="store_true",
        help=(
            "Query ARCO ERA5 to select/verify the 12 2022 same-season normal windows "
            "closest to the seasonal median and save to --verified-json."
        ),
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help=(
            "Run Aurora forecast generation. Runs both --generate and --evaluate if "
            "none of --verify-dates, --generate, or --evaluate is set."
        ),
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation against ERA5, HRES, and Persistence.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        default=list(DEFAULT_MODELS),
        help="Aurora 0.25° checkpoints to include (default: pretrained finetuned).",
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
        help="Directory for generated NetCDF normal forecasts.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path for normal evaluation results.",
    )
    parser.add_argument(
        "--verified-json",
        type=Path,
        default=DEFAULT_VERIFIED_JSON,
        help="Path to JSON file caching ERA5-verified 2022 normal windows.",
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
        help="Optional cap on the number of normal cases (for smoke testing).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Execute date verification, forecast generation, and/or evaluation."""
    args = parse_args(argv)
    default_pipeline = (
        not args.verify_dates and not args.generate and not args.evaluate
    )
    run_generate = args.generate or default_pipeline
    run_evaluate = args.evaluate or default_pipeline

    if args.verify_dates:
        verify_and_select_normal_windows(verified_json_path=args.verified_json)

    if run_generate:
        generate_oos_normal_forecasts(
            output_dir=args.forecast_dir,
            models=args.models,
            start_leads=args.start_leads,
            cache_dir=args.cache_dir,
            verified_json_path=args.verified_json,
            max_cases=args.max_cases,
        )

    if run_evaluate:
        evaluate_oos_normal(
            forecast_dir=args.forecast_dir,
            output_csv=args.output_csv,
            models=args.models,
            start_leads=args.start_leads,
            verified_json_path=args.verified_json,
            max_cases=args.max_cases,
        )


if __name__ == "__main__":
    main()
