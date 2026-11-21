"""
Out-of-sample heatwave evaluation pipeline for Aurora 0.25° vs. physical baselines.

Why this script exists
----------------------
``heatwave_evaluator.py`` and ``extreme_evaluator.py`` evaluate ``AuroraV1p5``
(trained/fine-tuned on 1981-2023 ERA5 & IFS analyses) on 2020-2022 heatwaves,
which causes severe in-sample training data leakage. This script resolves that
issue and fixes two methodological bugs in the legacy pipeline:

1. **Strict Out-of-Sample Evaluation**:
   - ``aurora-0.25-pretrained.ckpt`` (``AuroraPretrained``, trained 1979-2020)
     is out-of-sample on **2021 and 2022** ExtremeWeatherBench heatwaves.
   - ``aurora-0.25-finetuned.ckpt`` (``Aurora``, fine-tuned 2016-2021)
     is out-of-sample on **2022** ExtremeWeatherBench heatwaves.
   - Every output row records ``year`` and ``is_out_of_sample``.
2. **Physical & Skill Baselines**:
   - **ECMWF IFS HRES** (0.25° WeatherBench 2 Zarr, 2016-2022) as the consensus
     operational physical NWP baseline.
   - **ERA5 Persistence** (``t(init_time)`` held constant across lead times) as
     a lower-bound reference.
3. **No Cross-Case Forecast Contamination**:
   - Forecast NetCDF files include both ``model_key`` and ``case_id_number``,
     and evaluation matches files strictly by ``case_id_number`` and ``init_time``.
4. **Exact Synoptic Initialization Alignment (00/12 UTC)**:
   - Case start timestamps are floored to 12-hourly boundaries (00:00 / 12:00 UTC)
     so Aurora, HRES, and Persistence share the exact same ``init_time`` and
     ``lead_time`` without falling back to a 6-hour older HRES cycle.
5. **Isolated Pretrained Cache**:
   - Uses ``AuroraDataLoader(..., pretrained_only=True)`` in a dedicated cache
     directory (``weather_data/pretrained_global``) to keep 4-variable inputs
     isolated from Aurora 1.5's 18-variable NetCDF files.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import sys
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Sequence

# Set anonymous GCS access before importing extremeweatherbench/gcsfs
os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "false"
os.environ["GCSFS_TOKEN"] = "anon"

import extremeweatherbench as ewb
import numpy as np
import pandas as pd
import torch
import xarray as xr

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

sys.path.append(str(Path(__file__).resolve().parents[3]))

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
DEFAULT_YEARS: tuple[int, ...] = (2021, 2022)
DEFAULT_START_LEADS: tuple[int, ...] = (24, 72, 120)
DEFAULT_FORECAST_DIR = Path("ewb_oos_hw_forecasts")
DEFAULT_OUTPUT_CSV = Path("oos_heatwave_evaluations.csv")
PRETRAINED_CACHE_DIR = Path("weather_data/pretrained_global")


def align_to_synoptic_12h(dt: str | pd.Timestamp) -> pd.Timestamp:
    """
    Floor a timestamp to the nearest 00:00 or 12:00 UTC boundary.

    WeatherBench 2 HRES only initializes at 00:00 and 12:00 UTC. Snapping case
    start timestamps down to 12-hourly synoptic boundaries guarantees that
    Aurora, HRES, and Persistence share identical initialization and lead times.
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
    """Build an unambiguous per-model, per-case NetCDF forecast path."""
    return (
        Path(output_dir)
        / f"aurora_{model_key}_hw_{init_time.strftime('%Y%m%d_%H')}_L{start_lead}_case{case_id}.nc"
    )


def get_oos_heatwave_cases(
    years: Sequence[int] = DEFAULT_YEARS,
    max_cases: int | None = None,
) -> list:
    """
    Load ExtremeWeatherBench BoundingBoxRegion heatwave cases in *years*.

    Sorted deterministically by (start_date, case_id_number).
    """
    target_years = set(int(y) for y in years)
    all_cases = ewb.cases.load_cases()
    hw_cases = [
        c
        for c in all_cases
        if c.event_type == "heat_wave"
        and isinstance(c.location, ewb.regions.BoundingBoxRegion)
        and pd.to_datetime(c.start_date).year in target_years
    ]
    hw_cases.sort(key=lambda c: (pd.to_datetime(c.start_date), c.case_id_number))
    if max_cases is not None:
        hw_cases = hw_cases[:max_cases]
    return hw_cases


def load_aurora_025_model(model_key: str):
    """Instantiate and load the requested 0.25-degree Aurora checkpoint."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_key {model_key!r}. Expected one of {tuple(MODEL_CONFIGS)}."
        )
    ckpt_name = str(MODEL_CONFIGS[model_key]["checkpoint"])
    if model_key == "pretrained":
        model = AuroraPretrained(use_lora=False, autocast=True)
        model.load_checkpoint("microsoft/aurora", ckpt_name, revision="main", strict=False)
    else:
        model = Aurora(use_lora=True, autocast=True)
        model.load_checkpoint("microsoft/aurora", ckpt_name, revision="main", strict=True)
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
# Section 1: Forecast Generation
# =============================================================================


def generate_oos_heatwave_forecasts(
    output_dir: str | Path = DEFAULT_FORECAST_DIR,
    models: Sequence[str] = DEFAULT_MODELS,
    years: Sequence[int] = DEFAULT_YEARS,
    start_leads: Sequence[int] = DEFAULT_START_LEADS,
    cache_dir: str | Path = PRETRAINED_CACHE_DIR,
    max_cases: int | None = None,
) -> None:
    """
    Generate Aurora 0.25° forecasts for EWB heatwaves in *years* at *start_leads*.

    Groups pending forecasts by (model_key, init_time) so concurrent cases or
    matching initializations share a single GPU rollout while writing isolated
    per-case NetCDF files.
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

    # Collect pending jobs per model before loading checkpoints or data loader
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
                    output_path, model_key, init_time, int(start_lead), case.case_id_number
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
                        "event_end": event_end,
                        "start_lead": int(start_lead),
                        "filename": out_file,
                    }
                )
        if grouped_by_init:
            pending_by_model[model_key] = grouped_by_init

    if not pending_by_model:
        logger.info("All requested out-of-sample Aurora forecasts already exist.")
        return

    data_pipeline = AuroraDataLoader(
        cache_dir=Path(cache_dir), pretrained_only=True, res=0.25
    )

    for model_key, grouped_by_init in pending_by_model.items():
        logger.info("=" * 60)
        logger.info(
            f"Generating forecasts for model={model_key} "
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

                # Track per-job xarray slices and write immediately once each job's end is reached
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

                # Flush any remaining active jobs if max_end wasn't hit cleanly
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


def _select_hres_lead(hres_fcst: xr.Dataset, lt_hours: int) -> xr.DataArray:
    """Select a 2D HRES surface_air_temperature field handling int or timedelta64 coords."""
    if np.issubdtype(hres_fcst.lead_time.dtype, np.timedelta64):
        return hres_fcst["surface_air_temperature"].sel(
            lead_time=pd.to_timedelta(lt_hours, unit="h")
        )
    return hres_fcst["surface_air_temperature"].sel(lead_time=lt_hours)


def print_evaluation_summaries(df: pd.DataFrame) -> None:
    """Print strict 2022 common OOS summary and 2021-2022 pretrained OOS summary."""
    if df.empty:
        return

    df_valid = df.dropna(subset=["value"]).copy()
    df_valid = df_valid[df_valid["lead_time"] <= 240]

    # 1. Strict 2022 summary where both Pretrained & Finetuned are out-of-sample
    df_2022 = df_valid[df_valid["year"] == 2022]
    if not df_2022.empty:
        families_2022 = sorted(df_2022["model_family"].unique())
        common = None
        for fam in families_2022:
            keys = df_2022[df_2022["model_family"] == fam][
                ["case_id_number", "init_time", "valid_time"]
            ].drop_duplicates()
            common = keys if common is None else pd.merge(common, keys, on=["case_id_number", "init_time", "valid_time"], how="inner")

        if common is not None and not common.empty:
            df_2022_common = pd.merge(
                df_2022, common, on=["case_id_number", "init_time", "valid_time"], how="inner"
            )
            summary_2022 = (
                df_2022_common.groupby(["metric", "model_family"])["value"]
                .agg(["count", "mean", "std"])
            )
            logger.info(
                "\n=== Strict Out-of-Sample Summary: 2022 Common Targets (<= 240h) ===\n"
                + summary_2022.to_string()
            )

    # 2. Full 2021-2022 summary for models that are strictly OOS across both years
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
    max_cases: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate saved Aurora 0.25° forecasts against WeatherBench 2 HRES and
    ERA5 Persistence using ARCO ERA5 as ground truth.
    """
    forecast_path = Path(forecast_dir)
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"Forecast directory {forecast_path.resolve()} does not exist. "
            "Run with --generate first."
        )

    logger.info("=" * 60)
    logger.info("Starting Out-of-Sample Heatwave Evaluation Pipeline")
    logger.info("=" * 60)

    era5_ds = load_era5_zarr(variable="temperature")
    hres_ds = load_hres_zarr(variable="temperature")
    cases = get_oos_heatwave_cases(years=years, max_cases=max_cases)
    logger.info(f"Evaluating {len(cases)} EWB heatwave cases for years={list(years)}.")

    results: list[dict] = []
    # Deduplicate baseline evaluations across multiple Aurora models sharing (case, init_t, valid_t)
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

        # Fetch 6-hourly ERA5 ground truth for the case window
        era5_case_raw = era5_ds.sel(time=slice(aligned_start.to_pydatetime(), end_dt.to_pydatetime()))
        six_hr_mask = era5_case_raw.time.dt.hour.isin([0, 6, 12, 18])
        era5_case_raw = era5_case_raw.isel(time=np.where(six_hr_mask.values)[0])
        era5_case = case.location.mask(era5_case_raw).compute()
        era5_case = standardize_longitude(era5_case)
        del era5_case_raw

        for start_lead in start_leads:
            start_lead = int(start_lead)
            init_t = aligned_start - timedelta(hours=start_lead)

            # Locate available Aurora forecast files strictly matching this case & lead
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

            # Fetch ERA5 Persistence field once for this (case, init_t)
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

            # Fetch WeatherBench 2 HRES forecast once for this (case, init_t)
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

            # Evaluate each Aurora checkpoint & record baselines once per (case, init_t, valid_t)
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

                    # 1. Aurora evaluation
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

                    # 2. Matched HRES evaluation (once per unique case, init_time, valid_time)
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

                    # 3. Matched ERA5 Persistence evaluation (once per unique case, init_time, valid_time)
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
            "Evaluate Aurora 0.25° (Pretrained and/or Finetuned) against HRES and "
            "ERA5 Persistence on strictly out-of-sample EWB heatwaves."
        )
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run Aurora forecast generation. Runs both stages if neither --generate nor --evaluate is set.",
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
        "--years",
        type=int,
        nargs="+",
        default=list(DEFAULT_YEARS),
        help="Heatwave event years to include (default: 2021 2022).",
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
        help="Directory for generated NetCDF forecasts.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output CSV path for evaluation results.",
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
            max_cases=args.max_cases,
        )
    if run_evaluate:
        evaluate_oos_heatwaves(
            forecast_dir=args.forecast_dir,
            output_csv=args.output_csv,
            models=args.models,
            years=args.years,
            start_leads=args.start_leads,
            max_cases=args.max_cases,
        )


if __name__ == "__main__":
    main()
