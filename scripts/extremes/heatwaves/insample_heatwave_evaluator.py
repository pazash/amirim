"""
In-sample ("trained-on") heatwave evaluation pipeline for Aurora 0.25° vs. physical baselines.

Why this script exists
----------------------
``oos_heatwave_evaluator.py`` evaluates ``aurora-0.25-pretrained`` (trained
1979-2020) and ``aurora-0.25-finetuned`` (fine-tuned 2016-2021) on strictly
out-of-sample 2021-2022 heatwaves. To measure the **in-sample vs. out-of-sample
generalization gap** directly, this companion script applies the exact same
leak-free pipeline fixes to **12 major 2016-2020 heatwaves** (EWB Case 29 in
2020 plus 11 WMO/national-record heatwaves from 2016-2020) that both Aurora
checkpoints saw during training:

1. **100% In-Sample Aurora Evaluation (2016-2020)**:
   - Both ``aurora-0.25-pretrained.ckpt`` (1979-2020) and
     ``aurora-0.25-finetuned.ckpt`` (2016-2021) were trained on 2016-2020 data
     (``is_out_of_sample = False`` for both Aurora models).
2. **Identical Physical & Skill Baselines**:
   - **ECMWF IFS HRES** (0.25° WeatherBench 2 Zarr, 2016-2022).
   - **ERA5 Persistence** (``t(init_time)`` held constant across lead times).
3. **Methodological Parity with ``oos_heatwave_evaluator.py``**:
   - Isolated per-case, per-model NetCDF forecast files (no cross-case leakage).
   - Exact 12-hourly synoptic alignment (00:00 / 12:00 UTC) matching HRES.
   - Single-pass HRES & Persistence scoring per ``(case_id_number, init_time, valid_time)``.
   - Isolated 4-variable ERA5 input cache (``weather_data/pretrained_global``).
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass
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

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.extremes.heatwaves.oos_heatwave_evaluator import (
    DEFAULT_MODELS,
    DEFAULT_START_LEADS,
    MODEL_CONFIGS,
    PRETRAINED_CACHE_DIR,
    _append_metric_rows,
    _select_hres_lead,
    align_to_synoptic_12h,
    compute_all_metrics,
    forecast_filename,
    load_aurora_025_model,
)
from src.dataloader import AuroraDataLoader
from src.forecaster import AuroraForecaster
from src.utils import (
    align_and_subset_2d,
    aurora_batch_to_xarray,
    filter_batch_for_pretrained,
    load_era5_zarr,
    load_hres_zarr,
    standardize_longitude,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

DEFAULT_YEARS: tuple[int, ...] = (2016, 2017, 2018, 2019, 2020)
DEFAULT_FORECAST_DIR = Path("ewb_insample_hw_forecasts")
DEFAULT_OUTPUT_CSV = Path("insample_heatwave_evaluations.csv")


@dataclass
class InSampleHeatwaveCase:
    """Metadata container matching EWB IndividualCase fields used by evaluators."""

    case_id_number: int
    title: str
    start_date: str
    end_date: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    event_type: str = "heat_wave"

    def __post_init__(self) -> None:
        self.location = ewb.regions.BoundingBoxRegion(
            latitude_min=self.lat_min,
            latitude_max=self.lat_max,
            longitude_min=self.lon_min,
            longitude_max=self.lon_max,
        )


# Case 29 comes directly from ExtremeWeatherBench events.yaml (only 2020 HW case).
# Cases 201-211 are major WMO/national-record 2016-2020 heatwaves inside both
# Aurora 0.25 training windows (1979-2020 / 2016-2021) and WB2 HRES (2016-2022).
INSAMPLE_HEATWAVE_CASES: tuple[InSampleHeatwaveCase, ...] = (
    InSampleHeatwaveCase(
        case_id_number=29,
        title="2020 Australia",
        start_date="2020-11-25 00:00:00",
        end_date="2020-12-01 00:00:00",
        lat_min=-36.25,
        lat_max=-22.25,
        lon_min=128.5,
        lon_max=153.5,
    ),
    InSampleHeatwaveCase(
        case_id_number=201,
        title="2020 Western US & Death Valley Record",
        start_date="2020-08-14 00:00:00",
        end_date="2020-08-22 00:00:00",
        lat_min=31.5,
        lat_max=42.5,
        lon_min=-123.75,
        lon_max=-104.75,
    ),
    InSampleHeatwaveCase(
        case_id_number=202,
        title="2020 Siberian Arctic Record",
        start_date="2020-06-15 00:00:00",
        end_date="2020-06-25 00:00:00",
        lat_min=60.0,
        lat_max=73.0,
        lon_min=120.0,
        lon_max=145.0,
    ),
    InSampleHeatwaveCase(
        case_id_number=203,
        title="2020 Western Europe August",
        start_date="2020-08-06 00:00:00",
        end_date="2020-08-14 00:00:00",
        lat_min=41.0,
        lat_max=54.0,
        lon_min=-5.0,
        lon_max=12.0,
    ),
    InSampleHeatwaveCase(
        case_id_number=204,
        title="2019 Western Europe July Record",
        start_date="2019-07-21 00:00:00",
        end_date="2019-07-28 00:00:00",
        lat_min=41.0,
        lat_max=55.0,
        lon_min=-4.0,
        lon_max=14.0,
    ),
    InSampleHeatwaveCase(
        case_id_number=205,
        title="2019 France & Southern Europe June Record",
        start_date="2019-06-24 00:00:00",
        end_date="2019-07-02 00:00:00",
        lat_min=36.0,
        lat_max=50.0,
        lon_min=-5.0,
        lon_max=12.0,
    ),
    InSampleHeatwaveCase(
        case_id_number=206,
        title="2019 India & Pakistan May-June",
        start_date="2019-05-28 00:00:00",
        end_date="2019-06-07 00:00:00",
        lat_min=20.75,
        lat_max=34.75,
        lon_min=66.0,
        lon_max=82.0,
    ),
    InSampleHeatwaveCase(
        case_id_number=207,
        title="2018 Japan & East Asia July Record",
        start_date="2018-07-14 00:00:00",
        end_date="2018-07-25 00:00:00",
        lat_min=31.0,
        lat_max=41.0,
        lon_min=126.25,
        lon_max=142.25,
    ),
    InSampleHeatwaveCase(
        case_id_number=208,
        title="2018 Northern Europe & Scandinavia",
        start_date="2018-07-22 00:00:00",
        end_date="2018-08-01 00:00:00",
        lat_min=52.0,
        lat_max=68.0,
        lon_min=4.0,
        lon_max=26.0,
    ),
    InSampleHeatwaveCase(
        case_id_number=209,
        title="2017 Southern Europe Lucifer",
        start_date="2017-08-01 00:00:00",
        end_date="2017-08-09 00:00:00",
        lat_min=36.0,
        lat_max=47.0,
        lon_min=8.0,
        lon_max=26.0,
    ),
    InSampleHeatwaveCase(
        case_id_number=210,
        title="2017 Southwest US June",
        start_date="2017-06-17 00:00:00",
        end_date="2017-06-26 00:00:00",
        lat_min=31.0,
        lat_max=40.0,
        lon_min=-120.0,
        lon_max=-108.0,
    ),
    InSampleHeatwaveCase(
        case_id_number=211,
        title="2016 India Phalodi Record",
        start_date="2016-05-15 00:00:00",
        end_date="2016-05-24 00:00:00",
        lat_min=20.75,
        lat_max=33.75,
        lon_min=67.0,
        lon_max=80.0,
    ),
)


def get_insample_heatwave_cases(
    years: Sequence[int] = DEFAULT_YEARS,
    max_cases: int | None = None,
) -> list[InSampleHeatwaveCase]:
    """
    Return curated 2016-2020 in-sample heatwave cases filtered by *years*.

    Sorted deterministically by (start_date, case_id_number).
    """
    target_years = set(int(y) for y in years)
    cases = [
        c
        for c in INSAMPLE_HEATWAVE_CASES
        if pd.to_datetime(c.start_date).year in target_years
    ]
    cases.sort(key=lambda c: (pd.to_datetime(c.start_date), c.case_id_number))
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


# =============================================================================
# Section 1: Forecast Generation
# =============================================================================


def generate_insample_heatwave_forecasts(
    output_dir: str | Path = DEFAULT_FORECAST_DIR,
    models: Sequence[str] = DEFAULT_MODELS,
    years: Sequence[int] = DEFAULT_YEARS,
    start_leads: Sequence[int] = DEFAULT_START_LEADS,
    cache_dir: str | Path = PRETRAINED_CACHE_DIR,
    max_cases: int | None = None,
) -> None:
    """
    Generate Aurora 0.25° forecasts for in-sample 2016-2020 heatwaves.

    Groups pending forecasts by (model_key, init_time) so concurrent cases share
    a single GPU rollout while saving isolated per-case NetCDF files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cases = get_insample_heatwave_cases(years=years, max_cases=max_cases)
    logger.info(
        f"Loaded {len(cases)} in-sample heatwave cases for years={list(years)}."
    )
    if not cases:
        logger.warning("No matching in-sample heatwave cases found. Exiting generation.")
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
                init_time = aligned_start - pd.Timedelta(hours=int(start_lead))
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
        logger.info("All requested in-sample Aurora forecasts already exist.")
        return

    data_pipeline = AuroraDataLoader(
        cache_dir=Path(cache_dir), pretrained_only=True, res=0.25
    )

    for model_key, grouped_by_init in pending_by_model.items():
        logger.info("=" * 60)
        logger.info(
            f"Generating in-sample forecasts for model={model_key} "
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
                    valid_time = init_time + pd.Timedelta(hours=lead_hours)

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
# Section 2: Direct Evaluation (Aurora 0.25 In-Sample vs. HRES & Persistence)
# =============================================================================


def print_evaluation_summaries(df: pd.DataFrame) -> None:
    """Print common-target in-sample summary (2016-2020) and per-year breakdown."""
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
                common, keys, on=["case_id_number", "init_time", "valid_time"], how="inner"
            )
        )

    if common is not None and not common.empty:
        df_common = pd.merge(
            df_valid, common, on=["case_id_number", "init_time", "valid_time"], how="inner"
        )
        summary_common = (
            df_common.groupby(["metric", "model_family"])["value"]
            .agg(["count", "mean", "std"])
        )
        logger.info(
            "\n=== In-Sample Common-Target Summary (2016-2020, <= 240h) ===\n"
            + summary_common.to_string()
        )

    summary_by_year = (
        df_valid.groupby(["year", "metric", "model_family"])["value"]
        .agg(["count", "mean", "std"])
    )
    logger.info(
        "\n=== In-Sample Breakdown by Year (<= 240h) ===\n"
        + summary_by_year.to_string()
    )


def evaluate_insample_heatwaves(
    forecast_dir: str | Path = DEFAULT_FORECAST_DIR,
    output_csv: str | Path = DEFAULT_OUTPUT_CSV,
    models: Sequence[str] = DEFAULT_MODELS,
    years: Sequence[int] = DEFAULT_YEARS,
    start_leads: Sequence[int] = DEFAULT_START_LEADS,
    max_cases: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate saved in-sample Aurora 0.25° forecasts against WeatherBench 2 HRES
    and ERA5 Persistence using ARCO ERA5 as ground truth.
    """
    forecast_path = Path(forecast_dir)
    if not forecast_path.exists():
        raise FileNotFoundError(
            f"Forecast directory {forecast_path.resolve()} does not exist. "
            "Run with --generate first."
        )

    logger.info("=" * 60)
    logger.info("Starting In-Sample Heatwave Evaluation Pipeline (2016-2020)")
    logger.info("=" * 60)

    era5_ds = load_era5_zarr(variable="temperature")
    hres_ds = load_hres_zarr(variable="temperature")
    cases = get_insample_heatwave_cases(years=years, max_cases=max_cases)
    logger.info(
        f"Evaluating {len(cases)} in-sample heatwave cases for years={list(years)}."
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
        era5_case = case.location.mask(era5_case_raw).compute()
        era5_case = standardize_longitude(era5_case)
        del era5_case_raw

        for start_lead in start_leads:
            start_lead = int(start_lead)
            init_t = aligned_start - pd.Timedelta(hours=start_lead)

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
                    lt_td = pd.to_timedelta(lt)
                    lt_hours = int(lt_td.total_seconds() / 3600)
                    valid_time = file_init_t + lt_td

                    if not (aligned_start <= valid_time <= end_dt):
                        continue

                    try:
                        tgt_2d = era5_case["surface_air_temperature"].sel(
                            time=valid_time.to_pydatetime()
                        )
                    except KeyError:
                        continue

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
            "Evaluate Aurora 0.25° (Pretrained and Finetuned) against HRES and "
            "ERA5 Persistence on in-sample (trained-on) 2016-2020 heatwaves."
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
        help="In-sample heatwave event years to include (default: 2016 2017 2018 2019 2020).",
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
        generate_insample_heatwave_forecasts(
            output_dir=args.forecast_dir,
            models=args.models,
            years=args.years,
            start_leads=args.start_leads,
            cache_dir=args.cache_dir,
            max_cases=args.max_cases,
        )
    if run_evaluate:
        evaluate_insample_heatwaves(
            forecast_dir=args.forecast_dir,
            output_csv=args.output_csv,
            models=args.models,
            years=args.years,
            start_leads=args.start_leads,
            max_cases=args.max_cases,
        )


if __name__ == "__main__":
    main()
