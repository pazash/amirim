import gc
import logging
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

# Suppress warnings that aren't useful during evaluation
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

# Use anonymous access for GCS (ERA5, HRES)
os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "false"
os.environ["GCSFS_TOKEN"] = "anon"

import extremeweatherbench as ewb

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from aurora import AuroraV1p5
from src.dataloader import AuroraDataLoader
from src.forecaster import AuroraForecaster
from src.utils import (
    AR_IVT_LEVELS,
    align_and_subset_2d,
    aurora_batch_to_ivt_xarray,
    compute_ivt_from_dataset,
    conditional_bias_extremes,
    extreme_area_ratio,
    intersection_over_union,
    load_era5_zarr,
    load_hres_zarr,
    peak_amplitude_error,
    rmse,
    standardize_longitude,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def get_ar_cases(max_year: int = 2022) -> list:
    """
    Load ExtremeWeatherBench atmospheric river cases filtered up to max_year
    (default 2022 so all cases have matching HRES data in WeatherBench2).
    """
    all_cases = ewb.cases.load_cases()
    ar_cases = [
        c
        for c in all_cases
        if c.event_type == "atmospheric_river"
        and pd.to_datetime(c.start_date).year <= max_year
        and pd.to_datetime(c.end_date).year <= max_year
    ]
    return ar_cases


# =============================================================================
# Section 1: Aurora Forecast Generation (with on-the-fly IVT calculation)
# =============================================================================


def generate_ar_forecasts(
    output_dir: str = "ewb_ar_forecasts/",
    max_year: int = 2022,
):
    """
    Generate Aurora V1.5 forecasts for EWB atmospheric river cases at 24h, 72h,
    and 120h lead times prior to event start, computing 2D IVT on-the-fly and
    saving to NetCDF.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ar_cases = get_ar_cases(max_year=max_year)
    logger.info(
        f"Found {len(ar_cases)} atmospheric river cases (<= {max_year}) for generation."
    )

    target_start_lead_times = [24, 72, 120]

    # Pre-check which forecasts need to be generated before loading the model
    to_generate = []
    for case in ar_cases:
        event_start = pd.to_datetime(case.start_date)
        event_end = pd.to_datetime(case.end_date)
        for start_lead in target_start_lead_times:
            init_time = event_start - pd.Timedelta(hours=start_lead)
            filename = (
                output_path
                / f"aurora_ar_forecast_{init_time.strftime('%Y%m%d')}_L{start_lead}_full_event.nc"
            )
            if filename.exists():
                logger.info(
                    f"Skipping {case.title} at {start_lead}h start lead - {filename.name} already exists"
                )
            else:
                to_generate.append((case, event_start, event_end, start_lead, init_time, filename))

    if not to_generate:
        logger.info("All atmospheric river forecasts already exist.")
        return

    logger.info(f"Need to generate {len(to_generate)} forecast files. Initializing Aurora...")
    data_pipeline = AuroraDataLoader(cache_dir=Path("weather_data"))

    aurora_model = AuroraV1p5()
    aurora_model.load_checkpoint(
        "microsoft/aurora", "aurora-0.25-v1.5.ckpt", revision="main"
    )
    forecaster = AuroraForecaster(model=aurora_model)

    for case, event_start, event_end, start_lead, init_time, filename in to_generate:
        if filename.exists():
            # In case an identical (init_time, start_lead) was just produced by an overlapping case
            logger.info(f"File {filename.name} already generated in this session, skipping.")
            continue

        event_duration_hours = int((event_end - event_start).total_seconds() / 3600)
        logger.info(
            f"Processing: {case.title} (Case {case.case_id_number}) | "
            f"Init: {init_time} | Start Lead: {start_lead}h | Duration: {event_duration_hours}h"
        )

        try:
            input_batch, _ = data_pipeline.get_batch(
                init_time,
                history_steps=1,
                forecast_steps=0,
                bbox=None,
            )

            total_hours_to_roll = start_lead + event_duration_hours
            total_steps_needed = int(total_hours_to_roll / 6)
            start_index = max(0, int(start_lead / 6) - 1)
            event_ds_list = []

            for step_idx, batch in enumerate(
                forecaster.stream_rollout(
                    initial_batch=input_batch,
                    steps=total_steps_needed,
                    fine_lead_times=[6.0],
                )
            ):
                if step_idx >= start_index:
                    i = step_idx - start_index
                    current_lead_time = start_lead + (i * 6)
                    step_ds = aurora_batch_to_ivt_xarray(
                        batch,
                        init_time=init_time,
                        lead_time_hours=current_lead_time,
                        levels=AR_IVT_LEVELS,
                    )
                    event_ds_list.append(step_ds)
                del batch

            if event_ds_list:
                full_event_ds = xr.concat(event_ds_list, dim="lead_time")
                full_event_ds.to_netcdf(filename, engine="h5netcdf")
                logger.info(f"Saved {filename}")
                del full_event_ds

            del input_batch, event_ds_list
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Failed on {init_time} for lead {start_lead}h: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info("Cleaning up GPU resources...")
    del aurora_model
    del forecaster
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# =============================================================================
# Section 2: Direct Evaluation Pipeline (Aurora & HRES vs ERA5)
# =============================================================================


def get_aurora_files(forecast_dir: str = "ewb_ar_forecasts") -> list[Path]:
    """Find all Aurora atmospheric river forecast NC files."""
    forecast_path = Path(forecast_dir)
    nc_files = sorted(list(forecast_path.glob("*.nc")))
    if not nc_files:
        raise FileNotFoundError(
            f"No .nc files found in {forecast_path.resolve()}. "
            "Run the forecast generation step first."
        )
    return nc_files


def evaluate_atmospheric_rivers(
    forecast_dir: str = "ewb_ar_forecasts",
    output_csv: str = "atmospheric_river_evaluations.csv",
    max_year: int = 2022,
) -> pd.DataFrame:
    """Run direct evaluation of atmospheric river cases, writing results to CSV."""
    logger.info("=" * 60)
    logger.info("Starting Direct Atmospheric River Evaluation Pipeline")
    logger.info("=" * 60)

    # 1. Setup Lazy Data Connections (u, v, q on common AR levels 300-1000 hPa)
    era5_ds = load_era5_zarr(variable="atmospheric_river")
    hres_ds = load_hres_zarr(variable="atmospheric_river")
    aurora_files = get_aurora_files(forecast_dir=forecast_dir)

    # 2. Get Case Metadata
    logger.info("Loading EWB atmospheric river case metadata...")
    ar_cases = get_ar_cases(max_year=max_year)
    logger.info(f"Found {len(ar_cases)} atmospheric river cases (<= {max_year}).")

    # 3. Setup Results List
    results = []

    # 4. Evaluation Loop
    for case_idx, case in enumerate(ar_cases):
        logger.info(
            f"\nEvaluating Case {case.case_id_number}: {case.title} ({case_idx+1}/{len(ar_cases)})"
        )
        logger.info(f"  Time: {case.start_date} to {case.end_date}")
        logger.info(f"  Location: {case.location}")

        if not isinstance(case.location, ewb.regions.BoundingBoxRegion):
            logger.warning(
                f"  Skipping case {case.case_id_number} (not a BoundingBoxRegion)."
            )
            continue

        # --- A. Download ERA5 Case Data & Compute IVT ---
        logger.info("  Downloading ERA5 subset and computing IVT for this case...")
        era5_case_raw = era5_ds.sel(time=slice(case.start_date, case.end_date))
        six_hr_mask = era5_case_raw.time.dt.hour.isin([0, 6, 12, 18])
        era5_case_raw = era5_case_raw.isel(time=np.where(six_hr_mask.values)[0])
        era5_case_raw = case.location.mask(era5_case_raw).compute()
        era5_case_raw = standardize_longitude(era5_case_raw)

        era5_ivt = compute_ivt_from_dataset(
            era5_case_raw, levels=AR_IVT_LEVELS
        ).compute()
        del era5_case_raw
        gc.collect()

        # --- B. Evaluate Aurora & Matched HRES ---
        logger.info("  Evaluating Aurora and Matched HRES...")
        start_dt = pd.to_datetime(case.start_date)
        end_dt = pd.to_datetime(case.end_date)

        for nc_file in aurora_files:
            match = re.search(r"_L(\d+)_", nc_file.name)
            lead_bucket = f"L{match.group(1)}" if match else "L?"
            model_name = f"aurora1.5_{lead_bucket}"

            aurora_fcst = xr.open_dataset(nc_file)
            init_t = aurora_fcst.init_time.values[0]
            lead_times = aurora_fcst.lead_time.values

            # FAST SKIP: Does this forecast overlap with the case window at all?
            valid_times = init_t + lead_times
            if not np.any((valid_times >= start_dt) & (valid_times <= end_dt)):
                aurora_fcst.close()
                continue

            # Load and spatially mask Aurora forecast
            aurora_fcst = case.location.mask(aurora_fcst).compute()
            aurora_fcst = standardize_longitude(aurora_fcst)

            if "integrated_vapor_transport" in aurora_fcst.data_vars:
                aurora_var = "integrated_vapor_transport"
            elif {"u", "v", "q"}.issubset(set(aurora_fcst.data_vars)):
                aurora_fcst["integrated_vapor_transport"] = compute_ivt_from_dataset(
                    aurora_fcst, u_var="u", v_var="v", q_var="q", levels=AR_IVT_LEVELS
                ).compute()
                aurora_var = "integrated_vapor_transport"
            else:
                aurora_var = [
                    v
                    for v in aurora_fcst.data_vars
                    if v not in ("latitude", "longitude", "level")
                ][0]

            # Fetch matching HRES for this exact init_time (or closest one BEFORE it)
            try:
                hres_fcst_raw = hres_ds.sel(init_time=init_t, method="pad")
                hres_init_t = hres_fcst_raw.init_time.values

                # Restrict HRES lead_times to only those falling inside the case window
                # before downloading 3D u, v, q from GCS
                if np.issubdtype(hres_fcst_raw.lead_time.dtype, np.timedelta64):
                    lead_td = pd.to_timedelta(hres_fcst_raw.lead_time.values)
                else:
                    lead_td = pd.to_timedelta(
                        hres_fcst_raw.lead_time.values, unit="h"
                    )
                hres_valid_times = pd.to_datetime(hres_init_t) + lead_td
                in_window_mask = np.asarray(
                    (hres_valid_times >= start_dt) & (hres_valid_times <= end_dt)
                )
                if np.any(in_window_mask):
                    hres_fcst_raw = hres_fcst_raw.isel(
                        lead_time=np.where(in_window_mask)[0]
                    )
                    hres_fcst_raw = case.location.mask(hres_fcst_raw).compute()
                    hres_fcst_raw = standardize_longitude(hres_fcst_raw)
                    hres_ivt = compute_ivt_from_dataset(
                        hres_fcst_raw, levels=AR_IVT_LEVELS
                    ).compute()
                    del hres_fcst_raw
                    has_hres = True
                else:
                    has_hres = False
                    logger.warning(
                        f"  HRES has no lead times overlapping window [{start_dt}, {end_dt}] for init_time {hres_init_t}"
                    )
            except KeyError as e:
                has_hres = False
                logger.warning(f"  No matching HRES found for init_time {init_t}: {e}")

            for lt in aurora_fcst.lead_time.values:
                valid_time = init_t + lt
                lt_hours = int(pd.Timedelta(lt).total_seconds() / 3600)

                # Only evaluate if valid_time is inside the case window
                if start_dt <= pd.to_datetime(valid_time) <= end_dt:
                    try:
                        tgt_2d = era5_ivt.sel(time=valid_time)
                    except KeyError:
                        continue

                    # 1. Evaluate Aurora
                    fcst_2d_aurora = aurora_fcst[aurora_var].sel(
                        init_time=init_t, lead_time=lt
                    )
                    fcst_arr_aurora, tgt_arr_aurora = align_and_subset_2d(
                        fcst_2d_aurora, tgt_2d
                    )

                    metrics_aurora = {
                        "Peak_Amplitude_Error": peak_amplitude_error(
                            fcst_arr_aurora, tgt_arr_aurora
                        ),
                        "Conditional_Bias_Extremes": conditional_bias_extremes(
                            fcst_arr_aurora, tgt_arr_aurora
                        ),
                        "RootMeanSquaredError": rmse(fcst_arr_aurora, tgt_arr_aurora),
                        "Spatial_IOU": intersection_over_union(
                            fcst_arr_aurora, tgt_arr_aurora
                        ),
                        "Extreme_Area_Ratio": extreme_area_ratio(
                            fcst_arr_aurora, tgt_arr_aurora
                        ),
                    }

                    for m_name, value in metrics_aurora.items():
                        results.append(
                            {
                                "case_id_number": case.case_id_number,
                                "event_type": "atmospheric_river",
                                "forecast_source": model_name,
                                "target_source": "ERA5",
                                "metric": m_name,
                                "forecast_variable": "integrated_vapor_transport",
                                "target_variable": "integrated_vapor_transport",
                                "init_time": init_t,
                                "valid_time": valid_time,
                                "lead_time": lt_hours,
                                "value": value,
                            }
                        )

                    # 2. Evaluate matched HRES
                    if has_hres:
                        hres_lt = valid_time - hres_init_t
                        hres_lt_hours = int(
                            pd.Timedelta(hres_lt).total_seconds() / 3600
                        )

                        try:
                            if np.issubdtype(
                                hres_ivt.lead_time.dtype, np.timedelta64
                            ):
                                fcst_2d_hres = hres_ivt.sel(
                                    lead_time=pd.to_timedelta(hres_lt)
                                )
                            else:
                                fcst_2d_hres = hres_ivt.sel(lead_time=hres_lt_hours)
                        except KeyError as e:
                            logger.warning(
                                f"  HRES missing lead_time {hres_lt_hours}h for init {hres_init_t}: {e}"
                            )
                        else:
                            fcst_arr_hres, tgt_arr_hres = align_and_subset_2d(
                                fcst_2d_hres, tgt_2d
                            )

                            metrics_hres = {
                                "Peak_Amplitude_Error": peak_amplitude_error(
                                    fcst_arr_hres, tgt_arr_hres
                                ),
                                "Conditional_Bias_Extremes": conditional_bias_extremes(
                                    fcst_arr_hres, tgt_arr_hres
                                ),
                                "RootMeanSquaredError": rmse(
                                    fcst_arr_hres, tgt_arr_hres
                                ),
                                "Spatial_IOU": intersection_over_union(
                                    fcst_arr_hres, tgt_arr_hres
                                ),
                                "Extreme_Area_Ratio": extreme_area_ratio(
                                    fcst_arr_hres, tgt_arr_hres
                                ),
                            }

                            for m_name, value in metrics_hres.items():
                                results.append(
                                    {
                                        "case_id_number": case.case_id_number,
                                        "event_type": "atmospheric_river",
                                        "forecast_source": "HRES",
                                        "target_source": "ERA5",
                                        "metric": m_name,
                                        "forecast_variable": "integrated_vapor_transport",
                                        "target_variable": "integrated_vapor_transport",
                                        "init_time": hres_init_t,
                                        "valid_time": valid_time,
                                        "lead_time": hres_lt_hours,
                                        "value": value,
                                    }
                                )

            aurora_fcst.close()
            if has_hres:
                del hres_ivt
            gc.collect()
        del era5_ivt
        gc.collect()

    # 5. Save Results
    logger.info(f"\nEvaluation complete. Generated {len(results)} rows of data.")

    if results:
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
        ]
        for col in col_order:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[col_order]

        output_path = Path(output_csv)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path.resolve()}")

        logger.info("\n--- Summary by Metric and Forecast Source (Non-NaN counts) ---")
        summary = (
            df.dropna(subset=["value"])
            .groupby(["metric", "forecast_source"])["value"]
            .agg(["count", "mean", "std"])
        )
        logger.info(f"\n{summary.to_string()}")

        return df
    else:
        logger.warning(
            "No results were generated! Check if forecasts overlap with case dates."
        )
        return pd.DataFrame()


if __name__ == "__main__":
    generate_ar_forecasts()
    evaluate_atmospheric_rivers()

