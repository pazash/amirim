import gc
import logging
import os
import re
import sys
import warnings
from pathlib import Path

import aiohttp
import extremeweatherbench as ewb
import numpy as np
import pandas as pd
import torch
import xarray as xr

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from aurora import AuroraV1p5
from src.dataloader import AuroraDataLoader
from src.forecaster import AuroraForecaster
from src.utils import (
    TCWV_LEVELS,
    align_and_subset_2d,
    aurora_batch_to_tcwv_xarray,
    compute_tcwv_from_dataset,
    conditional_bias_extremes,
    extreme_area_ratio,
    intersection_over_union,
    load_era5_zarr,
    load_hres_zarr,
    peak_amplitude_error,
    rmse,
    standardize_longitude,
)

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "false"
os.environ["GCSFS_TOKEN"] = "anon"


class PrecipCase:
    def __init__(
        self, case_id, title, start_date, end_date, lat_min, lat_max, lon_min, lon_max
    ):
        self.case_id_number = case_id
        self.title = title
        self.start_date = start_date
        self.end_date = end_date
        self.event_type = "extreme_precipitation"
        self.location = ewb.regions.BoundingBoxRegion(
            latitude_min=lat_min,
            latitude_max=lat_max,
            longitude_min=lon_min,
            longitude_max=lon_max,
        )


PRECIP_CASES = [
    PrecipCase(101, "Hurricane Harvey", "2017-08-25", "2017-08-30", 27.0, 32.0, 360 - 98.0, 360 - 93.0),
    PrecipCase(102, "2021 European Floods", "2021-07-12", "2021-07-16", 49.0, 52.0, 5.0, 8.5),
    PrecipCase(103, "2021 Henan Floods", "2021-07-17", "2021-07-23", 32.5, 36.5, 112.0, 116.0),
    PrecipCase(104, "2016 Louisiana Floods", "2016-08-10", "2016-08-15", 29.5, 31.5, 360 - 92.5, 360 - 89.0),
    PrecipCase(105, "2018 Kerala Floods", "2018-08-08", "2018-08-20", 8.0, 13.0, 74.5, 78.0),
    PrecipCase(106, "Typhoon Hagibis", "2019-10-11", "2019-10-13", 33.0, 39.0, 136.0, 142.0),
    PrecipCase(107, "Cyclone Idai", "2019-03-14", "2019-03-17", -22.0, -15.0, 33.0, 38.0),
    PrecipCase(108, "2019 Iran Floods", "2019-03-17", "2019-04-10", 29.0, 38.0, 47.0, 57.0),
    PrecipCase(109, "2020 China Floods", "2020-07-01", "2020-07-22", 27.0, 33.0, 108.0, 118.0),
    PrecipCase(110, "2022 Pakistan Floods", "2022-08-14", "2022-08-30", 24.0, 34.0, 65.0, 74.0),
]


def generate_tcwv_forecasts(
    predict: bool = True, output_dir: str = "ewb_tcwv_forecasts/"
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Initializing DataLoader...")
    data_pipeline = AuroraDataLoader(cache_dir=Path("weather_data"))

    logger.info("Initializing Aurora Model...")
    aurora_model = AuroraV1p5()
    aurora_model.load_checkpoint(
        "microsoft/aurora", "aurora-0.25-v1.5.ckpt", revision="main"
    )
    forecaster = AuroraForecaster(model=aurora_model)

    target_start_lead_times = [24, 72, 120]

    for case in PRECIP_CASES:
        event_start = pd.to_datetime(case.start_date)
        event_end = pd.to_datetime(case.end_date)
        event_duration_hours = int((event_end - event_start).total_seconds() / 3600)

        for start_lead in target_start_lead_times:
            init_time = event_start - pd.Timedelta(hours=start_lead)
            filename = (
                Path(output_dir)
                / f"aurora_tcwv_{init_time.strftime('%Y%m%d')}_L{start_lead}_{case.case_id_number}.nc"
            )

            if filename.exists():
                logger.info(
                    f"Skipping {case.title} at {start_lead}h start lead - File already exists"
                )
                continue

            logger.info(
                f"Processing: {case.title} | Init: {init_time} | Start Lead: {start_lead}h"
            )

            try:
                input_batch, _ = data_pipeline.get_batch(
                    init_time, history_steps=1, forecast_steps=0, bbox=None
                )
                if predict:
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
                            step_ds = aurora_batch_to_tcwv_xarray(
                                batch,
                                init_time=init_time,
                                lead_time_hours=current_lead_time,
                                levels=TCWV_LEVELS,
                            )
                            event_ds_list.append(step_ds)
                        del batch

                    if event_ds_list:
                        full_event_ds = xr.concat(event_ds_list, dim="lead_time")
                        full_event_ds.to_netcdf(filename, engine="h5netcdf")
                        logger.info(f"Saved {filename}")
                        del full_event_ds

                    del event_ds_list
                del input_batch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Failed on {init_time} for lead {start_lead}h: {e}")

    logger.info("Cleaning up GPU resources...")
    del aurora_model
    del forecaster
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def evaluate_tcwv_events(
    forecast_dir: str = "ewb_tcwv_forecasts/",
    output_csv: str = "tcwv_evaluations.csv",
):
    logger.info("Starting Extreme Precipitation TCWV Evaluation Pipeline")
    era5_ds = load_era5_zarr(variable="tcwv")
    hres_ds = load_hres_zarr(variable="tcwv")

    forecast_path = Path(forecast_dir)
    nc_files = sorted(list(forecast_path.glob("*.nc")))
    if not nc_files:
        logger.warning(f"No .nc files found in {forecast_path}. Returning empty.")
        return

    results = []

    for case_idx, case in enumerate(PRECIP_CASES):
        logger.info(
            f"Evaluating Case {case.case_id_number}: {case.title} ({case_idx+1}/{len(PRECIP_CASES)})"
        )

        # Download ERA5 specific_humidity for this case at 6-hourly intervals and vertically integrate to TCWV
        era5_case_raw = era5_ds.sel(time=slice(case.start_date, case.end_date))
        six_hr_mask = era5_case_raw.time.dt.hour.isin([0, 6, 12, 18])
        era5_case_raw = era5_case_raw.isel(time=np.where(six_hr_mask.values)[0])
        era5_case_raw = case.location.mask(era5_case_raw).compute()
        era5_case_raw = standardize_longitude(era5_case_raw)

        era5_tcwv = compute_tcwv_from_dataset(
            era5_case_raw, q_var="specific_humidity", levels=TCWV_LEVELS
        ).compute()
        del era5_case_raw
        gc.collect()

        case_files = [f for f in nc_files if f"_{case.case_id_number}.nc" in f.name]
        start_dt = pd.to_datetime(case.start_date)
        end_dt = pd.to_datetime(case.end_date)

        for nc_file in case_files:
            match = re.search(r"_L(\d+)_", nc_file.name)
            lead_bucket = f"L{match.group(1)}" if match else "L?"
            model_name = f"aurora1.5_{lead_bucket}"

            aurora_fcst = xr.open_dataset(nc_file)
            init_t = aurora_fcst.init_time.values[0]

            aurora_fcst = case.location.mask(aurora_fcst).compute()
            aurora_fcst = standardize_longitude(aurora_fcst)

            if "total_column_water_vapour" in aurora_fcst.data_vars:
                aurora_var = "total_column_water_vapour"
            elif "q" in aurora_fcst.data_vars:
                aurora_fcst["total_column_water_vapour"] = compute_tcwv_from_dataset(
                    aurora_fcst, q_var="q", levels=TCWV_LEVELS
                ).compute()
                aurora_var = "total_column_water_vapour"
            else:
                aurora_var = [
                    v
                    for v in aurora_fcst.data_vars
                    if v not in ("latitude", "longitude", "level")
                ][0]

            try:
                hres_fcst_raw = hres_ds.sel(init_time=init_t, method="pad")
                hres_init_t = hres_fcst_raw.init_time.values

                # Slice HRES lead_times to the event window before downloading 3D specific_humidity.
                # WeatherBench2 HRES lead_time can be stored as integer hours (0, 6, ..., 240)
                # or timedelta64 depending on xarray's decode_timedelta behavior.
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
                    hres_tcwv = compute_tcwv_from_dataset(
                        hres_fcst_raw, q_var="specific_humidity", levels=TCWV_LEVELS
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
                logger.warning(
                    f"  No matching HRES found for init_time {init_t}: {e}"
                )

            for lt in aurora_fcst.lead_time.values:
                valid_time = init_t + lt
                lt_hours = int(pd.Timedelta(lt).total_seconds() / 3600)

                if start_dt <= pd.to_datetime(valid_time) <= end_dt:
                    try:
                        tgt_2d = era5_tcwv.sel(time=valid_time)
                    except KeyError:
                        continue

                    current_dt = valid_time
                    current_lead_time_hours = int(
                        (pd.to_datetime(current_dt) - pd.to_datetime(init_t)).total_seconds()
                        / 3600
                    )
                    fcst_2d_aurora = aurora_fcst[aurora_var].sel(
                        init_time=init_t,
                        lead_time=pd.to_timedelta(f"{current_lead_time_hours}h"),
                    )
                    fcst_arr_aurora, tgt_arr_aurora = align_and_subset_2d(
                        fcst_2d_aurora, tgt_2d
                    )

                    metrics_aurora = {
                        "Peak_Amplitude_Error": peak_amplitude_error(
                            fcst_arr_aurora, tgt_arr_aurora
                        ),
                        "Conditional_Bias_Extremes": conditional_bias_extremes(
                            fcst_arr_aurora, tgt_arr_aurora, 95.0
                        ),
                        "RootMeanSquaredError": rmse(fcst_arr_aurora, tgt_arr_aurora),
                        "Spatial_IOU": intersection_over_union(
                            fcst_arr_aurora, tgt_arr_aurora, 95.0
                        ),
                        "Extreme_Area_Ratio": extreme_area_ratio(
                            fcst_arr_aurora, tgt_arr_aurora, 95.0
                        ),
                    }

                    for m_name, value in metrics_aurora.items():
                        results.append(
                            {
                                "case_id_number": case.case_id_number,
                                "event_type": "extreme_precipitation",
                                "forecast_source": model_name,
                                "target_source": "ERA5",
                                "metric": m_name,
                                "forecast_variable": "total_column_water_vapour",
                                "target_variable": "total_column_water_vapour",
                                "init_time": init_t,
                                "valid_time": valid_time,
                                "lead_time": lt_hours,
                                "value": value,
                            }
                        )

                    if has_hres:
                        hres_lt = valid_time - hres_init_t
                        hres_lt_hours = int(
                            pd.Timedelta(hres_lt).total_seconds() / 3600
                        )

                        try:
                            if np.issubdtype(
                                hres_tcwv.lead_time.dtype, np.timedelta64
                            ):
                                fcst_2d_hres = hres_tcwv.sel(
                                    lead_time=pd.to_timedelta(hres_lt)
                                )
                            else:
                                fcst_2d_hres = hres_tcwv.sel(lead_time=hres_lt_hours)
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
                                    fcst_arr_hres, tgt_arr_hres, 95.0
                                ),
                                "RootMeanSquaredError": rmse(
                                    fcst_arr_hres, tgt_arr_hres
                                ),
                                "Spatial_IOU": intersection_over_union(
                                    fcst_arr_hres, tgt_arr_hres, 95.0
                                ),
                                "Extreme_Area_Ratio": extreme_area_ratio(
                                    fcst_arr_hres, tgt_arr_hres, 95.0
                                ),
                            }

                            for m_name, value in metrics_hres.items():
                                results.append(
                                    {
                                        "case_id_number": case.case_id_number,
                                        "event_type": "extreme_precipitation",
                                        "forecast_source": "HRES",
                                        "target_source": "ERA5",
                                        "metric": m_name,
                                        "forecast_variable": "total_column_water_vapour",
                                        "target_variable": "total_column_water_vapour",
                                        "init_time": hres_init_t,
                                        "valid_time": valid_time,
                                        "lead_time": hres_lt_hours,
                                        "value": value,
                                    }
                                )

            aurora_fcst.close()
            if has_hres:
                del hres_tcwv
            gc.collect()
        del era5_tcwv
        gc.collect()

    df = pd.DataFrame(results)
    if not df.empty:
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
        df.to_csv(output_csv, index=False)
        logger.info(f"Results saved to {output_csv}")
        summary = (
            df.dropna(subset=["value"])
            .groupby(["metric", "forecast_source"])["value"]
            .agg(["count", "mean", "std"])
        )
        logger.info(f"\n{summary.to_string()}")
    else:
        logger.warning("No results generated.")


if __name__ == "__main__":
    generate_tcwv_forecasts()
    evaluate_tcwv_events()

