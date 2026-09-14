import logging
import os
import re
import warnings
from pathlib import Path
import gc
import pandas as pd
import numpy as np
import xarray as xr
import torch
import aiohttp
import extremeweatherbench as ewb

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataloader import AuroraDataLoader
from src.forecaster import AuroraForecaster
from aurora import AuroraV1p5

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "false"
os.environ["GCSFS_TOKEN"] = "anon"

class PrecipCase:
    def __init__(self, case_id, title, start_date, end_date, lat_min, lat_max, lon_min, lon_max):
        self.case_id_number = case_id
        self.title = title
        self.start_date = start_date
        self.end_date = end_date
        self.event_type = "extreme_precipitation"
        self.location = ewb.regions.BoundingBoxRegion(
            name=title, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max
        )

PRECIP_CASES = [
    PrecipCase(101, "Hurricane Harvey", "2017-08-25", "2017-08-30", 27.0, 32.0, 360-98.0, 360-93.0),
    PrecipCase(102, "2021 European Floods", "2021-07-12", "2021-07-16", 49.0, 52.0, 5.0, 8.5),
    PrecipCase(103, "2021 Henan Floods", "2021-07-17", "2021-07-23", 32.5, 36.5, 112.0, 116.0),
    PrecipCase(104, "2016 Louisiana Floods", "2016-08-10", "2016-08-15", 29.5, 31.5, 360-92.5, 360-89.0),
    PrecipCase(105, "2018 Kerala Floods", "2018-08-08", "2018-08-20", 8.0, 13.0, 74.5, 78.0),
    PrecipCase(106, "Typhoon Hagibis", "2019-10-11", "2019-10-13", 33.0, 39.0, 136.0, 142.0),
    PrecipCase(107, "Cyclone Idai", "2019-03-14", "2019-03-17", -22.0, -15.0, 33.0, 38.0),
    PrecipCase(108, "2019 Iran Floods", "2019-03-17", "2019-04-10", 29.0, 38.0, 47.0, 57.0),
    PrecipCase(109, "2020 China Floods", "2020-07-01", "2020-07-22", 27.0, 33.0, 108.0, 118.0),
    PrecipCase(110, "2022 Pakistan Floods", "2022-08-14", "2022-08-30", 24.0, 34.0, 65.0, 74.0),
]

# --- Pure Metric Functions ---
def peak_amplitude_error(forecast_2d: np.ndarray, target_2d: np.ndarray) -> float:
    if np.isnan(target_2d).all() or np.isnan(forecast_2d).all(): return np.nan
    return float(np.nanmax(forecast_2d) - np.nanmax(target_2d))

def conditional_bias_extremes(forecast_2d: np.ndarray, target_2d: np.ndarray, percentile: float = 95.0) -> float:
    if np.isnan(target_2d).all(): return np.nan
    threshold = np.nanpercentile(target_2d, percentile)
    mask = target_2d >= threshold
    if not np.any(mask): return np.nan
    return float(np.nanmean(forecast_2d[mask] - target_2d[mask]))

def rmse(forecast_2d: np.ndarray, target_2d: np.ndarray) -> float:
    if np.isnan(target_2d).all() or np.isnan(forecast_2d).all(): return np.nan
    diff = forecast_2d - target_2d
    return float(np.sqrt(np.nanmean(diff**2)))

def intersection_over_union(forecast_2d: np.ndarray, target_2d: np.ndarray, percentile: float = 95.0) -> float:
    if np.isnan(target_2d).all() or np.isnan(forecast_2d).all(): return np.nan
    threshold = np.nanpercentile(target_2d, percentile)
    f_mask = forecast_2d >= threshold
    t_mask = target_2d >= threshold
    intersection = (f_mask & t_mask).sum()
    union = (f_mask | t_mask).sum()
    return float(intersection / union) if union > 0 else np.nan

def extreme_area_ratio(forecast_2d: np.ndarray, target_2d: np.ndarray, percentile: float = 95.0) -> float:
    if np.isnan(target_2d).all() or np.isnan(forecast_2d).all(): return np.nan
    threshold = np.nanpercentile(target_2d, percentile)
    t_count = (target_2d >= threshold).sum()
    if t_count == 0: return np.nan
    f_count = (forecast_2d >= threshold).sum()
    return float(f_count / t_count)


# --- Generation Phase ---
def aurora_batch_to_xarray(predicted_batch, init_time, lead_time_hours):
    var_name = "scaled_tp_1h"
    if var_name not in predicted_batch.surf_vars:
        # Fallback just in case
        var_name = list(predicted_batch.surf_vars.keys())[0]
        
    tensor = predicted_batch.surf_vars[var_name].detach().cpu().numpy().squeeze()
    
    lats = predicted_batch.metadata.lat
    lons = predicted_batch.metadata.lon
    
    ds = xr.Dataset(
        data_vars={
            "scaled_tp_1h": (["init_time", "lead_time", "latitude", "longitude"], tensor[np.newaxis, np.newaxis, :, :]),
        },
        coords={
            "init_time": [pd.to_datetime(init_time)],
            "lead_time": pd.to_timedelta([f"{lead_time_hours}h"]),
            "latitude": lats,
            "longitude": lons,
        }
    )
    return ds

def generate_precip_forecasts(predict=True, output_dir="ewb_precip_forecasts/"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("Initializing DataLoader...")
    data_pipeline = AuroraDataLoader(cache_dir=Path("weather_data"))
    
    logger.info("Initializing Aurora Model...")
    aurora_model = AuroraV1p5()
    aurora_model.load_checkpoint("microsoft/aurora", "aurora-0.25-v1.5.ckpt", revision="main")
    forecaster = AuroraForecaster(model=aurora_model)
    
    target_start_lead_times = [24, 72, 120]
    
    for case in PRECIP_CASES:
        event_start = pd.to_datetime(case.start_date)
        event_end = pd.to_datetime(case.end_date)
        event_duration_hours = int((event_end - event_start).total_seconds() / 3600)
        
        for start_lead in target_start_lead_times:
            init_time = event_start - pd.Timedelta(hours=start_lead)
            filename = Path(output_dir) / f"aurora_precip_{init_time.strftime('%Y%m%d')}_L{start_lead}_{case.case_id_number}.nc"
            
            if filename.exists():
                logger.info(f"Skipping {case.title} at {start_lead}h start lead - File already exists")
                continue
                
            logger.info(f"Processing: {case.title} | Init: {init_time} | Start Lead: {start_lead}h")
            
            try:
                input_batch, _ = data_pipeline.get_batches(
                    init_time, history_steps=1, forecast_steps=0, bbox=None
                )
                if predict:
                    total_hours_to_roll = start_lead + event_duration_hours
                    total_steps_needed = int(total_hours_to_roll / 6)
                    
                    predicted_batches = forecaster.predict_rollout(
                        initial_batch=input_batch, steps=total_steps_needed, fine_lead_times=[6.0]
                    )
                    
                    start_index = int(start_lead / 6) - 1
                    start_index = max(0, start_index)
                    
                    event_ds_list = []
                    for i, batch in enumerate(predicted_batches[start_index:]):
                        current_lead_time = start_lead + (i * 6)
                        step_ds = aurora_batch_to_xarray(batch, init_time, current_lead_time)
                        event_ds_list.append(step_ds)
                        
                    if event_ds_list:
                        full_event_ds = xr.concat(event_ds_list, dim="lead_time")
                        full_event_ds.to_netcdf(filename, engine='h5netcdf')
                        logger.info(f"Saved {filename}")
                    
            except Exception as e:
                logger.error(f"Failed on {init_time} for lead {start_lead}h: {e}")

    logger.info("Cleaning up GPU resources...")
    del aurora_model
    del forecaster
    torch.cuda.empty_cache()
    gc.collect()


# --- Evaluation Phase ---
def standardize_longitude(ds: xr.Dataset) -> xr.Dataset:
    if "longitude" in ds.coords:
        new_lon = ds["longitude"] % 360
        ds = ds.assign_coords(longitude=new_lon)
        ds = ds.sortby("longitude")
    return ds

def align_and_subset_2d(fcst_da: xr.DataArray, tgt_da: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    fcst_sub = fcst_da.interp_like(tgt_da, method="nearest", kwargs={"fill_value": "extrapolate"})
    return fcst_sub.values, tgt_da.values

def load_era5_zarr() -> xr.Dataset:
    logger.info("Connecting to ARCO ERA5 zarr store...")
    ds = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3", storage_options={"token": "anon"})
    return ds[["total_precipitation"]]

def load_hres_zarr() -> xr.Dataset:
    logger.info("Connecting to HRES WeatherBench2 zarr store...")
    ds = xr.open_zarr("gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr", storage_options={"token": "anon"})
    ds = ds.rename({"prediction_timedelta": "lead_time", "time": "init_time"})
    # Need to verify if total_precipitation_6hr is the correct name
    if "total_precipitation_6hr" in ds.data_vars:
        return ds[["total_precipitation_6hr"]]
    elif "total_precipitation" in ds.data_vars:
        return ds[["total_precipitation"]]
    else:
        logger.warning(f"No obvious precip variable found in HRES. Available: {list(ds.data_vars.keys())}")
        return ds

def evaluate_precip_events(forecast_dir="ewb_precip_forecasts/", output_csv="precip_evaluations.csv"):
    logger.info("Starting Extreme Precipitation Evaluation Pipeline")
    era5_ds = load_era5_zarr()
    hres_ds = load_hres_zarr()
    
    # Identify which variable to use for HRES
    if "total_precipitation_6hr" in hres_ds.data_vars:
        hres_var = "total_precipitation_6hr"
    elif "total_precipitation" in hres_ds.data_vars:
        hres_var = "total_precipitation"
    else:
        hres_var = None
    
    forecast_path = Path(forecast_dir)
    nc_files = sorted(list(forecast_path.glob("*.nc")))
    if not nc_files:
        logger.warning(f"No .nc files found in {forecast_path}. Returning empty.")
        return

    results = []
    
    for case_idx, case in enumerate(PRECIP_CASES):
        logger.info(f"Evaluating Case {case.case_id_number}: {case.title} ({case_idx+1}/{len(PRECIP_CASES)})")
        
        era5_case = era5_ds.sel(time=slice(case.start_date, case.end_date))
        era5_case = case.location.mask(era5_case).compute()
        era5_case = standardize_longitude(era5_case)
        
        case_files = [f for f in nc_files if f"_{case.case_id_number}.nc" in f.name]
        
        for nc_file in case_files:
            match = re.search(r"_L(\d+)_", nc_file.name)
            lead_bucket = f"L{match.group(1)}" if match else "L?"
            model_name = f"aurora1.5_{lead_bucket}"
            
            aurora_fcst = xr.open_dataset(nc_file)
            init_t = aurora_fcst.init_time.values[0]
            lead_times = aurora_fcst.lead_time.values
            
            aurora_fcst = case.location.mask(aurora_fcst).compute()
            aurora_fcst = standardize_longitude(aurora_fcst)
            
            try:
                hres_fcst = hres_ds.sel(init_time=init_t, method="pad")
                hres_init_t = hres_fcst.init_time.values
                hres_fcst = case.location.mask(hres_fcst).compute()
                hres_fcst = standardize_longitude(hres_fcst)
                has_hres = True and hres_var is not None
            except KeyError:
                has_hres = False
                
            for lt in aurora_fcst.lead_time.values:
                valid_time = init_t + lt
                lt_hours = int(pd.Timedelta(lt).total_seconds() / 3600)
                
                if pd.to_datetime(case.start_date) <= pd.to_datetime(valid_time) <= pd.to_datetime(case.end_date):
                    try:
                        tgt_2d = era5_case["total_precipitation"].sel(time=valid_time)
                    except KeyError:
                        continue
                    
                    fcst_2d_aurora = aurora_fcst["scaled_tp_1h"].sel(init_time=init_t, lead_time=lt)
                    fcst_arr_aurora, tgt_arr_aurora = align_and_subset_2d(fcst_2d_aurora, tgt_2d)
                    
                    metrics_aurora = {
                        "Peak_Amplitude_Error": peak_amplitude_error(fcst_arr_aurora, tgt_arr_aurora),
                        "Conditional_Bias_Extremes": conditional_bias_extremes(fcst_arr_aurora, tgt_arr_aurora, 95.0),
                        "RootMeanSquaredError": rmse(fcst_arr_aurora, tgt_arr_aurora),
                        "Spatial_IOU": intersection_over_union(fcst_arr_aurora, tgt_arr_aurora, 95.0),
                        "Extreme_Area_Ratio": extreme_area_ratio(fcst_arr_aurora, tgt_arr_aurora, 95.0),
                    }
                    
                    for m_name, value in metrics_aurora.items():
                        results.append({
                            "case_id_number": case.case_id_number,
                            "event_type": "extreme_precipitation",
                            "forecast_source": model_name,
                            "target_source": "ERA5",
                            "metric": m_name,
                            "forecast_variable": "scaled_tp_1h",
                            "target_variable": "total_precipitation",
                            "init_time": init_t,
                            "valid_time": valid_time,
                            "lead_time": lt_hours,
                            "value": value
                        })
                        
                    if has_hres:
                        hres_lt = valid_time - hres_init_t
                        hres_lt_hours = int(pd.Timedelta(hres_lt).total_seconds() / 3600)
                        
                        try:
                            # HRES lead_time is integer hours
                            fcst_2d_hres = hres_fcst[hres_var].sel(lead_time=hres_lt_hours)
                        except KeyError:
                            pass
                        else:
                            fcst_arr_hres, tgt_arr_hres = align_and_subset_2d(fcst_2d_hres, tgt_2d)
                            metrics_hres = {
                                "Peak_Amplitude_Error": peak_amplitude_error(fcst_arr_hres, tgt_arr_hres),
                                "Conditional_Bias_Extremes": conditional_bias_extremes(fcst_arr_hres, tgt_arr_hres, 95.0),
                                "RootMeanSquaredError": rmse(fcst_arr_hres, tgt_arr_hres),
                                "Spatial_IOU": intersection_over_union(fcst_arr_hres, tgt_arr_hres, 95.0),
                                "Extreme_Area_Ratio": extreme_area_ratio(fcst_arr_hres, tgt_arr_hres, 95.0),
                            }
                            
                            for m_name, value in metrics_hres.items():
                                results.append({
                                    "case_id_number": case.case_id_number,
                                    "event_type": "extreme_precipitation",
                                    "forecast_source": "HRES",
                                    "target_source": "ERA5",
                                    "metric": m_name,
                                    "forecast_variable": hres_var,
                                    "target_variable": "total_precipitation",
                                    "init_time": hres_init_t,
                                    "valid_time": valid_time,
                                    "lead_time": hres_lt_hours,
                                    "value": value
                                })
                                
            aurora_fcst.close()
            
    df = pd.DataFrame(results)
    if not df.empty:
        col_order = ["value", "lead_time", "init_time", "valid_time", "target_variable", "metric", "forecast_source", "target_source", "case_id_number", "event_type", "forecast_variable"]
        for col in col_order:
            if col not in df.columns: df[col] = pd.NA
        df = df[col_order]
        df.to_csv(output_csv, index=False)
        logger.info(f"Results saved to {output_csv}")
        summary = df.dropna(subset=["value"]).groupby(["metric", "forecast_source"])["value"].agg(["count", "mean", "std"])
        logger.info(f"\n{summary.to_string()}")
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    generate_precip_forecasts()
    evaluate_precip_events()
