import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Optional

import aiohttp
import numpy as np
import pandas as pd
import xarray as xr
from dask import array as da

# Suppress warnings that aren't useful during evaluation
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")
warnings.filterwarnings("ignore", message="Mean of empty slice")

import extremeweatherbench as ewb

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Use anonymous access for GCS (ERA5, HRES)
os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "false"
os.environ["GCSFS_TOKEN"] = "anon"

# =============================================================================
# Section 1: Pure Metric Functions
# =============================================================================
# All metric functions take a 2D numpy/dask array of forecast and target data 
# on the exact same spatial grid and return a float scalar.


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


def rmse(forecast_2d: np.ndarray, target_2d: np.ndarray) -> float:
    """Root mean squared error over spatial domain."""
    if np.isnan(target_2d).all() or np.isnan(forecast_2d).all():
        return np.nan
    diff = forecast_2d - target_2d
    return float(np.sqrt(np.nanmean(diff**2)))


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


# =============================================================================
# Section 2: Data Loading & Memory-Efficient Alignment Helpers
# =============================================================================


def standardize_longitude(ds: xr.Dataset) -> xr.Dataset:
    """Standardize dataset longitude to [0, 360] and sort coordinates."""
    if "longitude" in ds.coords:
        new_lon = ds["longitude"] % 360
        ds = ds.assign_coords(longitude=new_lon)
        ds = ds.sortby("longitude")
    return ds


def load_era5_zarr() -> xr.Dataset:
    """Load ARCO ERA5 zarr store. Returns the full lazy dataset."""
    logger.info("Connecting to ARCO ERA5 zarr store...")
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        storage_options={"token": "anon"},
    )
    # ERA5 ARCO variables
    ds = ds.rename({"2m_temperature": "surface_air_temperature"})
    return ds[["surface_air_temperature"]]


def load_hres_zarr() -> xr.Dataset:
    """Load HRES WeatherBench2 zarr store. Returns the full lazy dataset."""
    logger.info("Connecting to HRES WeatherBench2 zarr store...")
    ds = xr.open_zarr(
        "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
        storage_options={"token": "anon"},
    )
    ds = ds.rename(
        {
            "2m_temperature": "surface_air_temperature",
            "prediction_timedelta": "lead_time",
            "time": "init_time",
        }
    )
    return ds[["surface_air_temperature"]]


def get_aurora_files(forecast_dir: str = "ewb_forecasts") -> list[Path]:
    """Find all Aurora forecast NC files."""
    forecast_path = Path(forecast_dir)
    nc_files = sorted(list(forecast_path.glob("*.nc")))
    if not nc_files:
        raise FileNotFoundError(
            f"No .nc files found in {forecast_path.resolve()}. "
            "Run the forecast generation script first."
        )
    return nc_files


def align_and_subset_2d(
    fcst_da: xr.DataArray,
    tgt_da: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray]:
    """Align forecast to the target grid."""
    # Both arrays are already standardized and spatially subsetted to the bounding box.
    # We just need to align their exact grid points.
    fcst_sub = fcst_da.interp_like(tgt_da, method="nearest", kwargs={"fill_value": "extrapolate"})
    
    return fcst_sub.values, tgt_da.values


# =============================================================================
# Section 3: Evaluation Pipeline (Direct)
# =============================================================================


def evaluate_heatwaves(output_csv: str = "heatwave_evaluations.csv"):
    """Run direct evaluation of heatwave cases, writing results to CSV."""
    logger.info("=" * 60)
    logger.info("Starting Direct Heatwave Evaluation Pipeline")
    logger.info("=" * 60)

    # 1. Setup Lazy Data Connections
    era5_ds = load_era5_zarr()
    hres_ds = load_hres_zarr()
    aurora_files = get_aurora_files()

    # 2. Get Case Metadata
    logger.info("Loading EWB case metadata...")
    all_cases = ewb.load_cases()
    hw_cases = [[c for c in all_cases if c.event_type == "heat_wave"][0]]
    logger.info(f"Found {len(hw_cases)} heatwave cases.")

    # 3. Setup Results List
    results = []

    # 4. Evaluation Loop
    for case_idx, case in enumerate(hw_cases):
        logger.info(f"\nEvaluating Case {case.case_id_number} ({case_idx+1}/{len(hw_cases)})")
        logger.info(f"  Time: {case.start_date} to {case.end_date}")
        logger.info(f"  Location: {case.location}")

        # Ensure location is a BoundingBoxRegion for this simple pipeline
        if not isinstance(case.location, ewb.regions.BoundingBoxRegion):
            logger.warning(f"  Skipping case {case.case_id_number} (not a BoundingBoxRegion).")
            continue

        # Valid times for this case (typically 6-hourly or 12-hourly for HRES)
        # HRES has 12h resolution, ERA5 has 1h/6h, Aurora has 6h. 
        # We evaluate at every forecast valid_time that falls within the case window.

        # --- A. Download ERA5 Case Data ---
        logger.info("  Downloading ERA5 subset for this case...")
        era5_case = era5_ds.sel(time=slice(case.start_date, case.end_date))
        # Mask spatially, download to memory, then standardize
        era5_case = case.location.mask(era5_case).compute()
        era5_case = standardize_longitude(era5_case)

        # --- B. Evaluate Aurora & Matched HRES ---
        logger.info("  Evaluating Aurora and Matched HRES...")
        for nc_file in aurora_files:
            match = re.search(r"_L(\d+)_", nc_file.name)
            lead_bucket = f"L{match.group(1)}" if match else "L?"
            model_name = f"aurora1.5_{lead_bucket}"
            
            aurora_fcst = xr.open_dataset(nc_file)
            init_t = aurora_fcst.init_time.values[0]
            lead_times = aurora_fcst.lead_time.values
            
            # FAST SKIP: Does this forecast overlap with the case window at all?
            valid_times = init_t + lead_times
            start_dt = pd.to_datetime(case.start_date)
            end_dt = pd.to_datetime(case.end_date)
            if not np.any((valid_times >= start_dt) & (valid_times <= end_dt)):
                aurora_fcst.close()
                continue
            
            # Load and spatially mask to memory
            aurora_fcst = case.location.mask(aurora_fcst).compute()
            aurora_fcst = standardize_longitude(aurora_fcst)
            
            if "2t_aurora" in aurora_fcst.data_vars:
                aurora_var = "2t_aurora"
            else:
                aurora_var = [v for v in aurora_fcst.data_vars if v != "latitude" and v != "longitude"][0]
            
            # Fetch matching HRES for this exact init_time (or the closest one BEFORE it)
            # Aurora often initializes at 18:00, but HRES only has 00:00 and 12:00.
            try:
                hres_fcst = hres_ds.sel(init_time=init_t, method="pad")
                hres_init_t = hres_fcst.init_time.values
                hres_fcst = case.location.mask(hres_fcst).compute()
                hres_fcst = standardize_longitude(hres_fcst)
                has_hres = True
            except KeyError:
                has_hres = False
                logger.info(f"  No matching HRES found for init_time {init_t}")

            for lt in aurora_fcst.lead_time.values:
                valid_time = init_t + lt
                
                # Convert lead_time to total hours as an integer
                lt_hours = int(pd.Timedelta(lt).total_seconds() / 3600)
                
                # Only evaluate if valid_time is inside the case window
                if pd.to_datetime(case.start_date) <= pd.to_datetime(valid_time) <= pd.to_datetime(case.end_date):
                    try:
                        tgt_2d = era5_case["surface_air_temperature"].sel(time=valid_time)
                    except KeyError:
                        continue
                    
                    # 1. Evaluate Aurora
                    fcst_2d_aurora = aurora_fcst[aurora_var].sel(init_time=init_t, lead_time=lt)
                    fcst_arr_aurora, tgt_arr_aurora = align_and_subset_2d(fcst_2d_aurora, tgt_2d)
                    
                    metrics_aurora = {
                        "Peak_Amplitude_Error": peak_amplitude_error(fcst_arr_aurora, tgt_arr_aurora),
                        "Conditional_Bias_Extremes": conditional_bias_extremes(fcst_arr_aurora, tgt_arr_aurora),
                        "RootMeanSquaredError": rmse(fcst_arr_aurora, tgt_arr_aurora),
                        "Spatial_IOU": intersection_over_union(fcst_arr_aurora, tgt_arr_aurora),
                        "Extreme_Area_Ratio": extreme_area_ratio(fcst_arr_aurora, tgt_arr_aurora),
                    }
                    
                    for m_name, value in metrics_aurora.items():
                        results.append({
                            "case_id_number": case.case_id_number,
                            "event_type": "heat_wave",
                            "forecast_source": model_name,
                            "target_source": "ERA5",
                            "metric": m_name,
                            "forecast_variable": "surface_air_temperature",
                            "target_variable": "surface_air_temperature",
                            "init_time": init_t,
                            "valid_time": valid_time,
                            "lead_time": lt_hours,
                            "value": value
                        })
                        
                    # 2. Evaluate matched HRES
                    if has_hres:
                        # Determine the required HRES lead time to reach this valid_time
                        hres_lt = valid_time - hres_init_t
                        hres_lt_hours = int(pd.Timedelta(hres_lt).total_seconds() / 3600)
                        
                        try:
                            # Try to get the exact matching valid_time from HRES
                            fcst_2d_hres = hres_fcst["surface_air_temperature"].sel(lead_time=hres_lt)
                        except KeyError:
                            pass
                        else:
                            fcst_arr_hres, tgt_arr_hres = align_and_subset_2d(fcst_2d_hres, tgt_2d)
                            
                            metrics_hres = {
                                "Peak_Amplitude_Error": peak_amplitude_error(fcst_arr_hres, tgt_arr_hres),
                                "Conditional_Bias_Extremes": conditional_bias_extremes(fcst_arr_hres, tgt_arr_hres),
                                "RootMeanSquaredError": rmse(fcst_arr_hres, tgt_arr_hres),
                                "Spatial_IOU": intersection_over_union(fcst_arr_hres, tgt_arr_hres),
                                "Extreme_Area_Ratio": extreme_area_ratio(fcst_arr_hres, tgt_arr_hres),
                            }
                            
                            for m_name, value in metrics_hres.items():
                                results.append({
                                    "case_id_number": case.case_id_number,
                                    "event_type": "heat_wave",
                                    "forecast_source": "HRES",
                                    "target_source": "ERA5",
                                    "metric": m_name,
                                    "forecast_variable": "surface_air_temperature",
                                    "target_variable": "surface_air_temperature",
                                    "init_time": hres_init_t,
                                    "valid_time": valid_time,
                                    "lead_time": hres_lt_hours,
                                    "value": value
                                })
                        
            aurora_fcst.close()

    # 5. Save Results
    logger.info(f"\nEvaluation complete. Generated {len(results)} rows of data.")
    
    if results:
        df = pd.DataFrame(results)
        
        # Ensure correct column order to match what users expect
        col_order = [
            "value", "lead_time", "init_time", "valid_time", 
            "target_variable", "metric", "forecast_source", "target_source", 
            "case_id_number", "event_type", "forecast_variable"
        ]
        # Reorder and add any missing columns just in case
        for col in col_order:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[col_order]
        
        output_path = Path(output_csv)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path.resolve()}")
        
        # Print summary
        logger.info("\n--- Summary by Metric and Forecast Source (Non-NaN counts) ---")
        summary = df.dropna(subset=["value"]).groupby(["metric", "forecast_source"])["value"].agg(
            ["count", "mean", "std"]
        )
        logger.info(f"\n{summary.to_string()}")
        
        return df
    else:
        logger.warning("No results were generated! Check if forecasts overlap with case dates.")
        return pd.DataFrame()


if __name__ == "__main__":
    evaluate_heatwaves()
