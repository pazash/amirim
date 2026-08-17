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


def load_era5_zarr() -> xr.Dataset:
    """Load ARCO ERA5 zarr store. Returns the full lazy dataset."""
    logger.info("Connecting to ARCO ERA5 zarr store...")
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        storage_options={"token": "anon"},
    )
    # ERA5 ARCO variables
    ds = ds.rename({"temperature_2m": "surface_air_temperature"})
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
    bbox: "ewb.regions.BoundingBox",
) -> tuple[np.ndarray, np.ndarray]:
    """Subset arrays to the bounding box and ensure grids perfectly align."""
    # Subset target to bbox (EWB target spatial masking logic)
    tgt_sub = bbox.mask(tgt_da, drop=True)
    
    # Align forecast to the exact grid of the subsetted target
    # This interpolates/slices the forecast to exactly match the target's lat/lon
    fcst_sub = fcst_da.interp_like(tgt_sub, method="nearest", kwargs={"fill_value": "extrapolate"})
    
    return fcst_sub.values, tgt_sub.values


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
    hw_cases = [c for c in all_cases if c.event_type == "heat_wave"]
    logger.info(f"Found {len(hw_cases)} heatwave cases.")

    # 3. Setup Results List
    results = []

    # 4. Evaluation Loop
    for case_idx, case in enumerate(hw_cases):
        logger.info(f"\nEvaluating Case {case.case_id_number} ({case_idx+1}/{len(hw_cases)})")
        logger.info(f"  Time: {case.start_date} to {case.end_date}")
        logger.info(f"  Location: {case.location}")

        # Ensure location is a BoundingBox for this simple pipeline
        if not isinstance(case.location, ewb.regions.BoundingBox):
            logger.warning(f"  Skipping case {case.case_id_number} (not a BoundingBox).")
            continue

        # Valid times for this case (typically 6-hourly or 12-hourly for HRES)
        # HRES has 12h resolution, ERA5 has 1h/6h, Aurora has 6h. 
        # We evaluate at every forecast valid_time that falls within the case window.

        # --- A. Evaluate HRES ---
        logger.info("  Evaluating HRES...")
        # Find HRES init_times that could potentially have valid_times in the window
        # Max HRES lead time is ~10 days
        hres_init_start = pd.to_datetime(case.start_date) - pd.Timedelta(days=15)
        hres_init_end = pd.to_datetime(case.end_date)
        
        # Subset HRES to candidate init_times to keep memory low
        try:
            hres_case_subset = hres_ds.sel(init_time=slice(hres_init_start, hres_init_end))
        except Exception as e:
            logger.warning(f"  Could not subset HRES for case {case.case_id_number}: {e}")
            hres_case_subset = None

        if hres_case_subset is not None and len(hres_case_subset.init_time) > 0:
            for init_t in hres_case_subset.init_time.values:
                # Get the forecast for this specific init time
                fcst_init = hres_case_subset.sel(init_time=init_t).load()
                
                for lt in fcst_init.lead_time.values:
                    valid_time = init_t + lt
                    if pd.to_datetime(case.start_date) <= pd.to_datetime(valid_time) <= pd.to_datetime(case.end_date):
                        # Extract the 2D spatial slice
                        fcst_2d = fcst_init["surface_air_temperature"].sel(lead_time=lt)
                        try:
                            tgt_2d = era5_ds["surface_air_temperature"].sel(time=valid_time).load()
                        except KeyError:
                            # ERA5 might not have this exact timestamp if it's outside its range
                            continue
                        
                        # Align and subset to bounding box
                        fcst_arr, tgt_arr = align_and_subset_2d(fcst_2d, tgt_2d, case.location)
                        
                        # Compute metrics
                        metrics_computed = {
                            "Peak_Amplitude_Error": peak_amplitude_error(fcst_arr, tgt_arr),
                            "Conditional_Bias_Extremes": conditional_bias_extremes(fcst_arr, tgt_arr),
                            "RootMeanSquaredError": rmse(fcst_arr, tgt_arr),
                            "Spatial_IOU": intersection_over_union(fcst_arr, tgt_arr),
                            "Extreme_Area_Ratio": extreme_area_ratio(fcst_arr, tgt_arr),
                        }
                        
                        for metric_name, value in metrics_computed.items():
                            results.append({
                                "case_id_number": case.case_id_number,
                                "event_type": "heat_wave",
                                "forecast_source": "HRES",
                                "target_source": "ERA5",
                                "metric": metric_name,
                                "forecast_variable": "surface_air_temperature",
                                "target_variable": "surface_air_temperature",
                                "init_time": init_t,
                                "valid_time": valid_time,
                                "lead_time": lt,
                                "value": value
                            })
                            
        # --- B. Evaluate Aurora ---
        logger.info("  Evaluating Aurora...")
        for nc_file in aurora_files:
            # Parse lead bucket for labeling
            match = re.search(r"_L(\d+)_", nc_file.name)
            lead_bucket = f"L{match.group(1)}" if match else "L?"
            model_name = f"aurora1.5_{lead_bucket}"
            
            # Load the single Aurora forecast file
            aurora_fcst = xr.open_dataset(nc_file)
            if "2t_aurora" in aurora_fcst.data_vars:
                aurora_var = "2t_aurora"
            else:
                aurora_var = [v for v in aurora_fcst.data_vars if v != "latitude" and v != "longitude"][0]
            
            # For each lead_time step in this forecast
            init_t = aurora_fcst.init_time.values[0]
            for lt in aurora_fcst.lead_time.values:
                valid_time = init_t + lt
                
                # Only evaluate if valid_time is inside the case window
                if pd.to_datetime(case.start_date) <= pd.to_datetime(valid_time) <= pd.to_datetime(case.end_date):
                    fcst_2d = aurora_fcst[aurora_var].sel(init_time=init_t, lead_time=lt)
                    
                    try:
                        tgt_2d = era5_ds["surface_air_temperature"].sel(time=valid_time).load()
                    except KeyError:
                        continue
                    
                    fcst_arr, tgt_arr = align_and_subset_2d(fcst_2d, tgt_2d, case.location)
                    
                    # Compute metrics
                    metrics_computed = {
                        "Peak_Amplitude_Error": peak_amplitude_error(fcst_arr, tgt_arr),
                        "Conditional_Bias_Extremes": conditional_bias_extremes(fcst_arr, tgt_arr),
                        "RootMeanSquaredError": rmse(fcst_arr, tgt_arr),
                        "Spatial_IOU": intersection_over_union(fcst_arr, tgt_arr),
                        "Extreme_Area_Ratio": extreme_area_ratio(fcst_arr, tgt_arr),
                    }
                    
                    for metric_name, value in metrics_computed.items():
                        results.append({
                            "case_id_number": case.case_id_number,
                            "event_type": "heat_wave",
                            "forecast_source": model_name,
                            "target_source": "ERA5",
                            "metric": metric_name,
                            "forecast_variable": "surface_air_temperature",
                            "target_variable": "surface_air_temperature",
                            "init_time": init_t,
                            "valid_time": valid_time,
                            "lead_time": lt,
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
