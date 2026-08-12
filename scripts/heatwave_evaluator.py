"""
Heatwave Evaluation Pipeline using ExtremeWeatherBench (EWB).

Evaluates Aurora 1.5 and HRES forecasts against ERA5 reanalysis for heatwave
cases using 5 metrics:
  - 3 Amplitude: Peak Amplitude Error (PAE), Conditional Bias of Extremes (CBE), RMSE
  - 2 Spatial:   Intersection over Union (IoU/CSI),
                 Extreme Area Ratio (EAR)

Usage:
    python scripts/heatwave_evaluator.py
"""

import os
# Suppress GCS experimental warnings and force anonymous access
os.environ["GCSFS_EXPERIMENTAL_ZB_HNS_SUPPORT"] = "false"
os.environ["GCSFS_TOKEN"] = "anon"

import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

# Suppress harmless divide-by-zero warnings from dask/xarray
# that occur when xr.where() evaluates both branches before masking.
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

import aiohttp
import numpy as np
import pandas as pd
import xarray as xr

import extremeweatherbench as ewb

# Add project root to path for local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# =============================================================================
# Section 2: Custom Metric Classes
# =============================================================================
# Built-in metrics reused directly:
#   - ewb.metrics.RootMeanSquaredError  (RMSE)
#
# Built-in CSI exists but requires static thresholds. We need dynamic
# percentile thresholds computed from the target at evaluation time, so we
# subclass ThresholdMetric and compute the percentile inside _compute_metric.
# =============================================================================


class PeakAmplitudeError(ewb.BaseMetric):
    """Peak Amplitude Error: max(forecast) - max(target) over the spatial domain.

    Captures the well-documented amplitude bias in AI weather models where they
    underestimate the peak intensity of extreme events (Pasche et al., 2025).

    A negative PAE means the model underestimates the peak temperature.
    A positive PAE means the model overestimates it.

    This is distinct from the built-in MaximumMeanAbsoluteError, which computes
    the MAE of spatially-averaged maxima within a time-tolerance window and
    returns an unsigned error. PAE is a signed, direct comparison of spatial
    peak values per lead-time step.
    """

    def __init__(self, name: str = "Peak_Amplitude_Error", **kwargs):
        super().__init__(name=name, **kwargs)

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> xr.DataArray:
        """Compute max(forecast) - max(target) over lat/lon, preserving lead_time."""
        spatial_dims = [d for d in forecast.dims if d in ("latitude", "longitude")]

        # Use nanmax to handle any NaN padding at boundaries
        forecast_peak = forecast.max(dim=spatial_dims, skipna=True)
        target_peak = target.max(dim=spatial_dims, skipna=True)

        result = forecast_peak - target_peak

        # Reduce any remaining dims except preserve_dims
        reduce_dims = [d for d in result.dims if d != self.preserve_dims]
        if reduce_dims:
            result = result.mean(dim=reduce_dims)
        return result


class ConditionalBiasOfExtremes(ewb.BaseMetric):
    """Conditional Bias of Extremes: mean(f - o) where o >= Q_percentile.

    Unlike the built-in MeanError (which averages over ALL grid points), CBE
    computes bias ONLY at grid points where the target exceeds a high
    percentile. This avoids dilution by non-extreme areas and directly
    quantifies systematic underestimation where the extreme is actually
    occurring (Taggart et al., 2022).

    The percentile threshold is computed dynamically from the target at each
    evaluation step, ensuring the metric adapts to each case's actual
    temperature distribution.
    """

    def __init__(
        self, percentile: float = 90.0, name: str = "Conditional_Bias_Extremes", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.percentile = percentile

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> xr.DataArray:
        """Compute mean bias at grid points where target >= Qxx percentile."""
        # Guard against all-NaN slices (e.g., lead times with no forecast data)
        if target.isnull().all():
            return xr.DataArray(np.nan)

        threshold = float(target.quantile(self.percentile / 100.0, skipna=True))
        if np.isnan(threshold):
            return xr.DataArray(np.nan)

        # Optimization: Subtract first, then apply .where() once to save memory and time
        bias = forecast - target
        bias = bias.where(target >= threshold)

        # Average over all dims except preserve_dims
        reduce_dims = [d for d in bias.dims if d != self.preserve_dims]
        return bias.mean(dim=reduce_dims, skipna=True)


class IntersectionOverUnion(ewb.ThresholdMetric):
    """IoU / CSI with a dynamic percentile threshold from the target.

    The built-in ewb.CriticalSuccessIndex requires static forecast_threshold
    and target_threshold values at instantiation time. For heatwave evaluation,
    we need to compute the threshold dynamically from the target's temperature
    distribution at each evaluation step (e.g., 90th percentile).

    This class subclasses ThresholdMetric and leverages its
    transformed_contingency_manager() to compute the contingency table after
    dynamically determining the threshold.

    IoU = TP / (TP + FP + FN), which is equivalent to CSI (Critical Success
    Index) — the standard binary verification metric (Wilks, 2011).
    """

    def __init__(
        self, percentile: float = 90.0, name: str = "Spatial_IOU", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.percentile = percentile

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> xr.DataArray:
        """Compute IoU with threshold set to target's Qxx percentile."""
        dynamic_threshold = float(target.quantile(self.percentile / 100.0, skipna=True))

        transformed = self.transformed_contingency_manager(
            forecast=forecast,
            target=target,
            forecast_threshold=dynamic_threshold,
            target_threshold=dynamic_threshold,
            preserve_dims=self.preserve_dims,
        )

        counts = transformed.get_counts()
        tp = counts["tp_count"]
        fp = counts["fp_count"]
        fn = counts["fn_count"]

        union = tp + fp + fn
        # Return NaN (not 0.0) when union==0, so we can distinguish "no spatial
        # overlap" from "no forecast data at this lead time".
        return xr.where(union > 0, tp / union, np.nan)


class ExtremeAreaRatio(ewb.BaseMetric):
    """Extreme Area Ratio: A_forecast / A_target.

    Counts grid points exceeding the dynamic percentile threshold in both
    forecast and target, then computes their ratio. Inspired by the "A"
    component of the SAL framework (Wernli et al., 2008).

    EAR = 1.0 → perfect area match
    EAR < 1.0 → model under-predicts the spatial extent of the extreme
    EAR > 1.0 → model over-predicts the spatial extent

    This metric is not available as a built-in in EWB. The closest built-in,
    SpatialDisplacement, measures centre-of-mass shift rather than area ratio.
    """

    def __init__(
        self, percentile: float = 90.0, name: str = "Extreme_Area_Ratio", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.percentile = percentile

    def _compute_metric(
        self, forecast: xr.DataArray, target: xr.DataArray, **kwargs: Any
    ) -> xr.DataArray:
        """Compute ratio of extreme-area grid point counts."""
        # Guard against all-NaN slices
        if target.isnull().all():
            return xr.DataArray(np.nan)

        threshold = float(target.quantile(self.percentile / 100.0, skipna=True))
        if np.isnan(threshold):
            return xr.DataArray(np.nan)

        spatial_dims = [d for d in forecast.dims if d in ("latitude", "longitude")]

        # Count grid points exceeding the threshold
        fcst_count = (forecast >= threshold).sum(dim=spatial_dims, skipna=True).astype(float)
        tgt_count = (target >= threshold).sum(dim=spatial_dims, skipna=True).astype(float)

        # Avoid division by zero
        result = xr.where(tgt_count > 0, fcst_count / tgt_count, np.nan)

        # Reduce any remaining dims except preserve_dims
        reduce_dims = [d for d in result.dims if d != self.preserve_dims]
        if reduce_dims:
            result = result.mean(dim=reduce_dims, skipna=True)
        return result


# =============================================================================
# Section 3: Forecast & Target Setup (Memory-Efficient)
# =============================================================================


def load_aurora_forecasts(forecast_dir: str = "ewb_forecasts") -> list:
    """Load Aurora 1.5 forecast NC files, one XarrayForecast per file.

    Each NC file represents a single forecast run with one init_time and
    one lead-time-to-onset bucket (L24, L72, or L120). Loading them
    individually avoids the NaN-padding bug caused by concatenating files
    with mismatched lead_time ranges via open_mfdataset.

    The lead-time bucket is parsed from the filename (e.g., "_L120_") and
    embedded in the forecast name so the output CSV distinguishes results
    by initialization offset ("aurora1.5_L24", "aurora1.5_L72", etc.).

    Args:
        forecast_dir: Path to the directory containing the .nc files.

    Returns:
        List of ewb.XarrayForecast objects, one per NC file.
    """
    forecast_path = Path(forecast_dir)
    nc_files = sorted(list(forecast_path.glob("*.nc")))

    if not nc_files:
        raise FileNotFoundError(
            f"No .nc files found in {forecast_path.resolve()}. "
            "Run the forecast generation script first."
        )

    logger.info(f"Found {len(nc_files)} forecast files in {forecast_path}")

    forecasts = []
    for nc_file in nc_files:
        # Parse lead-time-to-onset bucket from filename (e.g., "L24", "L72", "L120")
        match = re.search(r"_L(\d+)_", nc_file.name)
        lead_bucket = f"L{match.group(1)}" if match else "L?"

        # Load lazily with dask to keep memory low.
        ds = xr.open_dataset(nc_file, chunks={"lead_time": -1})

        forecast = ewb.XarrayForecast(
            ds=ds,
            name=f"aurora1.5_{lead_bucket}",
            variables=["surface_air_temperature"],
            variable_mapping={"2t_aurora": "surface_air_temperature"},
        )
        forecasts.append(forecast)

    logger.info(
        f"Created {len(forecasts)} individual Aurora forecasts "
        f"(each with its own init_time and self-consistent lead_time range)"
    )
    return forecasts


def setup_hres_baseline() -> ewb.ZarrForecast:
    """Configure HRES from the public WeatherBench2 zarr store.

    HRES data covers 2016-2022 with 12-hourly init times.

    Returns:
        An ewb.ZarrForecast for the HRES model.
    """
    return ewb.ZarrForecast(
        source="gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
        name="HRES",
        variable_mapping=ewb.HRES_metadata_variable_mapping,
        storage_options={"remote_options": {"anon": True}},
    )


def setup_era5_target() -> ewb.ERA5:
    """Configure ERA5 as the verification target.

    Uses the Google ARCO ERA5 dataset with anonymous access and a generous
    timeout to handle large data transfers.

    Returns:
        An ewb.ERA5 target object.
    """
    return ewb.ERA5(
        variables=["surface_air_temperature"],
        storage_options={
            "token": "anon",
            "client_kwargs": {
                "timeout": aiohttp.ClientTimeout(
                    total=3600, connect=60, sock_read=3600
                )
            },
        },
    )


# =============================================================================
# Section 4: Evaluation Pipeline
# =============================================================================

VAR_KWARGS = {
    "forecast_variable": "surface_air_temperature",
    "target_variable": "surface_air_temperature",
}


def build_metric_list() -> list:
    """Build the list of all 5 metrics.

    Metrics:
        1. PAE  — Peak Amplitude Error (custom, BaseMetric)
        2. CBE  — Conditional Bias of Extremes (custom, BaseMetric)
        3. RMSE — Root Mean Squared Error (built-in ewb.metrics.RootMeanSquaredError)
        4. IoU  — Intersection over Union / CSI (custom ThresholdMetric with dynamic Q90)
        5. EAR  — Extreme Area Ratio (custom, BaseMetric)

    Returns:
        List of metric instances.
    """
    metrics = [
        # --- Amplitude Metrics ---
        PeakAmplitudeError(**VAR_KWARGS),
        ConditionalBiasOfExtremes(percentile=90.0, **VAR_KWARGS),
        ewb.metrics.RootMeanSquaredError(**VAR_KWARGS),
        # --- Spatial Extent Metrics ---
        IntersectionOverUnion(percentile=90.0, **VAR_KWARGS),
        ExtremeAreaRatio(percentile=90.0, **VAR_KWARGS),
    ]
    return metrics


def build_evaluation_objects(
    aurora_forecasts: list,
    hres_forecast: ewb.ZarrForecast,
    era5_target: ewb.ERA5,
) -> list:
    """Create EvaluationObject list for Aurora 1.5 and HRES.

    One EvaluationObject is created per Aurora NC file (each with its own
    init_time and lead_time range) plus one for HRES. EWB matches each
    forecast against the cases whose time windows it covers.

    Args:
        aurora_forecasts: List of Aurora 1.5 XarrayForecast objects.
        hres_forecast: HRES ZarrForecast.
        era5_target: ERA5 verification target.

    Returns:
        List of EvaluationObject instances.
    """
    metric_list = build_metric_list()

    eval_objects = []

    # One EvaluationObject per Aurora forecast file
    for forecast in aurora_forecasts:
        eval_objects.append(
            ewb.EvaluationObject(
                event_type="heat_wave",
                metric_list=metric_list,
                target=era5_target,
                forecast=forecast,
            )
        )

    # One EvaluationObject for HRES (covers all cases via its cloud zarr store)
    eval_objects.append(
        ewb.EvaluationObject(
            event_type="heat_wave",
            metric_list=metric_list,
            target=era5_target,
            forecast=hres_forecast,
        ),
    )

    return eval_objects


def run_evaluation(output_csv: str = "heatwave_evaluations.csv"):
    """Main entry point: load data, run EWB evaluation, save CSV.

    Memory-efficiency notes:
        - Each Aurora NC file is loaded individually with dask chunks.
        - HRES and ERA5 are loaded lazily from cloud zarr stores.
        - EWB evaluates case-by-case internally, subsetting before computing.
        - Results are accumulated as lightweight DataFrames.

    Args:
        output_csv: Path for the output CSV file.
    """
    logger.info("=" * 60)
    logger.info("Starting Heatwave Evaluation Pipeline")
    logger.info("=" * 60)

    # --- Load forecasts ---
    logger.info("Loading Aurora 1.5 forecasts (one per NC file)...")
    aurora_forecasts = load_aurora_forecasts("ewb_forecasts")

    logger.info("Configuring HRES baseline...")
    hres_forecast = setup_hres_baseline()

    # --- Load target ---
    logger.info("Configuring ERA5 target...")
    era5_target = setup_era5_target()

    # --- Build evaluation objects ---
    n_aurora = len(aurora_forecasts)
    logger.info(f"Building evaluation objects: {n_aurora} Aurora files + 1 HRES, 5 metrics each...")
    eval_objects = build_evaluation_objects(aurora_forecasts, hres_forecast, era5_target)

    # --- Load EWB case metadata ---
    logger.info("Loading EWB case metadata...")
    cases = ewb.load_cases()
    n_hw = sum(1 for c in cases if c.event_type == "heat_wave")
    logger.info(f"Found {n_hw} heatwave cases in EWB case database")

    # --- Run evaluation ---
    logger.info("Running EWB evaluation (this may take a while)...")
    runner = ewb.evaluation(
        case_metadata=cases,
        evaluation_objects=eval_objects,
    )

    # Run serially (n_jobs=1) to minimize peak memory usage.
    # For faster execution on a machine with enough RAM, increase n_jobs.
    outputs = runner.run_evaluation(n_jobs=1)

    # --- Save results ---
    output_path = Path(output_csv)
    outputs.to_csv(output_path, index=False)
    logger.info(f"Evaluations saved to {output_path.resolve()}")
    logger.info(f"Result shape: {outputs.shape}")

    # Print summary statistics
    if not outputs.empty:
        logger.info("\n--- Summary by Metric and Forecast Source ---")
        summary = outputs.groupby(["metric", "forecast_source"])["value"].agg(
            ["mean", "std", "count"]
        )
        logger.info(f"\n{summary.to_string()}")

    return outputs


# =============================================================================
# Section 5: Main
# =============================================================================

if __name__ == "__main__":
    run_evaluation()
