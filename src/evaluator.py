"""
AuroraEvaluator — reusable forecast verification toolkit.

Provides a class-based API over the pure metric functions in ``utils.py``,
handling the common evaluate-across-lead-times pattern found in multiple
evaluation scripts.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

from .utils import (
    rmse,
    peak_amplitude_error,
    conditional_bias_extremes,
    intersection_over_union,
    extreme_area_ratio,
    align_and_subset_2d,
)


class AuroraEvaluator:
    """
    Reusable toolkit for computing forecast verification metrics.

    Wraps the pure metric functions from ``utils.py`` into a convenient
    class that handles the common evaluate-across-lead-times pattern
    found in multiple evaluation scripts.

    Example::

        evaluator = AuroraEvaluator(metrics=["rmse", "peak_amplitude_error"])
        results = evaluator.evaluate_forecast_vs_target(
            forecast_ds, target_ds,
            forecast_var="2t_aurora", target_var="t2m",
            lead_times_h=[6, 24, 72, 120],
            init_time=datetime(2021, 6, 15, 12),
        )
        df = AuroraEvaluator.results_to_dataframe(results)
    """

    METRIC_REGISTRY: dict[str, Callable] = {
        "rmse": rmse,
        "peak_amplitude_error": peak_amplitude_error,
        "conditional_bias_extremes": conditional_bias_extremes,
        "intersection_over_union": intersection_over_union,
        "extreme_area_ratio": extreme_area_ratio,
    }

    def __init__(self, metrics: list[str] | None = None):
        """
        Args:
            metrics: Which metrics to compute. Defaults to ``["rmse"]``.
                     Must be keys from :attr:`METRIC_REGISTRY`.
        """
        if metrics is None:
            metrics = ["rmse"]
        unknown = set(metrics) - set(self.METRIC_REGISTRY)
        if unknown:
            raise ValueError(
                f"Unknown metrics: {unknown}. "
                f"Available: {list(self.METRIC_REGISTRY.keys())}"
            )
        self.metrics = {name: self.METRIC_REGISTRY[name] for name in metrics}

    def evaluate_2d(
        self, forecast_2d: np.ndarray, target_2d: np.ndarray
    ) -> dict[str, float]:
        """Compute all configured metrics on a single 2D field pair."""
        return {name: fn(forecast_2d, target_2d) for name, fn in self.metrics.items()}

    def evaluate_forecast_vs_target(
        self,
        forecast_ds: xr.Dataset,
        target_ds: xr.Dataset,
        forecast_var: str,
        target_var: str,
        lead_times_h: list[int],
        init_time,
        model_name: str = "aurora",
        eval_bbox: dict | None = None,
    ) -> list[dict]:
        """
        Full evaluation loop over lead times.

        For each lead time: select forecast/target, optionally crop to
        *eval_bbox*, align grids, compute all configured metrics, and
        collect results into a list of row-dicts.

        Args:
            forecast_ds: xarray Dataset with forecast data indexed by lead_time.
            target_ds: xarray Dataset with ERA5 ground truth indexed by time.
            forecast_var: Variable name in *forecast_ds* (e.g., ``"2t_aurora"``).
            target_var: Variable name in *target_ds* (e.g., ``"t2m"``).
            lead_times_h: List of lead times in hours to evaluate.
            init_time: Forecast initialization time (datetime-like).
            model_name: Label for the model in results.
            eval_bbox: Optional bounding box to crop the forecast before
                       comparison (keys: ``lat_min``, ``lat_max``,
                       ``lon_min``, ``lon_max``).

        Returns:
            List of dicts, one per (lead_time, metric) combination.
        """
        results: list[dict] = []
        init_dt = pd.to_datetime(init_time)

        for lt_h in lead_times_h:
            valid_dt = init_dt + timedelta(hours=lt_h)
            valid_str = valid_dt.strftime("%Y-%m-%dT%H:00:00")

            # Select target
            try:
                tgt_2d = target_ds[target_var].sel(time=valid_str)
            except KeyError:
                continue

            # Select forecast
            try:
                fcst_2d = forecast_ds[forecast_var].sel(
                    lead_time=pd.to_timedelta(f"{lt_h}h")
                ).squeeze()
            except KeyError:
                continue

            # Crop to eval bbox if provided
            if eval_bbox is not None:
                fcst_2d = fcst_2d.sel(
                    latitude=slice(eval_bbox["lat_max"], eval_bbox["lat_min"]),
                    longitude=slice(eval_bbox["lon_min"], eval_bbox["lon_max"]),
                )

            # Align and compute
            fcst_arr, tgt_arr = align_and_subset_2d(fcst_2d, tgt_2d)
            metric_values = self.evaluate_2d(fcst_arr, tgt_arr)

            for metric_name, value in metric_values.items():
                results.append({
                    "init_time": init_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "valid_time": valid_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "lead_time_hours": lt_h,
                    "metric": metric_name,
                    "model": model_name,
                    "value": value,
                })

        return results

    @staticmethod
    def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
        """Convert result rows to a DataFrame with standard column ordering."""
        df = pd.DataFrame(results)
        if df.empty:
            return df

        # Standard column order used across evaluation scripts
        preferred_order = [
            "value", "lead_time_hours", "init_time", "valid_time",
            "metric", "model",
        ]
        # Keep any extra columns that aren't in the preferred order
        extra = [c for c in df.columns if c not in preferred_order]
        ordered = [c for c in preferred_order if c in df.columns] + extra
        return df[ordered]
