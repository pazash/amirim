"""
ERA5 Precipitation vs. Total Column Water Vapor (TCWV) Correlation & Spatial Matching Evaluator.

Analyzes whether high-precipitation areas match high-TCWV areas across different times
in the 10 extreme precipitation events from extreme_precip_evaluator.py using ERA5 reanalysis only.

Metrics evaluated at 6-hourly timesteps (00, 06, 12, 18 UTC) and event-aggregated:
  1. Spearman_Rank_Correlation: Non-parametric rank correlation capturing monotonic /
     exponential coupling (Bretherton et al. 2004, Peters & Neelin 2006).
  2. Spatial_IOU_90 / Spatial_IOU_95: Exact grid-point Jaccard overlap between
     >=Q90 (or >=Q95) precipitation cores and >=Q90 (or >=Q95) TCWV plumes.
  3. FSS_90_w3 / FSS_95_w5: Fractions Skill Score (Roberts & Lean 2008) over 3x3 (0.75 deg)
     and 5x5 (1.25 deg) neighborhoods to account for mesoscale spatial displacement
     between moisture convergence plumes and convective rainfall cores.
  4. Precip_Q95_in_TCWV_Q80 / Precip_Q95_in_TCWV_Q90: Conditional Quantile Capture
     (Containment Rate) — fraction of extreme rain grid cells (>=Q95) that reside inside
     the moist TCWV envelope (>=Q80 or >=Q90).
  5. Mean_TCWV_Percentile_at_Precip_Q95: Average domain-relative TCWV percentile [0-100]
     at grid points experiencing extreme (>=Q95) rainfall.
  6. ROC_AUC_Precip_Q95: Area Under the ROC Curve measuring how well TCWV discriminates
     extreme (>=Q95) precipitation locations.
"""

import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import ndimage, stats

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from extreme_precip_evaluator import PRECIP_CASES
from src.utils import (
    TCWV_LEVELS,
    compute_tcwv_from_dataset,
    load_era5_zarr,
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


# =============================================================================
# Section 1: Spatial & Statistical Correlation Metrics
# =============================================================================


def _valid_pair(precip_2d: np.ndarray, tcwv_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened 1D arrays of valid (non-NaN) paired grid cells."""
    valid = ~np.isnan(precip_2d) & ~np.isnan(tcwv_2d)
    return precip_2d[valid], tcwv_2d[valid]


def spearman_spatial_corr(precip_2d: np.ndarray, tcwv_2d: np.ndarray) -> float:
    """Spearman rank correlation across valid spatial grid cells."""
    p, w = _valid_pair(precip_2d, tcwv_2d)
    if len(p) < 5 or np.nanstd(p) == 0 or np.nanstd(w) == 0:
        return np.nan
    res = stats.spearmanr(p, w)
    return float(res.statistic)


def pearson_spatial_corr(precip_2d: np.ndarray, tcwv_2d: np.ndarray) -> float:
    """Pearson linear correlation across valid spatial grid cells."""
    p, w = _valid_pair(precip_2d, tcwv_2d)
    if len(p) < 5 or np.nanstd(p) == 0 or np.nanstd(w) == 0:
        return np.nan
    res = stats.pearsonr(p, w)
    return float(res.statistic)


def spatial_percentile_iou(
    precip_2d: np.ndarray,
    tcwv_2d: np.ndarray,
    percentile: float = 90.0,
    min_precip_mm: float = 0.1,
) -> float:
    """
    Intersection over Union (IoU) between areas where precipitation >= Q_percentile
    and areas where TCWV >= Q_percentile.
    """
    valid = ~np.isnan(precip_2d) & ~np.isnan(tcwv_2d)
    if not np.any(valid):
        return np.nan
    p_thresh = max(float(np.nanpercentile(precip_2d[valid], percentile)), min_precip_mm)
    w_thresh = float(np.nanpercentile(tcwv_2d[valid], percentile))

    p_mask = (precip_2d >= p_thresh) & valid
    w_mask = (tcwv_2d >= w_thresh) & valid

    intersection = (p_mask & w_mask).sum()
    union = (p_mask | w_mask).sum()
    return float(intersection / union) if union > 0 else np.nan


def fractions_skill_score(
    precip_2d: np.ndarray,
    tcwv_2d: np.ndarray,
    percentile: float = 90.0,
    window_size: int = 3,
    min_precip_mm: float = 0.1,
) -> float:
    """
    Fractions Skill Score (FSS; Roberts & Lean 2008) between binary masks of
    precipitation >= Q_percentile and TCWV >= Q_percentile over a neighborhood
    window of (window_size x window_size) grid points.
    """
    valid = ~np.isnan(precip_2d) & ~np.isnan(tcwv_2d)
    if not np.any(valid):
        return np.nan

    p_thresh = max(float(np.nanpercentile(precip_2d[valid], percentile)), min_precip_mm)
    w_thresh = float(np.nanpercentile(tcwv_2d[valid], percentile))

    p_bin = np.where(valid, (precip_2d >= p_thresh).astype(np.float64), 0.0)
    w_bin = np.where(valid, (tcwv_2d >= w_thresh).astype(np.float64), 0.0)

    if p_bin.sum() == 0 and w_bin.sum() == 0:
        return np.nan

    f_p = ndimage.uniform_filter(p_bin, size=window_size, mode="nearest")
    f_w = ndimage.uniform_filter(tcwv_bin := w_bin, size=window_size, mode="nearest")

    mse = np.mean((f_p[valid] - f_w[valid]) ** 2)
    ref = np.mean(f_p[valid] ** 2 + f_w[valid] ** 2)
    if ref == 0:
        return np.nan
    return float(1.0 - (mse / ref))


def conditional_quantile_capture(
    precip_2d: np.ndarray,
    tcwv_2d: np.ndarray,
    precip_percentile: float = 95.0,
    tcwv_percentile: float = 80.0,
    min_precip_mm: float = 0.1,
) -> float:
    """
    Containment rate: fraction of extreme precipitation grid points (>= precip_percentile)
    that fall inside the high-TCWV plume (>= tcwv_percentile):
        P(TCWV >= Q_tcwv | Precip >= Q_precip)
    """
    valid = ~np.isnan(precip_2d) & ~np.isnan(tcwv_2d)
    if not np.any(valid):
        return np.nan

    p_thresh = max(
        float(np.nanpercentile(precip_2d[valid], precip_percentile)), min_precip_mm
    )
    w_thresh = float(np.nanpercentile(tcwv_2d[valid], tcwv_percentile))

    p_mask = (precip_2d >= p_thresh) & valid
    n_extreme_p = p_mask.sum()
    if n_extreme_p == 0:
        return np.nan

    w_mask = (tcwv_2d >= w_thresh) & valid
    return float((p_mask & w_mask).sum() / n_extreme_p)


def mean_tcwv_percentile_at_extreme_precip(
    precip_2d: np.ndarray,
    tcwv_2d: np.ndarray,
    precip_percentile: float = 95.0,
    min_precip_mm: float = 0.1,
) -> float:
    """
    Mean domain-relative percentile rank [0-100] of TCWV at grid points where
    precipitation >= precip_percentile.
    """
    p, w = _valid_pair(precip_2d, tcwv_2d)
    if len(p) < 5:
        return np.nan

    p_thresh = max(float(np.nanpercentile(p, precip_percentile)), min_precip_mm)
    extreme_mask = p >= p_thresh
    if not np.any(extreme_mask):
        return np.nan

    w_pct_ranks = stats.rankdata(w, method="average") / len(w) * 100.0
    return float(np.mean(w_pct_ranks[extreme_mask]))


def roc_auc_extreme_precip(
    precip_2d: np.ndarray,
    tcwv_2d: np.ndarray,
    precip_percentile: float = 95.0,
    min_precip_mm: float = 0.1,
) -> float:
    """
    ROC-AUC (equivalent to normalized Mann-Whitney U statistic) measuring the
    ability of TCWV to discriminate extreme precipitation (>= precip_percentile)
    grid cells from non-extreme grid cells.
    """
    p, w = _valid_pair(precip_2d, tcwv_2d)
    if len(p) < 5:
        return np.nan

    p_thresh = max(float(np.nanpercentile(p, precip_percentile)), min_precip_mm)
    pos = w[p >= p_thresh]
    neg = w[p < p_thresh]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    u_stat, _ = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    return float(u_stat / (len(pos) * len(neg)))


def compute_all_correlation_metrics(
    precip_2d: np.ndarray, tcwv_2d: np.ndarray
) -> dict[str, float]:
    """Compute the full suite of correlation and spatial matching metrics."""
    return {
        "Spearman_Rank_Correlation": spearman_spatial_corr(precip_2d, tcwv_2d),
        "Pearson_Linear_Correlation": pearson_spatial_corr(precip_2d, tcwv_2d),
        "Spatial_IOU_90": spatial_percentile_iou(precip_2d, tcwv_2d, 90.0),
        "Spatial_IOU_95": spatial_percentile_iou(precip_2d, tcwv_2d, 95.0),
        "FSS_90_w3": fractions_skill_score(precip_2d, tcwv_2d, 90.0, window_size=3),
        "FSS_90_w5": fractions_skill_score(precip_2d, tcwv_2d, 90.0, window_size=5),
        "FSS_95_w5": fractions_skill_score(precip_2d, tcwv_2d, 95.0, window_size=5),
        "Precip_Q95_in_TCWV_Q80": conditional_quantile_capture(
            precip_2d, tcwv_2d, precip_percentile=95.0, tcwv_percentile=80.0
        ),
        "Precip_Q95_in_TCWV_Q90": conditional_quantile_capture(
            precip_2d, tcwv_2d, precip_percentile=95.0, tcwv_percentile=90.0
        ),
        "Mean_TCWV_Percentile_at_Precip_Q95": mean_tcwv_percentile_at_extreme_precip(
            precip_2d, tcwv_2d, precip_percentile=95.0
        ),
        "ROC_AUC_Precip_Q95": roc_auc_extreme_precip(
            precip_2d, tcwv_2d, precip_percentile=95.0
        ),
    }


# =============================================================================
# Section 2: Main ERA5 Evaluation Pipeline
# =============================================================================


def evaluate_precip_tcwv_correlation(
    output_csv: str = "precip_tcwv_correlation.csv",
    plots_dir: str = "plots/precip_tcwv_correlation",
) -> pd.DataFrame:
    """
    Download ERA5 precipitation and TCWV for all 10 PRECIP_CASES, evaluate spatial
    correlation & overlap across 6-hourly timesteps (plus event aggregates), save CSV,
    and generate diagnostic plots.
    """
    logger.info("=" * 65)
    logger.info("Starting ERA5 Precipitation vs. TCWV Correlation Pipeline")
    logger.info("=" * 65)

    era5_ds = load_era5_zarr(variable="precip_and_tcwv")

    results = []
    binned_records = []

    for case_idx, case in enumerate(PRECIP_CASES):
        logger.info(
            f"\nProcessing Case {case.case_id_number}: {case.title} ({case_idx+1}/{len(PRECIP_CASES)})"
        )
        logger.info(f"  Period: {case.start_date} to {case.end_date}")

        # Build 6-hourly timestamps (00, 06, 12, 18 UTC) within the event window
        times_6h = pd.date_range(case.start_date, case.end_date, freq="6h")

        # 1. Download ERA5 Precipitation (convert m -> mm/h)
        logger.info("  Downloading ERA5 precipitation at 6-hourly steps...")
        precip_case_ds = era5_ds[["total_precipitation"]].sel(time=times_6h)
        precip_case_ds = case.location.mask(precip_case_ds).compute()
        precip_case_ds = standardize_longitude(precip_case_ds)
        precip_da = precip_case_ds["total_precipitation"] * 1000.0  # mm/h

        # 2. Download ERA5 3D specific_humidity and compute vertically integrated TCWV (kg/m^2)
        logger.info("  Downloading ERA5 specific_humidity & computing TCWV...")
        tcwv_case_raw = era5_ds[["specific_humidity"]].sel(time=times_6h)
        tcwv_case_raw = case.location.mask(tcwv_case_raw).compute()
        tcwv_case_raw = standardize_longitude(tcwv_case_raw)
        tcwv_da = compute_tcwv_from_dataset(
            tcwv_case_raw, q_var="specific_humidity", levels=TCWV_LEVELS
        ).compute()
        del tcwv_case_raw

        event_start_dt = pd.to_datetime(case.start_date)

        # 3. Evaluate at each 6-hourly timestep
        for t_val in times_6h:
            try:
                p_2d = precip_da.sel(time=t_val).values
                w_2d = tcwv_da.sel(time=t_val).values
            except KeyError:
                continue

            hours_since_start = int(
                (pd.to_datetime(t_val) - event_start_dt).total_seconds() / 3600
            )
            metrics = compute_all_correlation_metrics(p_2d, w_2d)

            for m_name, val in metrics.items():
                results.append(
                    {
                        "case_id_number": case.case_id_number,
                        "case_title": case.title,
                        "event_type": case.event_type,
                        "valid_time": str(t_val),
                        "hours_from_start": hours_since_start,
                        "evaluation_mode": "6h_instantaneous",
                        "metric": m_name,
                        "value": val,
                    }
                )

            # Collect decile-binned precipitation vs TCWV for exponential curve plotting
            p_flat, w_flat = _valid_pair(p_2d, w_2d)
            if len(p_flat) >= 10:
                w_pct = stats.rankdata(w_flat, method="average") / len(w_flat) * 100.0
                bins = np.linspace(0, 100, 11)
                bin_indices = np.digitize(w_pct, bins) - 1
                bin_indices = np.clip(bin_indices, 0, 9)
                for b_idx in range(10):
                    in_bin = bin_indices == b_idx
                    if np.any(in_bin):
                        binned_records.append(
                            {
                                "case_id_number": case.case_id_number,
                                "case_title": case.title,
                                "tcwv_decile_center": (bins[b_idx] + bins[b_idx + 1]) / 2.0,
                                "mean_tcwv_kgm2": float(np.mean(w_flat[in_bin])),
                                "mean_precip_mmh": float(np.mean(p_flat[in_bin])),
                                "p95_precip_mmh": float(np.percentile(p_flat[in_bin], 95)),
                            }
                        )

        # 4. Evaluate Event-Aggregated Spatial Maps (Event Mean Precip vs Event Mean TCWV)
        p_event_mean = precip_da.mean(dim="time").values
        w_event_mean = tcwv_da.mean(dim="time").values
        agg_metrics = compute_all_correlation_metrics(p_event_mean, w_event_mean)

        for m_name, val in agg_metrics.items():
            results.append(
                {
                    "case_id_number": case.case_id_number,
                    "case_title": case.title,
                    "event_type": case.event_type,
                    "valid_time": "EVENT_MEAN",
                    "hours_from_start": -1,
                    "evaluation_mode": "event_aggregated",
                    "metric": m_name,
                    "value": val,
                }
            )

    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(output_csv, index=False)
        logger.info(f"\nSaved {len(df)} rows to {Path(output_csv).resolve()}")

        # Print Summary Table
        logger.info("\n--- Summary of 6-Hourly Instantaneous Correlation Metrics ---")
        inst_df = df[df["evaluation_mode"] == "6h_instantaneous"]
        summary_inst = (
            inst_df.dropna(subset=["value"])
            .groupby("metric")["value"]
            .agg(["count", "mean", "median", "std"])
        )
        logger.info(f"\n{summary_inst.to_string()}")

        logger.info("\n--- Summary of Event-Aggregated Correlation Metrics ---")
        agg_df = df[df["evaluation_mode"] == "event_aggregated"]
        summary_agg = (
            agg_df.dropna(subset=["value"])
            .groupby("metric")["value"]
            .agg(["count", "mean", "median", "std"])
        )
        logger.info(f"\n{summary_agg.to_string()}")

        # Generate Plots
        binned_df = pd.DataFrame(binned_records)
        generate_correlation_plots(df, binned_df, Path(plots_dir))

    return df


# =============================================================================
# Section 3: Diagnostic Plotting
# =============================================================================


def generate_correlation_plots(
    df: pd.DataFrame, binned_df: pd.DataFrame, output_dir: Path
):
    """Generate diagnostic plots showing Precip-TCWV coupling across times and events."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    inst_df = df[df["evaluation_mode"] == "6h_instantaneous"].copy()

    # Plot 1: Time evolution of key metrics across events (normalized or hours_from_start)
    key_ts_metrics = [
        "Spearman_Rank_Correlation",
        "Pearson_Linear_Correlation",
        "ROC_AUC_Precip_Q95",
        "Precip_Q95_in_TCWV_Q80",
        "FSS_90_w5",
        "Spatial_IOU_90",
    ]
    ts_sub = inst_df[inst_df["metric"].isin(key_ts_metrics)]
    if not ts_sub.empty:
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=ts_sub,
            x="hours_from_start",
            y="value",
            hue="metric",
            marker="o",
            err_style="band",
        )
        plt.title(
            "Evolution of Precipitation vs. TCWV Spatial Matching Across Event Duration"
        )
        plt.xlabel("Hours From Event Start")
        plt.ylabel("Metric Value")
        plt.ylim(-0.1, 1.05)
        plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_dir / "timeseries_precip_tcwv_metrics.png", dpi=300)
        plt.close()

    # Plot 2: Per-event comparison bar chart (6h Instantaneous vs Event-Aggregated)
    compare_metrics = [
        "Spearman_Rank_Correlation",
        "Pearson_Linear_Correlation",
        "ROC_AUC_Precip_Q95",
        "Precip_Q95_in_TCWV_Q80",
        "FSS_90_w5",
        "Spatial_IOU_90",
    ]
    comp_sub = df[df["metric"].isin(compare_metrics)]
    if not comp_sub.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=comp_sub,
            x="metric",
            y="value",
            hue="evaluation_mode",
            palette="Set2",
        )
        plt.title("ERA5 Precipitation vs. TCWV Matching: 6-Hourly vs. Event-Aggregated")
        plt.xlabel("Metric")
        plt.ylabel("Score")
        plt.xticks(rotation=20, ha="right")
        plt.ylim(0, 1.05)
        plt.legend(title="Evaluation Mode")
        plt.tight_layout()
        plt.savefig(output_dir / "instantaneous_vs_aggregated_metrics.png", dpi=300)
        plt.close()

    # Plot 3: Heatmap of metrics by Case
    pivot_case = (
        inst_df.groupby(["case_title", "metric"])["value"]
        .mean()
        .unstack("metric")
    )
    if not pivot_case.empty:
        plt.figure(figsize=(14, 7))
        sns.heatmap(
            pivot_case,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar_kws={"label": "Mean 6-Hourly Score"},
        )
        plt.title("Mean 6-Hourly Precipitation–TCWV Correlation Metrics by Extreme Event")
        plt.ylabel("Event")
        plt.xlabel("Metric")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "case_metrics_heatmap.png", dpi=300)
        plt.close()

    # Plot 4: Non-linear / Exponential Pickup of Precipitation with TCWV Decile
    if not binned_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        sns.lineplot(
            ax=axes[0],
            data=binned_df,
            x="tcwv_decile_center",
            y="mean_precip_mmh",
            hue="case_title",
            marker="o",
            alpha=0.75,
        )
        axes[0].set_title("Mean Precipitation Rate vs. Local TCWV Percentile")
        axes[0].set_xlabel("Domain TCWV Percentile (%)")
        axes[0].set_ylabel("Mean Precipitation (mm/h)")
        axes[0].legend_.remove()

        sns.lineplot(
            ax=axes[1],
            data=binned_df,
            x="tcwv_decile_center",
            y="p95_precip_mmh",
            hue="case_title",
            marker="s",
            alpha=0.75,
        )
        axes[1].set_title("95th Percentile Precipitation Rate vs. Local TCWV Percentile")
        axes[1].set_xlabel("Domain TCWV Percentile (%)")
        axes[1].set_ylabel("95th Percentile Precipitation (mm/h)")
        axes[1].legend(
            title="Extreme Event", bbox_to_anchor=(1.02, 1), loc="upper left"
        )

        plt.suptitle(
            "Exponential Pickup of Extreme Rainfall in High-TCWV Regimes (ERA5)",
            fontsize=14,
        )
        plt.tight_layout()
        plt.savefig(output_dir / "exponential_precip_vs_tcwv_pickup.png", dpi=300)
        plt.close()

    logger.info(f"Diagnostic plots saved to {output_dir.resolve()}")


if __name__ == "__main__":
    evaluate_precip_tcwv_correlation()

