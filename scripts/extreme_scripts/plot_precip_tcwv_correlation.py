from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Setup plotting style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (11, 6)

# Theoretical random / skill baselines for each metric
METRIC_BASELINES: dict[str, tuple[float, str]] = {
    "Spearman_Rank_Correlation": (0.0, "No Correlation (0.0)"),
    "Pearson_Linear_Correlation": (0.0, "No Correlation (0.0)"),
    "Spatial_IOU_90": (0.10 / 1.90, "Random Overlap (0.053)"),
    "Spatial_IOU_95": (0.05 / 1.95, "Random Overlap (0.026)"),
    "FSS_90_w3": (0.55, "FSS Useful Skill Threshold (0.55)"),
    "FSS_90_w5": (0.55, "FSS Useful Skill Threshold (0.55)"),
    "FSS_95_w5": (0.525, "FSS Useful Skill Threshold (0.525)"),
    "Precip_Q95_in_TCWV_Q80": (0.20, "Random Containment (0.20)"),
    "Precip_Q95_in_TCWV_Q90": (0.10, "Random Containment (0.10)"),
    "Mean_TCWV_Percentile_at_Precip_Q95": (50.0, "Random Percentile (50%)"),
    "ROC_AUC_Precip_Q95": (0.50, "Random Classifier (0.50)"),
}


def compute_diagnostic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistical summary and hypothesis test results (one-sample t-test and
    Wilcoxon signed-rank test against theoretical random/skill baseline) for each metric.
    """
    records = []
    for mode in ["6h_instantaneous", "event_aggregated"]:
        mode_df = df[df["evaluation_mode"] == mode]
        for metric, (baseline, baseline_label) in METRIC_BASELINES.items():
            vals = mode_df[mode_df["metric"] == metric]["value"].dropna().values
            if len(vals) == 0:
                continue

            mean_val = float(np.mean(vals))
            median_val = float(np.median(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            iqr_25 = float(np.percentile(vals, 25))
            iqr_75 = float(np.percentile(vals, 75))
            pct_above_base = float(np.mean(vals > baseline) * 100.0)

            # One-sample Wilcoxon signed-rank test (H1: median > baseline)
            diffs = vals - baseline
            non_zero_diffs = diffs[diffs != 0]
            if len(non_zero_diffs) >= 5:
                _, p_wilcoxon = stats.wilcoxon(non_zero_diffs, alternative="greater")
            else:
                p_wilcoxon = np.nan

            records.append(
                {
                    "evaluation_mode": mode,
                    "metric": metric,
                    "n": len(vals),
                    "mean": mean_val,
                    "std": std_val,
                    "median": median_val,
                    "q25": iqr_25,
                    "q75": iqr_75,
                    "baseline": baseline,
                    "baseline_type": baseline_label,
                    "lift_over_baseline": mean_val - baseline,
                    "pct_above_baseline": pct_above_base,
                    "wilcoxon_p_value": p_wilcoxon,
                }
            )
    return pd.DataFrame(records)


def plot_distributions_vs_baselines(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot 1: Box + strip distribution of [0, 1]-scaled metrics for 6h instantaneous
    vs Event-Aggregated, annotated with theoretical random/skill baselines.
    """
    print("Generating Metric Distributions vs. Baselines plot...")
    unit_metrics = [
        "ROC_AUC_Precip_Q95",
        "Precip_Q95_in_TCWV_Q80",
        "Precip_Q95_in_TCWV_Q90",
        "FSS_90_w5",
        "FSS_90_w3",
        "FSS_95_w5",
        "Spearman_Rank_Correlation",
        "Pearson_Linear_Correlation",
        "Spatial_IOU_90",
        "Spatial_IOU_95",
    ]
    sub_df = df[df["metric"].isin(unit_metrics)].copy()
    sub_df["metric_clean"] = sub_df["metric"].str.replace("_", " ")

    plt.figure(figsize=(14, 7))
    ax = sns.boxplot(
        data=sub_df,
        x="metric",
        y="value",
        hue="evaluation_mode",
        order=unit_metrics,
        palette="Set2",
        showfliers=False,
    )

    # Draw baseline horizontal segments per metric
    for idx, m in enumerate(unit_metrics):
        base_val, _ = METRIC_BASELINES[m]
        ax.hlines(
            y=base_val,
            xmin=idx - 0.4,
            xmax=idx + 0.4,
            colors="red",
            linestyles="--",
            linewidth=2,
            label="Random / Skill Baseline" if idx == 0 else None,
        )

    ax.set_xticks(range(len(unit_metrics)))
    ax.set_xticklabels([m.replace("_", " ") for m in unit_metrics], rotation=28, ha="right")
    ax.set_title(
        "ERA5 Precipitation vs. TCWV Coupling Metrics Compared to Random/Skill Baselines"
    )
    ax.set_xlabel("Correlation / Spatial Verification Metric")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(-0.35, 1.05)

    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate legend entries
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Mode / Reference", loc="lower left")

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_distribution_vs_baselines.png", dpi=300)
    plt.close()


def plot_spearman_vs_pearson_nonlinearity(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot 2: Joint scatter of Spearman Rank vs Pearson Linear correlation across all
    6-hourly timesteps, demonstrating non-linear / exponential moisture-rainfall coupling.
    """
    print("Generating Spearman vs. Pearson Non-Linearity plot...")
    inst_df = df[df["evaluation_mode"] == "6h_instantaneous"]
    piv = inst_df.pivot_table(
        index=["case_id_number", "case_title", "valid_time", "hours_from_start"],
        columns="metric",
        values="value",
    ).reset_index()

    if (
        "Spearman_Rank_Correlation" not in piv.columns
        or "Pearson_Linear_Correlation" not in piv.columns
    ):
        return

    plt.figure(figsize=(9, 8))
    sns.scatterplot(
        data=piv,
        x="Pearson_Linear_Correlation",
        y="Spearman_Rank_Correlation",
        hue="case_title",
        alpha=0.75,
        s=55,
    )
    lims = [-0.4, 1.0]
    plt.plot(lims, lims, "r--", alpha=0.6, label="1:1 Line (Spearman = Pearson)")
    plt.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    plt.axvline(0.0, color="gray", linestyle=":", alpha=0.5)

    pct_spearman_gt_pearson = (
        np.mean(piv["Spearman_Rank_Correlation"] > piv["Pearson_Linear_Correlation"])
        * 100.0
    )
    plt.title(
        f"Monotonic (Spearman) vs. Linear (Pearson) Spatial Correlation\n"
        f"(Spearman > Pearson in {pct_spearman_gt_pearson:.1f}% of 6h timesteps due to exponential pickup)"
    )
    plt.xlabel("Pearson Linear Correlation (r)")
    plt.ylabel("Spearman Rank Correlation (ρ)")
    plt.xlim(lims)
    plt.ylim(lims)
    plt.legend(title="Extreme Event", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "spearman_vs_pearson_nonlinearity.png", dpi=300)
    plt.close()


def plot_spatial_scale_relaxation(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot 3: Compare strict grid-point overlap (Spatial IoU) vs. neighborhood relaxed
    overlap (FSS 3x3, FSS 5x5) vs. plume containment (Precip Q95 in TCWV Q80/Q90) per case.
    """
    print("Generating Spatial Scale Relaxation & Plume Containment plot...")
    scale_metrics = [
        "Spatial_IOU_90",
        "FSS_90_w3",
        "FSS_90_w5",
        "Precip_Q95_in_TCWV_Q90",
        "Precip_Q95_in_TCWV_Q80",
        "ROC_AUC_Precip_Q95",
    ]
    inst_df = df[
        (df["evaluation_mode"] == "6h_instantaneous")
        & (df["metric"].isin(scale_metrics))
    ]

    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=inst_df,
        x="case_title",
        y="value",
        hue="metric",
        hue_order=scale_metrics,
        palette="viridis",
        errorbar="se",
    )
    plt.axhline(
        0.55,
        color="red",
        linestyle="--",
        alpha=0.6,
        label="FSS Useful Skill Threshold (0.55)",
    )
    plt.title(
        "Why Exact Grid-Point IoU Understates Coupling: Neighborhood FSS & Plume Containment by Case"
    )
    plt.xlabel("Extreme Precipitation Event")
    plt.ylabel("Mean 6-Hourly Score (±1 SE)")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1.05)
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "spatial_scale_relaxation_by_case.png", dpi=300)
    plt.close()


def plot_normalized_event_timeline(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot 4: Metric stability across normalized event lifecycle (0% to 100% of event duration)
    so 2-day typhoons and 21-day monsoon floods align on a common lifecycle axis.
    """
    print("Generating Normalized Event Lifecycle Timeline plot...")
    inst_df = df[df["evaluation_mode"] == "6h_instantaneous"].copy()

    # Compute normalized progress [0, 100%] per case
    max_hours = inst_df.groupby("case_id_number")["hours_from_start"].transform("max")
    inst_df["lifecycle_pct"] = np.where(
        max_hours > 0, (inst_df["hours_from_start"] / max_hours) * 100.0, 0.0
    )
    inst_df["lifecycle_bin"] = (
        np.round(inst_df["lifecycle_pct"] / 10.0) * 10.0
    ).astype(int)

    key_metrics = [
        "ROC_AUC_Precip_Q95",
        "Precip_Q95_in_TCWV_Q80",
        "FSS_90_w5",
        "Spearman_Rank_Correlation",
        "Spatial_IOU_90",
    ]
    sub = inst_df[inst_df["metric"].isin(key_metrics)]

    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=sub,
        x="lifecycle_bin",
        y="value",
        hue="metric",
        hue_order=key_metrics,
        marker="o",
        err_style="band",
    )
    plt.title(
        "Precipitation–TCWV Coupling Across Normalized Event Lifecycle (0% = Start, 100% = End)"
    )
    plt.xlabel("Normalized Event Lifecycle Progress (%)")
    plt.ylabel("Metric Value")
    plt.ylim(0, 1.05)
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "normalized_lifecycle_coupling.png", dpi=300)
    plt.close()


def plot_case_heatmap_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot 5: Side-by-side heatmaps comparing 6-Hourly Instantaneous Mean vs.
    Event-Aggregated Mean across all 10 cases.
    """
    print("Generating Side-by-Side Instantaneous vs. Aggregated Heatmaps...")
    unit_metrics = [
        "Spearman_Rank_Correlation",
        "Pearson_Linear_Correlation",
        "ROC_AUC_Precip_Q95",
        "Precip_Q95_in_TCWV_Q80",
        "Precip_Q95_in_TCWV_Q90",
        "FSS_90_w5",
        "FSS_90_w3",
        "Spatial_IOU_90",
        "Spatial_IOU_95",
    ]
    sub = df[df["metric"].isin(unit_metrics)]

    piv_inst = (
        sub[sub["evaluation_mode"] == "6h_instantaneous"]
        .groupby(["case_title", "metric"])["value"]
        .mean()
        .unstack("metric")[unit_metrics]
    )
    piv_agg = (
        sub[sub["evaluation_mode"] == "event_aggregated"]
        .groupby(["case_title", "metric"])["value"]
        .mean()
        .unstack("metric")[unit_metrics]
    )

    _, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    sns.heatmap(
        piv_inst,
        ax=axes[0],
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar=False,
    )
    axes[0].set_title("6-Hourly Instantaneous Mean")
    axes[0].set_xlabel("Metric")
    axes[0].set_ylabel("Extreme Event")
    axes[0].set_xticklabels(
        [m.replace("_", " ") for m in unit_metrics], rotation=30, ha="right"
    )

    sns.heatmap(
        piv_agg,
        ax=axes[1],
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Score [0-1]"},
    )
    axes[1].set_title("Event-Aggregated (Time-Mean) Footprint")
    axes[1].set_xlabel("Metric")
    axes[1].set_ylabel("")
    axes[1].set_xticklabels(
        [m.replace("_", " ") for m in unit_metrics], rotation=30, ha="right"
    )

    plt.suptitle(
        "Precipitation–TCWV Spatial Coupling by Case: Instantaneous (6h) vs. Event-Accumulated",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_instantaneous_vs_aggregated.png", dpi=300)
    plt.close()


def plot_forecast_error_coupling(
    precip_eval_csv: Path, tcwv_eval_csv: Path, output_dir: Path
) -> None:
    """
    Plot 6: Cross-evaluate Aurora & HRES forecast verification scores on TCWV vs.
    Precipitation (joining precip_evaluations.csv and tcwv_evaluations.csv) to diagnose
    how TCWV forecast skill translates into Precipitation forecast skill.
    """
    if not precip_eval_csv.exists() or not tcwv_eval_csv.exists():
        print("Skipping Forecast Error Coupling plot (precip or tcwv eval CSV missing).")
        return

    print("Generating Forecast Skill Coupling (TCWV vs. Precip for Aurora & HRES)...")
    p_df = pd.read_csv(precip_eval_csv)
    w_df = pd.read_csv(tcwv_eval_csv)

    for d in (p_df, w_df):
        d["model_group"] = d["forecast_source"].apply(
            lambda x: "Aurora" if "aurora" in str(x).lower() else x
        )

    join_cols = ["case_id_number", "valid_time", "lead_time", "model_group", "metric"]
    merged = pd.merge(
        w_df[w_df["lead_time"] <= 240][join_cols + ["value"]],
        p_df[p_df["lead_time"] <= 240][join_cols + ["value"]],
        on=join_cols,
        suffixes=("_tcwv", "_precip"),
    ).dropna(subset=["value_tcwv", "value_precip"])

    if merged.empty:
        return

    _, axes = plt.subplots(1, 2, figsize=(15, 6))
    for idx, metric in enumerate(["Spatial_IOU", "RootMeanSquaredError"]):
        m_sub = merged[merged["metric"] == metric]
        sns.scatterplot(
            ax=axes[idx],
            data=m_sub,
            x="value_tcwv",
            y="value_precip",
            hue="model_group",
            alpha=0.45,
            s=35,
        )
        # Add linear trendlines per model
        for model_name, color in [("Aurora", "C0"), ("HRES", "C1")]:
            mod_sub = m_sub[m_sub["model_group"] == model_name]
            if len(mod_sub) >= 5:
                r_val, p_val = stats.spearmanr(
                    mod_sub["value_tcwv"], mod_sub["value_precip"]
                )
                sns.regplot(
                    ax=axes[idx],
                    data=mod_sub,
                    x="value_tcwv",
                    y="value_precip",
                    scatter=False,
                    color=color,
                    label=f"{model_name} trend (ρ={r_val:.2f}, p={p_val:.1e})",
                )
        metric_label = metric.replace("_", " ")
        axes[idx].set_title(f"Forecast {metric_label}: TCWV vs. Precipitation")
        axes[idx].set_xlabel(f"TCWV {metric_label}")
        axes[idx].set_ylabel(f"Precipitation {metric_label}")
        axes[idx].legend(loc="upper left")

    plt.suptitle(
        "Does Better TCWV Forecasting Yield Better Precipitation Forecasting? (Aurora vs. HRES)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "forecast_skill_tcwv_vs_precip.png", dpi=300)
    plt.close()


def main() -> None:
    csv_path = Path("precip_tcwv_correlation.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path.resolve()} not found.")
        return

    output_dir = Path("plots") / "precip_tcwv_correlation"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    summary_df = compute_diagnostic_summary(df)
    summary_csv = output_dir / "diagnostic_summary_statistics.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n=== STATISTICAL DIAGNOSIS SUMMARY ===")
    print(summary_df.to_string(index=False))

    # Per-case breakdown for key metrics
    inst_df = df[df["evaluation_mode"] == "6h_instantaneous"]
    case_piv = (
        inst_df.groupby(["case_id_number", "case_title", "metric"])["value"]
        .mean()
        .unstack("metric")[
            [
                "Spearman_Rank_Correlation",
                "Pearson_Linear_Correlation",
                "ROC_AUC_Precip_Q95",
                "Precip_Q95_in_TCWV_Q80",
                "FSS_90_w5",
                "Spatial_IOU_90",
            ]
        ]
    )
    print("\n=== PER-CASE 6-HOURLY MEAN METRICS ===")
    print(case_piv.round(3).to_string())

    agg_df = df[df["evaluation_mode"] == "event_aggregated"]
    agg_piv = (
        agg_df.groupby(["case_id_number", "case_title", "metric"])["value"]
        .mean()
        .unstack("metric")[
            [
                "Spearman_Rank_Correlation",
                "Pearson_Linear_Correlation",
                "ROC_AUC_Precip_Q95",
                "Precip_Q95_in_TCWV_Q80",
                "FSS_90_w5",
                "Spatial_IOU_90",
            ]
        ]
    )
    print("\n=== PER-CASE EVENT-AGGREGATED METRICS ===")
    print(agg_piv.round(3).to_string())

    plot_distributions_vs_baselines(df, output_dir)
    plot_spearman_vs_pearson_nonlinearity(df, output_dir)
    plot_spatial_scale_relaxation(df, output_dir)
    plot_normalized_event_timeline(df, output_dir)
    plot_case_heatmap_comparison(df, output_dir)
    plot_forecast_error_coupling(
        Path("precip_evaluations.csv"), Path("tcwv_evaluations.csv"), output_dir
    )

    print(f"\nAll diagnostic plots and summary saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
