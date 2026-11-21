"""
Plotting suite for out-of-sample heatwave evaluations (oos_heatwave_evaluations.csv).

Generates comparative charts modeled on ``plot_heatwave_eval.py`` and
``plot_ar_eval.py``:
  1. Primary comparison (Aurora 0.25 Pretrained vs. Aurora 0.25 Finetuned vs.
     ECMWF HRES) in ``plots/oos_heatwave/``.
  2. Extended comparison including the ERA5 Persistence lower-bound baseline in
     ``plots/oos_heatwave/with_persistence/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "aurora_v1p5": "Aurora V1.5",
    "aurora_pretrained": "Aurora 0.25 Pretrained",
    "aurora_finetuned": "Aurora 0.25 Finetuned",
    "HRES": "ECMWF HRES",
    "Persistence": "ERA5 Persistence",
}

MODEL_ORDER: list[str] = [
    "Aurora V1.5",
    "Aurora 0.25 Pretrained",
    "Aurora 0.25 Finetuned",
    "ECMWF HRES",
    "ERA5 Persistence",
]

MODEL_PALETTE: dict[str, str] = {
    "Aurora V1.5": "#9467bd",
    "Aurora 0.25 Pretrained": "#1f77b4",
    "Aurora 0.25 Finetuned": "#2ca02c",
    "ECMWF HRES": "#d62728",
    "ERA5 Persistence": "#7f7f7f",
}


def load_and_clean_data(
    csv_path: Path,
    include_persistence: bool = False,
) -> pd.DataFrame:
    """
    Load OOS heatwave evaluations and align all active models onto the exact
    intersection of (case_id_number, init_time, valid_time) targets up to 240h.
    """
    print(f"Loading data from {csv_path} (include_persistence={include_persistence})...")
    df = pd.read_csv(csv_path)

    df = df[df["lead_time"] <= 240].copy()
    df = df.dropna(subset=["value"])

    if "model_family" in df.columns:
        df["model_group"] = df["model_family"].map(
            lambda x: MODEL_DISPLAY_NAMES.get(str(x), str(x))
        )
    else:
        df["model_group"] = df["forecast_source"].astype(str)

    if "is_out_of_sample" in df.columns:
        aurora_mask = df["model_group"].str.contains("aurora", case=False, na=False)
        has_oos_aurora = bool(
            (df.loc[aurora_mask, "is_out_of_sample"] == True).any()  # noqa: E712
        )
        if has_oos_aurora:
            df = df[df["is_out_of_sample"] == True]  # noqa: E712

    if not include_persistence:
        df = df[df["model_group"] != MODEL_DISPLAY_NAMES["Persistence"]]

    active_groups = [g for g in MODEL_ORDER if g in set(df["model_group"].unique())]
    if not active_groups:
        raise ValueError("No valid model groups found in evaluation CSV.")

    key_cols = ["case_id_number", "init_time", "valid_time"]
    common_targets: pd.DataFrame | None = None
    for group in active_groups:
        group_keys = df[df["model_group"] == group][key_cols].drop_duplicates()
        common_targets = (
            group_keys
            if common_targets is None
            else pd.merge(common_targets, group_keys, on=key_cols, how="inner")
        )

    if common_targets is None or common_targets.empty:
        raise ValueError(
            "No common (case_id_number, init_time, valid_time) targets shared across models."
        )

    df = pd.merge(df, common_targets, on=key_cols, how="inner")
    print(
        f"Filtered down to {len(common_targets)} common evaluation targets across {active_groups}."
    )
    return df


def _active_order_and_palette(df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    present = set(df["model_group"].unique())
    order = [m for m in MODEL_ORDER if m in present]
    palette = {m: MODEL_PALETTE[m] for m in order if m in MODEL_PALETTE}
    return order, palette


def plot_degradation_over_lead_time(
    df: pd.DataFrame,
    output_dir: Path,
    regime_label: str = "Out-of-Sample Heatwave",
) -> None:
    """Plot 1: Line plots for all 5 metrics showing degradation over lead time."""
    print("Generating Lead Time Degradation plots...")
    hue_order, palette = _active_order_and_palette(df)

    for metric in df["metric"].unique():
        plt.figure(figsize=(10, 6))
        metric_df = df[df["metric"] == metric]
        metric_label = metric.replace("_", " ")

        sns.lineplot(
            data=metric_df,
            x="lead_time",
            y="value",
            hue="model_group",
            hue_order=hue_order,
            palette=palette,
            marker="o",
            err_style="band",
        )

        plt.title(f"{regime_label} Degradation Over Lead Time: {metric_label}")
        plt.xlabel("Lead Time (Hours)")
        plt.ylabel(f"{metric_label} Value")

        if metric in ("Spatial_IOU", "Extreme_Area_Ratio"):
            plt.axhline(
                y=1.0, color="black", linestyle="--", alpha=0.5, label="Perfect Score (1.0)"
            )
        elif metric in (
            "Peak_Amplitude_Error",
            "Conditional_Bias_Extremes",
            "RootMeanSquaredError",
        ):
            plt.axhline(
                y=0.0, color="black", linestyle="--", alpha=0.5, label="Perfect Score (0.0)"
            )

        plt.legend(title="Model")
        plt.tight_layout()
        plt.savefig(output_dir / f"degradation_{metric}.png", dpi=300)
        plt.close()


def plot_focused_peak_bias_and_iou(
    df: pd.DataFrame,
    output_dir: Path,
    regime_label: str = "Out-of-Sample Heatwave",
) -> None:
    """Plot 2: Focused 1x2 comparison on Peak Amplitude Error and Spatial IOU."""
    print("Generating Focused Peak Bias and IOU plot...")
    hue_order, palette = _active_order_and_palette(df)
    focused_metrics = ["Peak_Amplitude_Error", "Spatial_IOU"]

    _, axes = plt.subplots(1, 2, figsize=(16, 6))
    for i, metric in enumerate(focused_metrics):
        metric_df = df[df["metric"] == metric]
        metric_label = metric.replace("_", " ")
        sns.lineplot(
            ax=axes[i],
            data=metric_df,
            x="lead_time",
            y="value",
            hue="model_group",
            hue_order=hue_order,
            palette=palette,
            marker="o",
            err_style="band",
        )
        axes[i].set_title(f"{regime_label}s: {metric_label} vs Lead Time")
        axes[i].set_xlabel("Lead Time (Hours)")
        axes[i].set_ylabel(metric_label)

        if metric == "Spatial_IOU":
            axes[i].axhline(
                y=1.0, color="black", linestyle="--", alpha=0.5, label="Perfect Score (1.0)"
            )
        elif metric == "Peak_Amplitude_Error":
            axes[i].axhline(
                y=0.0, color="black", linestyle="--", alpha=0.5, label="Perfect Score (0.0)"
            )
        axes[i].legend(title="Model")

    plt.tight_layout()
    plt.savefig(output_dir / "focused_peak_bias_iou.png", dpi=300)
    plt.close()


def plot_error_distributions(
    df: pd.DataFrame,
    output_dir: Path,
    regime_label: str = "Out-of-Sample Heatwave",
) -> None:
    """Plot 3: Boxplots for error distributions grouped into 3-day lead-time buckets."""
    print("Generating Error Distribution Boxplots...")
    hue_order, palette = _active_order_and_palette(df)

    bins = [0, 72, 144, 241]
    labels = ["0-3 Days", "3-6 Days", "6-10 Days"]
    binned_df = df.copy()
    binned_df["lead_time_bucket"] = pd.cut(
        binned_df["lead_time"], bins=bins, labels=labels, right=False
    )

    for metric in binned_df["metric"].unique():
        plt.figure(figsize=(12, 6))
        metric_df = binned_df[binned_df["metric"] == metric]
        metric_label = metric.replace("_", " ")

        sns.boxplot(
            data=metric_df,
            x="lead_time_bucket",
            y="value",
            hue="model_group",
            hue_order=hue_order,
            palette=palette,
        )

        if metric in ("Spatial_IOU", "Extreme_Area_Ratio"):
            plt.axhline(y=1.0, color="black", linestyle="--", alpha=0.4)
        elif metric in (
            "Peak_Amplitude_Error",
            "Conditional_Bias_Extremes",
            "RootMeanSquaredError",
        ):
            plt.axhline(y=0.0, color="black", linestyle="--", alpha=0.4)

        plt.title(f"{regime_label} Error Distribution: {metric_label}")
        plt.xlabel("Lead Time Window")
        plt.ylabel(f"{metric_label} Value")
        plt.legend(title="Model")
        plt.tight_layout()
        plt.savefig(output_dir / f"distribution_{metric}.png", dpi=300)
        plt.close()


def plot_spatial_overlap_vs_bias(
    df: pd.DataFrame,
    output_dir: Path,
    regime_label: str = "Out-of-Sample Heatwave",
) -> None:
    """Plot 4: Scatter plot of Spatial Overlap (IOU) vs. Extreme Area Ratio."""
    print("Generating Spatial Overlap vs Area Bias Scatter Plot...")
    hue_order, palette = _active_order_and_palette(df)

    pivot_df = df.pivot_table(
        index=["case_id_number", "init_time", "valid_time", "model_group", "lead_time"],
        columns="metric",
        values="value",
    ).reset_index()

    if "Spatial_IOU" not in pivot_df.columns or "Extreme_Area_Ratio" not in pivot_df.columns:
        print("Skipping Spatial Overlap plot (metrics missing).")
        return

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=pivot_df,
        x="Extreme_Area_Ratio",
        y="Spatial_IOU",
        hue="model_group",
        hue_order=hue_order,
        palette=palette,
        size="lead_time",
        sizes=(20, 180),
        alpha=0.55,
    )

    plt.axhline(y=1.0, color="black", linestyle="--", alpha=0.4, label="Perfect IOU (1.0)")
    plt.axvline(
        x=1.0, color="black", linestyle=":", alpha=0.4, label="Perfect Area Bias (1.0)"
    )

    plt.title(f"{regime_label}s: Spatial Overlap (IOU) vs Extreme Area Bias")
    plt.xlabel("Extreme Area Ratio (Forecast Area / Target Area)")
    plt.ylabel("Spatial IOU")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "spatial_overlap_vs_bias.png", dpi=300)
    plt.close()


def generate_plot_suite(
    csv_path: Path, output_dir: Path, include_persistence: bool
) -> None:
    """Load data and write all comparative plots into *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_clean_data(csv_path, include_persistence=include_persistence)
    aurora_mask = df["model_group"].str.contains("aurora", case=False, na=False)
    is_insample = (
        "is_out_of_sample" in df.columns
        and bool(aurora_mask.any())
        and not bool((df.loc[aurora_mask, "is_out_of_sample"] == True).any())  # noqa: E712
    )
    regime_label = (
        "In-Sample (Trained-On) Heatwave"
        if is_insample
        else "Out-of-Sample Heatwave"
    )
    plot_degradation_over_lead_time(df, output_dir, regime_label=regime_label)
    plot_focused_peak_bias_and_iou(df, output_dir, regime_label=regime_label)
    plot_error_distributions(df, output_dir, regime_label=regime_label)
    plot_spatial_overlap_vs_bias(df, output_dir, regime_label=regime_label)
    print(f"Plots saved to {output_dir.resolve()}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate out-of-sample heatwave evaluation plots."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("oos_heatwave_evaluations.csv"),
        help="Path to out-of-sample heatwave evaluation CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots") / "oos_heatwave",
        help="Output directory for generated plots.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.csv_path.exists():
        print(
            f"Error: {args.csv_path.resolve()} not found. "
            "Run oos_heatwave_evaluator.py first."
        )
        return

    generate_plot_suite(args.csv_path, args.output_dir, include_persistence=False)
    generate_plot_suite(
        args.csv_path, args.output_dir / "with_persistence", include_persistence=True
    )


if __name__ == "__main__":
    main()
