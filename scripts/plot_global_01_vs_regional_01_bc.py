"""
Generate comparison plots from the outputs of `compare_global_01_vs_regional_01_bc.py`.

Generates:
  1. Multi-variable RMSE comparison plots across the 3 model types
     (Global_0.1, Regional_0.1_BC, Regional_0.1), including bar charts, line charts,
     and box plots in a dedicated subfolder under `plots/` (`plots/global_01_vs_regional_01_bc`).
  2. Resource usage comparison plots (Wall-Clock Time, CPU RAM Usage, Peak GPU Memory)
     across models (including Global_0.1, Regional_0.1_BC, 0.1+bc, Regional_0.1)
     both as a multi-panel composite figure and individual plots in a dedicated
     resource subfolder (`plots/global_01_vs_regional_01_bc/resources`).
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Metadata and presentation settings for each evaluated variable (matching plot_context_comparison.py)
VARIABLE_METADATA = {
    "2t": {
        "title": "2m Surface Temperature",
        "unit": "K",
        "order": 1,
    },
    "t_850": {
        "title": "Temperature (850 hPa - Low)",
        "unit": "K",
        "order": 2,
    },
    "t_250": {
        "title": "Temperature (250 hPa - High)",
        "unit": "K",
        "order": 3,
    },
    "wind_speed_850": {
        "title": "Wind Speed (850 hPa - Low)",
        "unit": "m/s",
        "order": 4,
    },
    "wind_speed_250": {
        "title": "Wind Speed (250 hPa - High)",
        "unit": "m/s",
        "order": 5,
    },
    "z_850": {
        "title": "Geopotential (850 hPa - Low)",
        "unit": "m²/s²",
        "order": 6,
    },
    "z_250": {
        "title": "Geopotential (250 hPa - High)",
        "unit": "m²/s²",
        "order": 7,
    },
    "q_850": {
        "title": "Specific Humidity (850 hPa - Low)",
        "unit": "kg/kg",
        "order": 8,
    },
    "q_250": {
        "title": "Specific Humidity (250 hPa - High)",
        "unit": "kg/kg",
        "order": 9,
    },
}

DEFAULT_VAR_ORDER = [
    "2t",
    "t_850",
    "t_250",
    "wind_speed_850",
    "wind_speed_250",
    "z_850",
    "z_250",
    "q_850",
    "q_250",
]

MODEL_ORDER = ["Global_0.1", "Regional_0.1_BC", "0.1+bc", "Regional_0.1"]


def _get_ordered_models(df: pd.DataFrame) -> list[str]:
    """Return models present in `df` ordered by `MODEL_ORDER` first, then any extras."""
    present = df["model"].unique().tolist()
    ordered = [m for m in MODEL_ORDER if m in present]
    for m in present:
        if m not in ordered:
            ordered.append(m)
    return ordered


def load_and_preprocess_rmse_data(csv_path: Path) -> pd.DataFrame:
    """Loads RMSE evaluation CSV and standardizes model names and variable column."""
    df = pd.read_csv(csv_path)

    # Clean model names for legend (e.g. "Aurora_Global_0.1" -> "Global_0.1")
    df["model"] = df["model"].astype(str).str.replace("Aurora_", "", regex=False)

    # Ensure numeric types for plotting
    df["lead_time_hours"] = pd.to_numeric(df["lead_time_hours"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if "variable" not in df.columns:
        df["variable"] = "2t"

    return df


def load_and_preprocess_resource_data(csv_path: Path) -> pd.DataFrame:
    """Loads resource benchmark CSV and ensures numeric types."""
    df = pd.read_csv(csv_path)
    df["model"] = df["model"].astype(str).str.replace("Aurora_", "", regex=False)
    df["lead_time_hours"] = pd.to_numeric(df["lead_time_hours"], errors="coerce")

    for col in ("time_sec", "ram_usage_MiB", "gpu_memory_MiB"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# 1. Multi-Variable RMSE Grid Plots
# ---------------------------------------------------------------------------
def plot_all_variables_rmse_grid(
    df: pd.DataFrame,
    output_path: Path,
    plot_type: str = "bar",
) -> None:
    """
    Generates a single composite image containing a subplot for each variable.
    The x-axis represents lead time (hours) and the y-axis represents the RMSE.
    """
    present_vars = [v for v in DEFAULT_VAR_ORDER if v in df["variable"].unique()]
    for v in df["variable"].unique():
        if v not in present_vars:
            present_vars.append(v)

    n_vars = len(present_vars)
    if n_vars == 0:
        logger.warning("No variables found in dataset to plot.")
        return

    ncols = 3 if n_vars >= 3 else n_vars
    nrows = math.ceil(n_vars / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        sharex=False,
        sharey=False,
    )

    if n_vars == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    hue_order = _get_ordered_models(df)
    palette = sns.color_palette("viridis", n_colors=len(hue_order))
    all_lead_times = sorted(df["lead_time_hours"].dropna().unique())

    handles, labels = None, None

    for idx, var_id in enumerate(present_vars):
        ax = axes[idx]
        var_df = df[df["variable"] == var_id]

        meta = VARIABLE_METADATA.get(var_id, {"title": var_id, "unit": ""})
        title_str = meta["title"]
        unit_str = meta["unit"]

        if plot_type == "bar":
            sns.barplot(
                data=var_df,
                x="lead_time_hours",
                y="value",
                hue="model",
                hue_order=hue_order,
                order=all_lead_times,
                palette=palette,
                ax=ax,
            )
        elif plot_type == "line":
            sns.lineplot(
                data=var_df,
                x="lead_time_hours",
                y="value",
                hue="model",
                hue_order=hue_order,
                style="model",
                style_order=hue_order,
                markers=True,
                dashes=False,
                linewidth=2.4,
                markersize=8,
                errorbar=("ci", 95),
                palette=palette,
                ax=ax,
            )
        elif plot_type == "box":
            sns.boxplot(
                data=var_df,
                x="lead_time_hours",
                y="value",
                hue="model",
                hue_order=hue_order,
                order=all_lead_times,
                palette=palette,
                ax=ax,
            )

        ax.set_title(title_str, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Lead Time (Hours)", fontsize=11)
        ax.set_ylabel(f"RMSE ({unit_str})" if unit_str else "RMSE", fontsize=11)

        if plot_type == "line":
            ax.set_xticks(all_lead_times)

        # For specific humidity (small values), use scientific notation
        if unit_str == "kg/kg":
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, -2))
            ax.yaxis.set_major_formatter(formatter)

        curr_handles, curr_labels = ax.get_legend_handles_labels()
        if curr_handles and (handles is None or len(curr_handles) > len(handles)):
            handles, labels = curr_handles, curr_labels

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    for unused_idx in range(n_vars, len(axes)):
        axes[unused_idx].set_visible(False)

    if handles and labels:
        fig.legend(
            handles,
            labels,
            title="Model Type",
            title_fontsize="13",
            fontsize="11",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=len(labels),
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            framealpha=0.9,
        )

    plot_title_kind = (
        "Bar Plot"
        if plot_type == "bar"
        else ("Line Plot (95% CI)" if plot_type == "line" else "Box Plot")
    )
    fig.suptitle(
        f"Aurora 0.1° Global vs Regional (with & without BC): RMSE by Variable ({plot_title_kind})",
        fontsize=17,
        fontweight="bold",
        y=0.998,
    )

    plt.subplots_adjust(
        top=0.92,
        bottom=0.07,
        left=0.06,
        right=0.97,
        hspace=0.35,
        wspace=0.28,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved RMSE {plot_type} grid plot to: {output_path.resolve()}")


# ---------------------------------------------------------------------------
# 2. Resource Usage Plots (Subplots + Individual Charts)
# ---------------------------------------------------------------------------
RESOURCE_METRICS = [
    {
        "column": "time_sec",
        "title": "Wall-Clock Rollout Time",
        "ylabel": "Time (seconds)",
        "filename": "wall_clock_time.png",
    },
    {
        "column": "gpu_memory_MiB",
        "title": "Peak GPU Memory Allocated",
        "ylabel": "GPU Memory (MiB)",
        "filename": "gpu_peak_memory.png",
    },
    {
        "column": "ram_usage_MiB",
        "title": "CPU RAM Usage (RSS)",
        "ylabel": "CPU RAM (MiB)",
        "filename": "cpu_ram_usage.png",
    },
]


def plot_resource_usage_subplots(
    df: pd.DataFrame,
    output_path: Path,
    plot_type: str = "bar",
) -> None:
    """
    Generates a composite 1x3 subplot figure comparing all 3 resource metrics
    (time_sec, gpu_memory_MiB, ram_usage_MiB) across the 3 model types.
    """
    available_metrics = [m for m in RESOURCE_METRICS if m["column"] in df.columns]
    if not available_metrics:
        logger.warning("No expected resource columns found in resource CSV.")
        return

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_metrics,
        figsize=(6.5 * n_metrics, 5.5),
        sharex=False,
        sharey=False,
    )
    if n_metrics == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    hue_order = _get_ordered_models(df)
    palette = sns.color_palette("viridis", n_colors=len(hue_order))
    all_lead_times = sorted(df["lead_time_hours"].dropna().unique())

    handles, labels = None, None

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        col = metric["column"]

        if plot_type == "bar":
            sns.barplot(
                data=df,
                x="lead_time_hours",
                y=col,
                hue="model",
                hue_order=hue_order,
                order=all_lead_times,
                palette=palette,
                ax=ax,
            )
        else:
            sns.lineplot(
                data=df,
                x="lead_time_hours",
                y=col,
                hue="model",
                hue_order=hue_order,
                style="model",
                style_order=hue_order,
                markers=True,
                dashes=False,
                linewidth=2.5,
                markersize=8,
                palette=palette,
                ax=ax,
            )
            ax.set_xticks(all_lead_times)

        ax.set_title(metric["title"], fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Lead Time (Hours)", fontsize=11)
        ax.set_ylabel(metric["ylabel"], fontsize=11)

        curr_handles, curr_labels = ax.get_legend_handles_labels()
        if curr_handles and (handles is None or len(curr_handles) > len(handles)):
            handles, labels = curr_handles, curr_labels

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    if handles and labels:
        fig.legend(
            handles,
            labels,
            title="Model Type",
            title_fontsize="13",
            fontsize="11",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=len(labels),
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            framealpha=0.9,
        )

    kind_label = "Bar Chart" if plot_type == "bar" else "Line Chart"
    fig.suptitle(
        f"Resource Usage Comparison Across Model Types ({kind_label})",
        fontsize=16,
        fontweight="bold",
        y=1.08,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved resource {plot_type} subplots to: {output_path.resolve()}")


def plot_individual_resource_metrics(
    df: pd.DataFrame,
    resource_out_dir: Path,
) -> None:
    """
    Generates individual bar and line charts for each resource metric, plus a
    speedup bar chart at maximum lead time relative to Global_0.1.
    """
    resource_out_dir.mkdir(parents=True, exist_ok=True)
    hue_order = _get_ordered_models(df)
    palette = sns.color_palette("viridis", n_colors=len(hue_order))
    all_lead_times = sorted(df["lead_time_hours"].dropna().unique())

    for metric in RESOURCE_METRICS:
        col = metric["column"]
        if col not in df.columns:
            continue

        # Bar plot
        plt.figure(figsize=(9, 5.5))
        sns.barplot(
            data=df,
            x="lead_time_hours",
            y=col,
            hue="model",
            hue_order=hue_order,
            order=all_lead_times,
            palette=palette,
        )
        plt.title(f"{metric['title']} by Lead Time", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Lead Time (Hours)", fontsize=12)
        plt.ylabel(metric["ylabel"], fontsize=12)
        plt.legend(title="Model Type", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        bar_save_path = resource_out_dir / f"bar_{metric['filename']}"
        plt.savefig(bar_save_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Line plot
        plt.figure(figsize=(9, 5.5))
        sns.lineplot(
            data=df,
            x="lead_time_hours",
            y=col,
            hue="model",
            hue_order=hue_order,
            style="model",
            style_order=hue_order,
            markers=True,
            dashes=False,
            linewidth=2.5,
            markersize=8,
            palette=palette,
        )
        plt.title(f"{metric['title']} vs Lead Time", fontsize=14, fontweight="bold", pad=12)
        plt.xlabel("Lead Time (Hours)", fontsize=12)
        plt.ylabel(metric["ylabel"], fontsize=12)
        plt.xticks(all_lead_times)
        plt.legend(title="Model Type", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        line_save_path = resource_out_dir / f"line_{metric['filename']}"
        plt.savefig(line_save_path, dpi=300, bbox_inches="tight")
        plt.close()

    # Speedup relative to Global_0.1 at max lead time
    if "time_sec" in df.columns and not df.empty:
        max_lt = df["lead_time_hours"].max()
        df_max_lt = (
            df[df["lead_time_hours"] == max_lt]
            .groupby("model", as_index=False)["time_sec"]
            .mean()
        )
        if "Global_0.1" in df_max_lt["model"].values:
            global_time = float(
                df_max_lt.loc[df_max_lt["model"] == "Global_0.1", "time_sec"].iloc[0]
            )
            if global_time > 0:
                df_max_lt["speedup"] = global_time / df_max_lt["time_sec"]
                df_max_lt["model"] = pd.Categorical(
                    df_max_lt["model"], categories=hue_order, ordered=True
                )
                df_max_lt = df_max_lt.sort_values("model")

                plt.figure(figsize=(8, 5.5))
                ax = sns.barplot(
                    data=df_max_lt,
                    x="model",
                    y="speedup",
                    hue="model",
                    hue_order=hue_order,
                    palette=palette,
                    dodge=False,
                    legend=False,
                )
                plt.title(
                    f"Rollout Speedup Relative to Global_0.1 (Lead Time: {int(max_lt)}h)",
                    fontsize=14,
                    fontweight="bold",
                    pad=15,
                )
                plt.xlabel("Model Type", fontsize=12)
                plt.ylabel("Speedup Multiplier (x)", fontsize=12)

                max_speedup = df_max_lt["speedup"].max()
                plt.ylim(0, max_speedup * 1.15)
                for i, val in enumerate(df_max_lt["speedup"]):
                    ax.text(
                        i,
                        val + max_speedup * 0.02,
                        f"{val:.2f}x",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

                plt.tight_layout()
                speedup_path = resource_out_dir / "speedup_vs_global_01.png"
                plt.savefig(speedup_path, dpi=300, bbox_inches="tight")
                plt.close()
                logger.info(f"Saved speedup plot to: {speedup_path.resolve()}")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def generate_all_plots(
    rmse_csv: str = "aurora_01_global_vs_regional_bc_rmse.csv",
    resource_csv: str = "benchmark_01_global_vs_regional_bc.csv",
    output_dir: str = "plots/global_01_vs_regional_01_bc",
    resource_output_dir: str | None = None,
) -> None:
    """
    Reads the RMSE and resource CSVs generated by `compare_global_01_vs_regional_01_bc.py`
    and generates both the multi-variable RMSE comparison plots and resource usage plots.
    """
    out_path = Path(output_dir)
    res_out_path = (
        Path(resource_output_dir)
        if resource_output_dir is not None
        else out_path / "resources"
    )

    sns.set_theme(style="whitegrid")

    # 1. RMSE comparison plots across all variables and the 3 model types
    rmse_file = Path(rmse_csv)
    if rmse_file.exists():
        out_path.mkdir(parents=True, exist_ok=True)
        df_rmse = load_and_preprocess_rmse_data(rmse_file)

        # Primary requested bar chart grid comparing all variables RMSE across the 3 model types
        bar_image_path = out_path / "all_variables_rmse_by_lead_time_bar.png"
        plot_all_variables_rmse_grid(df_rmse, bar_image_path, plot_type="bar")

        # Line plot grid (RMSE vs Lead Time with 95% CI, matching plot_context_comparison.py)
        line_image_path = out_path / "all_variables_rmse_vs_lead_time_line.png"
        plot_all_variables_rmse_grid(df_rmse, line_image_path, plot_type="line")

        # Box plot grid showing distribution across initialization dates
        box_image_path = out_path / "all_variables_rmse_distribution_boxplot.png"
        plot_all_variables_rmse_grid(df_rmse, box_image_path, plot_type="box")
    else:
        logger.warning(
            f"RMSE CSV file '{rmse_csv}' not found. Skipping RMSE plots "
            "(run `compare_global_01_vs_regional_01_bc.py --evaluate` first)."
        )

    # 2. Resource usage comparison plots
    res_file = Path(resource_csv)
    if res_file.exists():
        res_out_path.mkdir(parents=True, exist_ok=True)
        df_res = load_and_preprocess_resource_data(res_file)

        # Composite 1x3 subplot comparing Time, GPU Memory, and RAM across the 3 models (Bar & Line)
        plot_resource_usage_subplots(
            df_res,
            res_out_path / "resource_usage_comparison_bar.png",
            plot_type="bar",
        )
        plot_resource_usage_subplots(
            df_res,
            res_out_path / "resource_usage_comparison_line.png",
            plot_type="line",
        )
        # Also save a copy of the composite resource subplot in the main subfolder for convenience
        out_path.mkdir(parents=True, exist_ok=True)
        plot_resource_usage_subplots(
            df_res,
            out_path / "resource_usage_subplots_bar.png",
            plot_type="bar",
        )

        # Individual metric plots & speedup chart in the resource subfolder
        plot_individual_resource_metrics(df_res, res_out_path)
    else:
        logger.warning(
            f"Resource CSV file '{resource_csv}' not found. Skipping resource plots "
            "(run `compare_global_01_vs_regional_01_bc.py --generate` first)."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate RMSE and resource usage plots from compare_global_01_vs_regional_01_bc.py outputs"
    )
    parser.add_argument(
        "--rmse-csv",
        type=str,
        default="aurora_01_global_vs_regional_bc_rmse.csv",
        help="Path to the RMSE evaluation CSV file",
    )
    parser.add_argument(
        "--resource-csv",
        type=str,
        default="benchmark_01_global_vs_regional_bc.csv",
        help="Path to the resource benchmark CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/global_01_vs_regional_01_bc",
        help="Subfolder under plots/ to save the RMSE comparison plots",
    )
    parser.add_argument(
        "--resource-output-dir",
        type=str,
        default=None,
        help="Subfolder to save resource usage plots (defaults to <output-dir>/resources)",
    )
    args = parser.parse_args()

    generate_all_plots(
        rmse_csv=args.rmse_csv,
        resource_csv=args.resource_csv,
        output_dir=args.output_dir,
        resource_output_dir=args.resource_output_dir,
    )

