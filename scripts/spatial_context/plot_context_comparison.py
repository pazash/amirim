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

# Metadata and presentation settings for each evaluated variable
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


def load_and_preprocess_data(csv_path: Path) -> pd.DataFrame:
    """Loads RMSE evaluation CSV and standardizes model names and variable column."""
    df = pd.read_csv(csv_path)

    # Clean model names for legend
    df["model"] = df["model"].str.replace("Aurora_", "", regex=False)

    # Ensure numeric types for plotting
    df["lead_time_hours"] = pd.to_numeric(df["lead_time_hours"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Backward compatibility: default missing variable column to 2t
    if "variable" not in df.columns:
        df["variable"] = "2t"

    return df


def plot_all_variables_rmse_grid(
    df: pd.DataFrame,
    output_path: Path,
    plot_type: str = "line",
):
    """
    Generates a single composite image containing a subplot for each variable.
    The x-axis represents lead time (hours) and the y-axis represents the RMSE.
    """
    # Identify unique variables present in data, sorted according to logical order
    present_vars = [v for v in DEFAULT_VAR_ORDER if v in df["variable"].unique()]
    # Append any unexpected variables if present
    for v in df["variable"].unique():
        if v not in present_vars:
            present_vars.append(v)

    n_vars = len(present_vars)
    if n_vars == 0:
        logger.warning("No variables found in dataset to plot.")
        return

    # Determine grid layout (e.g. 3x3 for 9 variables)
    ncols = 3 if n_vars >= 3 else n_vars
    nrows = math.ceil(n_vars / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        sharex=False,
        sharey=False,
    )

    # Normalize axes to flat 1D array
    if n_vars == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).flatten()

    palette = sns.color_palette("viridis", n_colors=df["model"].nunique())
    all_lead_times = sorted(df["lead_time_hours"].unique())

    handles, labels = None, None

    for idx, var_id in enumerate(present_vars):
        ax = axes[idx]
        var_df = df[df["variable"] == var_id]

        meta = VARIABLE_METADATA.get(var_id, {"title": var_id, "unit": ""})
        title_str = meta["title"]
        unit_str = meta["unit"]

        if plot_type == "line":
            sns.lineplot(
                data=var_df,
                x="lead_time_hours",
                y="value",
                hue="model",
                style="model",
                markers=True,
                dashes=False,
                linewidth=2.4,
                markersize=8,
                errorbar=("ci", 95),
                palette=palette,
                ax=ax,
            )
        elif plot_type == "bar":
            sns.barplot(
                data=var_df,
                x="lead_time_hours",
                y="value",
                hue="model",
                order=all_lead_times,
                palette=palette,
                ax=ax,
            )
        elif plot_type == "box":
            sns.boxplot(
                data=var_df,
                x="lead_time_hours",
                y="value",
                hue="model",
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

        # Capture legend handles & labels once, then remove individual legend
        curr_handles, curr_labels = ax.get_legend_handles_labels()
        if curr_handles and (handles is None or len(curr_handles) > len(handles)):
            handles, labels = curr_handles, curr_labels

        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Hide any unused subplots in the grid
    for unused_idx in range(n_vars, len(axes)):
        axes[unused_idx].set_visible(False)

    # Add single global legend at the top
    if handles and labels:
        fig.legend(
            handles,
            labels,
            title="Context Scope",
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

    plot_title_kind = "Line Plot (95% CI)" if plot_type == "line" else ("Bar Plot" if plot_type == "bar" else "Box Plot")
    fig.suptitle(
        f"Context Scope Comparison: RMSE vs Lead Time by Variable ({plot_title_kind})",
        fontsize=17,
        fontweight="bold",
        y=0.998,
    )

    # Adjust layout to accommodate suptitle and top legend
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
    logger.info(f"Saved {plot_type} grid plot to: {output_path.resolve()}")


def generate_context_comparison_plots(
    csv_path: str = "aurora_context_comparison_rmse.csv",
    output_dir: str = "plots/context_comparison",
):
    """
    Reads the context comparison RMSE data and generates:
    1. A single composite image with subplots for each variable (x: lead time, y: RMSE).
    2. Bar and Box plot versions for detailed distribution analysis.
    """
    csv_file = Path(csv_path)
    out_path = Path(output_dir)

    if not csv_file.exists():
        logger.error(f"Data file {csv_path} not found. Please run the evaluation script first.")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    df = load_and_preprocess_data(csv_file)
    sns.set_theme(style="whitegrid")

    # 1. Primary requested plot: Single image with a subplot for each variable (Line plot: RMSE vs Lead Time)
    primary_image_path = out_path / "all_variables_rmse_vs_lead_time_line.png"
    plot_all_variables_rmse_grid(df, primary_image_path, plot_type="line")

    # Also save as avg_rmse_vs_lead_time_line.png for backward compatibility
    compat_line_path = out_path / "avg_rmse_vs_lead_time_line.png"
    plot_all_variables_rmse_grid(df, compat_line_path, plot_type="line")

    # 2. Bar plot grid by context scope and lead time
    bar_image_path = out_path / "all_variables_rmse_by_lead_time_bar.png"
    plot_all_variables_rmse_grid(df, bar_image_path, plot_type="bar")

    # 3. Box plot grid showing variance across different initializations
    box_image_path = out_path / "all_variables_rmse_distribution_boxplot.png"
    plot_all_variables_rmse_grid(df, box_image_path, plot_type="box")

    logger.info(f"All context comparison plots successfully generated in {out_path.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate context comparison plots for Aurora scopes")
    parser.add_argument(
        "--csv",
        type=str,
        default="aurora_context_comparison_rmse.csv",
        help="Path to the RMSE results CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/context_comparison",
        help="Directory to save generated plot images",
    )
    args = parser.parse_args()

    generate_context_comparison_plots(csv_path=args.csv, output_dir=args.output_dir)

