import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def compare_performance(extreme_csv="heatwave_evaluations.csv", normal_csv="normal_evaluations.csv", output_dir="plots"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not Path(extreme_csv).exists() or not Path(normal_csv).exists():
        logger.error(f"Missing input CSV files. Ensure both {extreme_csv} and {normal_csv} exist.")
        return

    logger.info(f"Loading data: {extreme_csv} and {normal_csv}")
    df_extreme = pd.read_csv(extreme_csv)
    df_normal = pd.read_csv(normal_csv)
    
    # We want to aggregate by metric, forecast_source, and lead_time
    # Since aurora models might be named "aurora1.5_L24", we simplify to "Aurora" for plotting, 
    # but since HRES is also there, we map accordingly.
    def clean_model_name(name):
        if "aurora" in name.lower():
            return "Aurora V1.5"
        return name
        
    df_extreme["model"] = df_extreme["forecast_source"].apply(clean_model_name)
    df_normal["model"] = df_normal["forecast_source"].apply(clean_model_name)

    # 1. RMSE Comparison
    rmse_extreme = df_extreme[df_extreme["metric"] == "RootMeanSquaredError"].groupby(["model", "lead_time"])["value"].mean().reset_index()
    rmse_normal = df_normal[df_normal["metric"] == "RootMeanSquaredError"].groupby(["model", "lead_time"])["value"].mean().reset_index()
    
    rmse_merged = pd.merge(rmse_extreme, rmse_normal, on=["model", "lead_time"], suffixes=("_extreme", "_normal"))
    rmse_merged["rmse_ratio"] = rmse_merged["value_extreme"] / rmse_merged["value_normal"]

    # 2. Peak Amplitude Error Comparison
    peak_extreme = df_extreme[df_extreme["metric"] == "Peak_Amplitude_Error"].groupby(["model", "lead_time"])["value"].mean().reset_index()
    peak_normal = df_normal[df_normal["metric"] == "Peak_Amplitude_Error"].groupby(["model", "lead_time"])["value"].mean().reset_index()
    
    peak_merged = pd.merge(peak_extreme, peak_normal, on=["model", "lead_time"], suffixes=("_extreme", "_normal"))
    peak_merged["peak_bias_diff"] = peak_merged["value_extreme"] - peak_merged["value_normal"]

    # Save numeric results
    rmse_merged.to_csv(f"{output_dir}/rmse_comparison.csv", index=False)
    peak_merged.to_csv(f"{output_dir}/peak_error_comparison.csv", index=False)
    logger.info(f"Saved numerical comparisons to {output_dir}/")

    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    
    # Plot 1: RMSE Ratio
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=rmse_merged, x="lead_time", y="rmse_ratio", hue="model", marker="o", linewidth=2)
    plt.axhline(1.0, color="black", linestyle="--", alpha=0.6, label="Equal Performance")
    plt.title("RMSE Ratio (Extreme / Normal Weather)\nHigher ratio = model struggles more with extremes", fontsize=14)
    plt.xlabel("Lead Time (Hours)", fontsize=12)
    plt.ylabel("RMSE Ratio", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rmse_ratio.png", dpi=300)
    plt.close()
    
    # Plot 2: Peak Amplitude Error Comparison
    # We plot the absolute mean peak error for both regimes
    plt.figure(figsize=(12, 6))
    
    # Prepare data for Seaborn
    peak_melt = pd.melt(
        peak_merged, 
        id_vars=["model", "lead_time"], 
        value_vars=["value_extreme", "value_normal"],
        var_name="regime", 
        value_name="peak_error"
    )
    peak_melt["regime"] = peak_melt["regime"].replace({"value_extreme": "Extreme (Heatwave)", "value_normal": "Normal Weather"})
    
    sns.lineplot(
        data=peak_melt, 
        x="lead_time", 
        y="peak_error", 
        hue="model", 
        style="regime", 
        markers=True, 
        dashes=True,
        linewidth=2
    )
    plt.axhline(0, color="black", linestyle="-", alpha=0.3)
    plt.title("Mean Peak Amplitude Error (Forecast - Target)\nNegative values indicate underprediction of peaks", fontsize=14)
    plt.xlabel("Lead Time (Hours)", fontsize=12)
    plt.ylabel("Peak Error (K)", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/peak_amplitude_error.png", dpi=300)
    plt.close()
    
    logger.info(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    compare_performance()
