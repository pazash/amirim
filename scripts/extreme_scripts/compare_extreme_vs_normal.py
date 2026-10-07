import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def compare_performance(extreme_csv="heatwave_evaluations.csv", normal_csv="normal_evaluations.csv", output_dir="plots/extreme_vs_normal"):
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

    # Filter for lead times up to 240 hours as requested
    df_extreme = df_extreme[df_extreme["lead_time"] <= 240]
    df_normal = df_normal[df_normal["lead_time"] <= 240]

    def filter_common_targets(df):
        aurora_targets = df[df['model'] == 'Aurora V1.5'][['valid_time', 'case_id_number']].drop_duplicates()
        hres_targets = df[df['model'] == 'HRES'][['valid_time', 'case_id_number']].drop_duplicates()
        common_targets = pd.merge(aurora_targets, hres_targets, on=['valid_time', 'case_id_number'], how='inner')
        return pd.merge(df, common_targets, on=['valid_time', 'case_id_number'], how='inner')

    # Apply the fair-comparison filter
    df_extreme = filter_common_targets(df_extreme)
    df_normal = filter_common_targets(df_normal)

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
    
    # Plot 3: Regular Aurora vs Extreme Aurora RMSE
    aurora_rmse = rmse_merged[rmse_merged["model"] == "Aurora V1.5"]
    if not aurora_rmse.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=pd.melt(
                aurora_rmse,
                id_vars=["lead_time"],
                value_vars=["value_normal", "value_extreme"],
                var_name="regime",
                value_name="rmse"
            ).replace({"value_normal": "Normal Weather", "value_extreme": "Extreme (Heatwave)"}),
            x="lead_time", y="rmse", hue="regime", marker="o", linewidth=2
        )
        plt.title("Aurora V1.5 RMSE: Normal vs Extreme Weather", fontsize=14)
        plt.xlabel("Lead Time (Hours)", fontsize=12)
        plt.ylabel("RMSE", fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/aurora_rmse_normal_vs_extreme.png", dpi=300)
        plt.close()

    # Plot 4: Aurora RMSE compared to HRES RMSE (Normal vs Extreme)
    # Pivot to calculate difference (Aurora - HRES)
    try:
        rmse_extreme_pivot = rmse_extreme.pivot(index="lead_time", columns="model", values="value").reset_index()
        rmse_normal_pivot = rmse_normal.pivot(index="lead_time", columns="model", values="value").reset_index()
        
        if "Aurora V1.5" in rmse_extreme_pivot.columns and "HRES" in rmse_extreme_pivot.columns and \
           "Aurora V1.5" in rmse_normal_pivot.columns and "HRES" in rmse_normal_pivot.columns:
            
            rmse_extreme_pivot["diff"] = rmse_extreme_pivot["Aurora V1.5"] - rmse_extreme_pivot["HRES"]
            rmse_normal_pivot["diff"] = rmse_normal_pivot["Aurora V1.5"] - rmse_normal_pivot["HRES"]
            
            diff_df = pd.DataFrame({
                "lead_time": rmse_normal_pivot["lead_time"],
                "Normal Weather": rmse_normal_pivot["diff"],
                "Extreme (Heatwave)": rmse_extreme_pivot["diff"]
            })
            
            diff_melt = pd.melt(diff_df, id_vars=["lead_time"], value_vars=["Normal Weather", "Extreme (Heatwave)"], 
                                var_name="regime", value_name="rmse_diff")
            
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=diff_melt, x="lead_time", y="rmse_diff", hue="regime", marker="o", linewidth=2)
            plt.axhline(0, color="black", linestyle="--", alpha=0.6, label="Equal Performance")
            plt.title("Aurora V1.5 vs HRES RMSE Difference\nNegative values = Aurora is better than HRES", fontsize=14)
            plt.xlabel("Lead Time (Hours)", fontsize=12)
            plt.ylabel("RMSE Difference (Aurora - HRES)", fontsize=12)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/aurora_vs_hres_rmse_diff.png", dpi=300)
            plt.close()
    except Exception as e:
        logger.warning(f"Could not generate Aurora vs HRES comparison plot: {e}")

    # Plot 5: Percentage Degradation in RMSE (Extreme vs Normal)
    # Formula: ((Extreme RMSE - Normal RMSE) / Normal RMSE) * 100
    rmse_merged["rmse_degradation_pct"] = ((rmse_merged["value_extreme"] - rmse_merged["value_normal"]) / rmse_merged["value_normal"]) * 100
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=rmse_merged, x="lead_time", y="rmse_degradation_pct", hue="model", alpha=0.85)
    plt.axhline(0, color="black", linestyle="-", linewidth=1.5)
    plt.title("RMSE Percentage Degradation (Extreme vs Normal Weather)\nHow much higher is the error during heatwaves?", fontsize=14)
    plt.xlabel("Lead Time (Hours)", fontsize=12)
    plt.ylabel("RMSE Degradation (%)", fontsize=12)
    
    from matplotlib.ticker import PercentFormatter
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=100))
    
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rmse_percentage_degradation.png", dpi=300)
    plt.close()

    logger.info(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    compare_performance()
