import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_context_comparison_plots(csv_path: str = "aurora_context_comparison_rmse.csv", output_dir: str = "plots/context_comparison"):
    """
    Reads the context comparison RMSE data and generates plots showing how
    the bounding box shape (context) affects Aurora's performance.
    """
    csv_file = Path(csv_path)
    out_path = Path(output_dir)
    
    if not csv_file.exists():
        logger.error(f"Data file {csv_path} not found. Please run the evaluation script first.")
        return
        
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Clean up model names for the plots
    df["model"] = df["model"].str.replace("Aurora_", "")
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # 1. Line plot: Average RMSE vs Lead Time for each scope
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df, 
        x="lead_time_hours", 
        y="value", 
        hue="model", 
        marker="o",
        linewidth=2.5,
        markersize=8,
        errorbar=('ci', 95), # Show 95% confidence interval
        palette="viridis"
    )
    plt.title("Average RMSE vs Lead Time by Context Scope", fontsize=16, pad=15)
    plt.xlabel("Lead Time (Hours)", fontsize=12)
    plt.ylabel("RMSE (2m Temperature, K)", fontsize=12)
    plt.xticks([24, 72, 120])
    plt.legend(title="Scope", title_fontsize='11', fontsize='10')
    plt.tight_layout()
    plt.savefig(out_path / "avg_rmse_vs_lead_time_line.png", dpi=300)
    plt.close()
    
    # 2. Bar plot: Average RMSE by Model grouped by Lead Time
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df, 
        x="lead_time_hours", 
        y="value", 
        hue="model",
        palette="viridis"
    )
    plt.title("Average RMSE by Context Scope and Lead Time", fontsize=16, pad=15)
    plt.xlabel("Lead Time (Hours)", fontsize=12)
    plt.ylabel("RMSE (2m Temperature, K)", fontsize=12)
    plt.legend(title="Scope", title_fontsize='11', fontsize='10')
    plt.tight_layout()
    plt.savefig(out_path / "avg_rmse_by_lead_time_bar.png", dpi=300)
    plt.close()
    
    # 3. Box plot: Distribution of RMSE to show variance across different initializations
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df, 
        x="lead_time_hours", 
        y="value", 
        hue="model",
        palette="viridis"
    )
    plt.title("RMSE Distribution by Context Scope and Lead Time", fontsize=16, pad=15)
    plt.xlabel("Lead Time (Hours)", fontsize=12)
    plt.ylabel("RMSE (2m Temperature, K)", fontsize=12)
    plt.legend(title="Scope", title_fontsize='11', fontsize='10')
    plt.tight_layout()
    plt.savefig(out_path / "rmse_distribution_boxplot.png", dpi=300)
    plt.close()

    logger.info(f"Plots successfully generated and saved to {out_path.absolute()}")

if __name__ == "__main__":
    generate_context_comparison_plots()
