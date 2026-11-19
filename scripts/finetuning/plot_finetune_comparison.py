import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_finetune_comparison_plots(csv_path: str = "finetune_comparison_rmse.csv", output_dir: str = "plots/finetune_comparison"):
    """
    Reads the finetune comparison RMSE data and generates plots showing how
    the finetuned model compares to the pretrained model on different scopes.
    """
    csv_file = Path(csv_path)
    out_path = Path(output_dir)
    
    if not csv_file.exists():
        logger.error(f"Data file {csv_path} not found. Please run the evaluation script first.")
        return
        
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # 1. Line plot: Average RMSE vs Lead Time for each model
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
    plt.title("Average RMSE vs Lead Time: Pretrained vs Finetuned", fontsize=16, pad=15)
    plt.xlabel("Lead Time (Hours)", fontsize=12)
    plt.ylabel("RMSE (2m Temperature, K)", fontsize=12)
    plt.xticks([6, 24, 72, 120])
    plt.legend(title="Model & Scope", title_fontsize='11', fontsize='10')
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
    plt.title("Average RMSE by Model and Lead Time", fontsize=16, pad=15)
    plt.xlabel("Lead Time (Hours)", fontsize=12)
    plt.ylabel("RMSE (2m Temperature, K)", fontsize=12)
    plt.legend(title="Model & Scope", title_fontsize='11', fontsize='10')
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
    plt.title("RMSE Distribution by Model and Lead Time", fontsize=16, pad=15)
    plt.xlabel("Lead Time (Hours)", fontsize=12)
    plt.ylabel("RMSE (2m Temperature, K)", fontsize=12)
    plt.legend(title="Model & Scope", title_fontsize='11', fontsize='10')
    plt.tight_layout()
    plt.savefig(out_path / "rmse_distribution_boxplot.png", dpi=300)
    plt.close()

    # 4. Line plot: Average RMSE vs Lead Time for Global models only
    df_global = df[df['model'].str.contains('Global', case=False, na=False)]
    if not df_global.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df_global, 
            x="lead_time_hours", 
            y="value", 
            hue="model", 
            marker="o",
            linewidth=2.5,
            markersize=8,
            errorbar=('ci', 95),
            palette="viridis"
        )
        plt.title("Average RMSE vs Lead Time (Global Scope Only)", fontsize=16, pad=15)
        plt.xlabel("Lead Time (Hours)", fontsize=12)
        plt.ylabel("RMSE (2m Temperature, K)", fontsize=12)
        plt.xticks([6, 24, 72, 120])
        plt.legend(title="Model", title_fontsize='11', fontsize='10')
        plt.tight_layout()
        plt.savefig(out_path / "avg_rmse_vs_lead_time_line_global_only.png", dpi=300)
        plt.close()

    logger.info(f"Plots successfully generated and saved to {out_path.absolute()}")

if __name__ == "__main__":
    generate_finetune_comparison_plots()
