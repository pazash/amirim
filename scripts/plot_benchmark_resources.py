import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_plots():
    # Define paths
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "benchmark_aurora_resources.csv"
    plots_dir = base_dir / "plots" / "benchmark_resources"
    
    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}")
        return

    # Ensure plots directory exists
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Set seaborn style for better aesthetics
    sns.set_theme(style="whitegrid", context="talk")
    
    # List of metrics to plot
    metrics = [
        {
            "column": "wall_clock_sec",
            "title": "Wall Clock Time vs Lead Time",
            "ylabel": "Time (seconds)",
            "filename": "wall_clock_time.png"
        },
        {
            "column": "gpu_peak_alloc_MiB",
            "title": "GPU Peak Allocation vs Lead Time",
            "ylabel": "GPU Memory (MiB)",
            "filename": "gpu_peak_allocation.png"
        },
        {
            "column": "ram_peak_estimate_MiB",
            "title": "CPU RAM Peak Estimate vs Lead Time",
            "ylabel": "CPU RAM (MiB)",
            "filename": "cpu_ram_peak.png"
        }
    ]
    
    for metric in metrics:
        if metric["column"] not in df.columns:
            print(f"Skipping {metric['column']}, not found in CSV.")
            continue
            
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(
            data=df, 
            x="lead_time_h", 
            y=metric["column"], 
            hue="scope", 
            marker="o", 
            linewidth=2.5, 
            markersize=8
        )
        
        plt.title(metric["title"], pad=15)
        plt.xlabel("Lead Time (hours)")
        plt.ylabel(metric["ylabel"])
        plt.xticks(df["lead_time_h"].unique())
        
        # Move legend outside the plot
        plt.legend(title="Scope", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        save_path = plots_dir / metric["filename"]
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {save_path}")

    # Create a bar chart showing relative efficiency to global at max lead time (e.g., 120h)
    max_lt = df["lead_time_h"].max()
    df_max_lt = df[df["lead_time_h"] == max_lt]
    
    if not df_max_lt.empty and "Global" in df_max_lt["scope"].values:
        global_time = df_max_lt[df_max_lt["scope"] == "Global"]["wall_clock_sec"].values[0]
        
        df_max_lt = df_max_lt.copy()
        df_max_lt["speedup"] = global_time / df_max_lt["wall_clock_sec"]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df_max_lt.sort_values("speedup", ascending=False),
            x="scope",
            y="speedup",
            hue="scope",
            palette="viridis",
            dodge=False,
            legend=False
        )
        
        plt.title(f"Speedup Relative to Global Scope (Lead Time: {max_lt}h)", pad=15)
        plt.xlabel("Scope")
        plt.ylabel("Speedup Multiplier (x)")
        
        for i, val in enumerate(df_max_lt.sort_values("speedup", ascending=False)["speedup"]):
            plt.text(i, val + 0.5, f"{val:.1f}x", ha='center', va='bottom', fontweight='bold')
            
        plt.tight_layout()
        save_path = plots_dir / "speedup_vs_global.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {save_path}")
        
if __name__ == "__main__":
    create_plots()
