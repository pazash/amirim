import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Setup plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_clean_data(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter to only show up to 240 hours (HRES maximum)
    df = df[df['lead_time'] <= 240]
    
    # Group all aurora models into a single "Aurora" source for clean comparison
    df['model_group'] = df['forecast_source'].apply(lambda x: 'Aurora' if 'aurora' in x.lower() else x)
    
    # Only keep (valid_time, case_id_number) pairs that have both Aurora and HRES data
    aurora_targets = df[df['model_group'] == 'Aurora'][['valid_time', 'case_id_number']].drop_duplicates()
    hres_targets = df[df['model_group'] == 'HRES'][['valid_time', 'case_id_number']].drop_duplicates()
    
    # Inner join to find the exact intersection of targets evaluated by both models
    common_targets = pd.merge(aurora_targets, hres_targets, on=['valid_time', 'case_id_number'], how='inner')
    
    # Filter original df to only include these common targets
    df = pd.merge(df, common_targets, on=['valid_time', 'case_id_number'], how='inner')
    
    print(f"Filtered down to {len(common_targets)} common evaluation targets.")
    return df

def plot_degradation_over_lead_time(df, output_dir):
    """Plot 1: Line plots for all metrics showing degradation over lead time."""
    print("Generating Lead Time Degradation plots...")
    metrics = df['metric'].unique()
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        metric_df = df[df['metric'] == metric]
        
        # Lineplot automatically calculates mean and 95% CI
        sns.lineplot(
            data=metric_df, 
            x='lead_time', 
            y='value', 
            hue='model_group', 
            marker='o',
            err_style="band"
        )
        
        plt.title(f'Forecast Degradation Over Lead Time: {metric.replace("_", " ")}')
        plt.xlabel('Lead Time (Hours)')
        plt.ylabel(f'{metric.replace("_", " ")} Value')
        
        # Add a baseline for perfect scores if applicable
        if metric == 'Spatial_IOU' or metric == 'Extreme_Area_Ratio':
            plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Score (1.0)')
        elif metric == 'Peak_Amplitude_Error' or metric == 'Conditional_Bias_Extremes' or metric == 'RootMeanSquaredError':
            plt.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Perfect Score (0.0)')
            
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(output_dir / f'degradation_{metric}.png', dpi=300)
        plt.close()

def plot_focused_peak_bias_and_iou(df, output_dir):
    """Plot 4: Focused comparison between HRES and Aurora on Peak Bias and IOU."""
    print("Generating Focused Peak Bias and IOU plot...")
    
    focused_metrics = ['Peak_Amplitude_Error', 'Spatial_IOU']
    focused_df = df[df['metric'].isin(focused_metrics)]
    
    # Create a 1x2 subplot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for i, metric in enumerate(focused_metrics):
        metric_df = df[df['metric'] == metric]
        sns.lineplot(
            ax=axes[i],
            data=metric_df, 
            x='lead_time', 
            y='value', 
            hue='model_group', 
            marker='o',
            err_style="band"
        )
        axes[i].set_title(f'{metric.replace("_", " ")} vs Lead Time')
        axes[i].set_xlabel('Lead Time (Hours)')
        axes[i].set_ylabel(metric.replace("_", " "))
        
        if metric == 'Spatial_IOU':
            axes[i].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect Score')
        elif metric == 'Peak_Amplitude_Error':
            axes[i].axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label='Perfect Score')
            
    plt.tight_layout()
    plt.savefig(output_dir / 'focused_peak_bias_iou.png', dpi=300)
    plt.close()

def plot_error_distributions(df, output_dir):
    """Plot 2: Box plots for error distributions grouped by lead time buckets."""
    print("Generating Error Distribution Boxplots...")
    metrics = df['metric'].unique()
    
    # Create lead time bins (e.g. 0-72h, 72-144h, 144-240h+)
    bins = [0, 72, 144, 240, 1000]
    labels = ['0-3 Days', '3-6 Days', '6-10 Days', '10+ Days']
    df['lead_time_bucket'] = pd.cut(df['lead_time'], bins=bins, labels=labels, right=False)
    
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        metric_df = df[df['metric'] == metric]
        
        sns.boxplot(
            data=metric_df, 
            x='lead_time_bucket', 
            y='value', 
            hue='model_group',
            palette="Set2"
        )
        
        plt.title(f'Error Distribution Across Lead Times: {metric.replace("_", " ")}')
        plt.xlabel('Lead Time Window')
        plt.ylabel(f'{metric.replace("_", " ")} Value')
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(output_dir / f'distribution_{metric}.png', dpi=300)
        plt.close()

def plot_spatial_overlap_vs_bias(df, output_dir):
    """Plot 3: Scatter plot for Spatial Overlap vs Area Bias."""
    print("Generating Spatial Overlap vs Area Bias Scatter Plot...")
    
    # We need to pivot the data so we have IOU and Area Ratio in the same row
    # Grouping by valid_time, init_time, case, and model_group
    pivot_df = df.pivot_table(
        index=['case_id_number', 'valid_time', 'model_group', 'lead_time'],
        columns='metric',
        values='value'
    ).reset_index()
    
    if 'Spatial_IOU' in pivot_df.columns and 'Extreme_Area_Ratio' in pivot_df.columns:
        plt.figure(figsize=(10, 8))
        
        # Color by model, size by lead time
        sns.scatterplot(
            data=pivot_df, 
            x='Extreme_Area_Ratio', 
            y='Spatial_IOU', 
            hue='model_group',
            size='lead_time',
            sizes=(20, 200),
            alpha=0.7
        )
        
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.3, label='Perfect IOU (1.0)')
        plt.axvline(x=1.0, color='red', linestyle=':', alpha=0.3, label='Perfect Area Bias (1.0)')
        
        plt.title('Spatial Overlap (IOU) vs Extreme Area Bias')
        plt.xlabel('Extreme Area Ratio (Forecast Area / Target Area)')
        plt.ylabel('Spatial IOU')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(output_dir / 'spatial_overlap_vs_bias.png', dpi=300)
        plt.close()
    else:
        print("Skipping Spatial Overlap plot (metrics missing).")

def main():
    csv_path = Path("precip_evaluations.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path.resolve()} not found. Please run the evaluation script first.")
        return
        
    output_dir = Path("plots") / "precip"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_and_clean_data(csv_path)
    
    # Generate all plots
    plot_degradation_over_lead_time(df, output_dir)
    plot_focused_peak_bias_and_iou(df, output_dir)
    plot_error_distributions(df, output_dir)
    plot_spatial_overlap_vs_bias(df, output_dir)
    
    print(f"\nAll plots successfully generated in {output_dir.resolve()}")

if __name__ == "__main__":
    main()
