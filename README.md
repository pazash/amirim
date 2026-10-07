# Amirim — Fine-Tuning Aurora for Regional Weather Prediction

> **Bachelor's project in Computer Science & Earth Science**

## Motivation

Earth foundation models like [Aurora](https://github.com/microsoft/aurora) (Microsoft, *Nature* 2025) achieve impressive global forecast skill, but their accuracy degrades when evaluated on specific regions of interest. This project investigates whether **LoRA fine-tuning** on region-specific ERA5 data can improve Aurora's predictions within a focused domain — the **Eastern Mediterranean and Middle East** — without sacrificing performance elsewhere.

### Why Regional?

Global training objectives average gradients over the entire sphere, under-representing smaller regions with distinctive meteorology. By fine-tuning on a regional crop, we hypothesize the model can better capture local dynamics (e.g., Mediterranean cyclogenesis, Sharav heatwaves) while keeping the base model weights frozen. Additionally, generating predictions and fine-tuning on a regional bounding box is vastly more computationally efficient, requiring significantly less CPU RAM, GPU memory, and wall-clock time compared to global inference.

### Extreme Weather — Side Investigation

An initial direction of this project explored whether Aurora's base model struggles with **extreme weather events** (heatwaves, heavy precipitation). Preliminary results showed that the pretrained Aurora base model *already outperforms* the HRES physical model on several heatwave metrics — an unexpected finding that shifted focus toward the regional fine-tuning approach. The extreme-weather evaluation scripts and documentation remain in the repository under `scripts/extreme_scripts/` and `docs/` for reference.

---

## Project Structure

```
amirim/
├── src/                          # Core reusable modules
│   ├── dataloader.py             # ERA5 data acquisition and Aurora batch formatting
│   └── forecaster.py             # Model forward passes and autoregressive rollouts
│
├── scripts/                      # Executable experiment scripts
│   ├── finetune_aurora_6h.py     # LoRA fine-tuning pipeline
│   ├── evaluate_pretrained_vs_finetuned.py
│   ├── compare_global_vs_med_aurora.py
│   ├── compute_forecast_skill.py
│   ├── benchmark_aurora_resources.py
│   ├── plot_context_comparison.py
│   ├── plot_finetune_comparison.py
│   ├── plot_benchmark_resources.py
│   ├── plot_scope_temperature_maps.py
│   └── extreme_scripts/          # Extreme weather evaluation (side investigation)
│
├── docs/                         # Project documentation
│   ├── ai_project_context.md     # AI assistant context for this repo
│   ├── extreme_evaluation_metrics.md
│   └── extreme_precipitation_cases.md
│
├── weather_data/                 # Cached ERA5 NetCDF files (gitignored)
└── plots/                        # Generated visualizations
```

---

## Core Modules (`src/`)

### `dataloader.py` — `AuroraDataLoader`

Handles the full data pipeline from raw ERA5 to model-ready PyTorch batches:

- **Downloads** surface and pressure-level ERA5 data from the Copernicus CDS API.
- **Caches** results as NetCDF files to avoid redundant API calls.
- **Supports bounding boxes** for regional crops (with proper cache isolation).
- **Bulk download mode** — fetches entire months in single API calls for efficient training (2 calls/month vs ~120 per-sample calls).
- **Formats** data into Aurora-compatible `Batch` objects, including variable name mapping, static variable slicing, and insolation calculation.

### `forecaster.py` — `AuroraForecaster`

Thin wrapper around the Aurora model that manages:

- Device placement (GPU/CPU auto-detection).
- Single forward passes (`predict_step`).
- Multi-step autoregressive rollouts (`predict_rollout`) with configurable sub-step resolution.

---

## Scripts Reference

### Training & Evaluation

| Script | Summary |
|---|---|
| **`finetune_aurora_6h.py`** | Full LoRA fine-tuning pipeline for `AuroraPretrained`. Trains on 5 years of regional ERA5 data (2016–2020) with latitude-weighted MAE loss, cosine LR schedule with warmup, gradient accumulation, and stratified monthly validation. Saves best/epoch/final LoRA weights. |
| **`evaluate_pretrained_vs_finetuned.py`** | Compares the pretrained Aurora model against the LoRA-finetuned version. Generates forecasts for both on 10 diverse test dates across Original and Global scopes, then computes RMSE against ERA5 ground truth on the Mediterranean evaluation box. |
| **`compare_global_vs_med_aurora.py`** | Studies how the **spatial context** (input bounding box size) affects Aurora's prediction accuracy. Runs Aurora on 4 scopes — Original, Enlarged Small, Enlarged Large, Global — centered on the Mediterranean, and evaluates all on the same region. |
| **`compute_forecast_skill.py`** | Computes forecast skill scores for the context-comparison experiment against two baselines: persistence (tomorrow = today) and sample climatology. Reports Skill Score = 1 − MSE_model/MSE_baseline. |
| **`benchmark_aurora_resources.py`** | Benchmarks CPU RAM, GPU memory, and wall-clock time for Aurora rollouts across all 4 prediction scopes. Quantifies the computational savings of regional vs. global inference. |

### Visualization

| Script | Summary |
|---|---|
| **`plot_context_comparison.py`** | Generates line, bar, and box plots of RMSE across lead times for each spatial context scope. |
| **`plot_finetune_comparison.py`** | Generates comparative plots (line, bar, box) of RMSE between pretrained and finetuned models, including a Global-only view. |
| **`plot_benchmark_resources.py`** | Plots wall-clock time, GPU memory, and CPU RAM vs. lead time for each scope, plus speedup bar charts relative to Global. |
| **`plot_scope_temperature_maps.py`** | Creates side-by-side geographic maps (Cartopy) of 2m temperature forecasts from each scope vs. ERA5 ground truth, with shared colour scales. |

### Extreme Weather Scripts (`extreme_scripts/`)

| Script | Summary |
|---|---|
| **`extreme_evaluator.py`** | Early-stage prototype that generates Aurora forecasts for EWB heatwave cases and evaluates them using ExtremeWeatherBench's framework with custom IoU and Peak Bias metrics. |
| **`heatwave_evaluator.py`** | Production heatwave evaluator. Directly computes 5 metrics (PAE, CBE, RMSE, IoU, EAR) per lead time for Aurora and HRES, loading ERA5 and HRES from cloud Zarr stores. Outputs to `heatwave_evaluations.csv`. |
| **`normal_weather_evaluator.py`** | Generates and evaluates Aurora forecasts for **non-extreme** dates matching each heatwave case (same calendar window, different years), providing a normal-weather baseline for comparison. |
| **`extreme_precip_evaluator.py`** | Evaluates Aurora on 10 curated extreme precipitation events (Hurricane Harvey, European Floods, etc.), comparing against ERA5 and HRES on precipitation-related metrics. |
| **`compare_extreme_vs_normal.py`** | Compares Aurora and HRES performance metrics between heatwave and normal-weather regimes. Produces RMSE ratio, peak error, and degradation percentage plots. |
| **`plot_heatwave_eval.py`** | Generates metric degradation curves, box plots, and spatial overlap scatter plots from `heatwave_evaluations.csv`. |
| **`plot_heatwave_maps.py`** | Creates geographic temperature maps (Aurora vs. HRES vs. ERA5) for individual heatwave cases. |
| **`plot_precip_eval.py`** | Same visualization suite as `plot_heatwave_eval.py` but for precipitation evaluation results. |
| **`plot_precip_maps.py`** | Creates geographic precipitation maps (Aurora vs. HRES vs. ERA5) for individual extreme precipitation cases. |

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended; CPU works but is slow)
- [CDS API key](https://cds.climate.copernicus.eu/how-to-api) configured in `~/.cdsapirc`

### Installation

```bash
pip install torch xarray cdsapi h5netcdf pandas numpy matplotlib seaborn cartopy
pip install microsoft-aurora extremeweatherbench peft psutil
```

### Running

All scripts should be run from the project root:

```bash
# Fine-tune the model
python scripts/finetune_aurora_6h.py

# Evaluate pretrained vs finetuned
python scripts/evaluate_pretrained_vs_finetuned.py

# Compare spatial contexts
python scripts/compare_global_vs_med_aurora.py

# Generate plots
python scripts/plot_finetune_comparison.py
```

---

## Key Technical Details

- **Aurora variable naming**: Aurora uses ECMWF shortNames (`2t`, `10u`, `msl`, etc.) instead of CDS parameter names (`2m_temperature`, etc.). See `dataloader.py` for mapping dictionaries.
- **Grid constraints**: Aurora's patch-based architecture requires spatial dimensions to be multiples of 16. All bounding boxes are designed to satisfy this.
- **Longitude convention**: Aurora requires longitudes in [0, 360) and strictly increasing. The Mediterranean evaluation box is shifted east to avoid crossing the prime meridian.
- **LoRA fine-tuning**: Uses Aurora's native LoRA implementation (`use_lora=True`) rather than the external `peft` library. Only `lora_*` parameters are trainable (~0.1% of total).
