# AI Project Context: Amirim — Fine-Tuning Aurora for Regional Weather Prediction

> This document provides context for AI assistants working on this codebase.
> It describes the project's goals, architecture, data flow, and conventions.

---

## 1. Project Overview

This is a **Bachelor's degree project** in Computer Science and Earth Science.

**Primary research question**: Can LoRA fine-tuning of Aurora (Microsoft's Earth foundation model) on region-specific ERA5 reanalysis data improve forecast accuracy for the **Eastern Mediterranean and Middle East**, compared to the pretrained base model?

**Approach**:
1. Fine-tune Aurora using LoRA adapters on ERA5 data cropped to a regional bounding box.
2. Evaluate the finetuned model against the pretrained model across multiple lead times (6h–120h).
3. Measure whether regional training context improves predictions within the target region.

**Side investigation (currently deprioritized)**: An earlier direction explored whether Aurora struggles with extreme weather events (heatwaves, heavy precipitation). Initial findings showed the pretrained Aurora base model already outperforms the HRES physical model on heatwave metrics — a surprising result that redirected focus to regional fine-tuning. The extreme-weather scripts remain in the repo under `scripts/extreme_scripts/`.

---

## 2. Repository Architecture

### 2.1 `src/` — Core Modules

These are importable, reusable modules. All experiment-specific logic lives in `scripts/`.

- **`dataloader.py`** — `AuroraDataLoader`
  - Fetches ERA5 reanalysis data (surface + pressure levels) from the Copernicus CDS API.
  - Caches downloads as `.nc` (NetCDF) files with separate cache directories per scope.
  - Supports **bounding-box crops** for regional data and **global** (full-globe) data.
  - Provides two data loading modes:
    - `get_batches()` — on-demand per-sample download (for evaluation).
    - `get_batches_from_bulk()` — reads from pre-downloaded monthly bulk cache (for training).
  - `bulk_download_range()` — downloads entire months in single CDS API calls (2 calls/month vs ~120 per-sample).
  - Handles variable name mapping between CDS names (`2m_temperature`) and Aurora shortNames (`2t`).
  - Calculates static variables, insolation, and manages geospatial slicing.
  - **`pretrained_only` mode**: When `True`, downloads only the 4 surface variables needed by `AuroraPretrained` (vs. 18 for `AuroraV1p5`).

- **`forecaster.py`** — `AuroraForecaster`
  - Wraps the Aurora model for inference.
  - Handles device placement (auto GPU/CPU detection).
  - `predict_step()` — single forward pass.
  - `predict_rollout()` — multi-step autoregressive rollout using Aurora's `rollout()` utility.
  - Moves predictions to CPU after generation to free GPU memory.

### 2.2 `scripts/` — Experiment Scripts

#### Training
- **`finetune_aurora_6h.py`** — Production fine-tuning script.
  - Uses `AuroraPretrained` with native LoRA (`use_lora=True`, `lora_mode="single"`).
  - Trains on ERA5 2016–2020 (5 years, ~7,300 samples per epoch).
  - Validates on stratified dates across all 12 months of 2021.
  - Latitude-weighted MAE loss, cosine schedule with warmup, gradient accumulation (8 steps).
  - Outputs: `aurora-regional-lora-6h/lora_weights_best.pt`, epoch checkpoints, `training_log.csv`.

#### Evaluation
- **`evaluate_pretrained_vs_finetuned.py`** — Compares pretrained vs. finetuned RMSE on 10 test dates.
- **`compare_global_vs_med_aurora.py`** — Studies how input spatial context size affects accuracy.
- **`compute_forecast_skill.py`** — Computes skill scores vs. persistence and climatology baselines.
- **`benchmark_aurora_resources.py`** — Measures compute cost (RAM, GPU, time) per scope.

#### Visualization
- **`plot_context_comparison.py`**, **`plot_finetune_comparison.py`**, **`plot_benchmark_resources.py`**, **`plot_scope_temperature_maps.py`** — Generate publication-quality figures.

#### Extreme Weather (`extreme_scripts/`)
- **`extreme_evaluator.py`** — Early prototype using ExtremeWeatherBench.
- **`heatwave_evaluator.py`** — Direct heatwave evaluation against cloud ERA5/HRES Zarr stores.
- **`normal_weather_evaluator.py`** — Normal-weather baseline evaluator.
- **`extreme_precip_evaluator.py`** — Extreme precipitation evaluation on 10 curated events.
- **`compare_extreme_vs_normal.py`**, **`plot_heatwave_eval.py`**, **`plot_heatwave_maps.py`**, **`plot_precip_eval.py`**, **`plot_precip_maps.py`** — Comparison and visualization scripts.

### 2.3 `docs/` — Documentation

- **`extreme_evaluation_metrics.md`** — Scientific specification of all extreme-weather metrics (PAE, CBE, RMSE, IoU, EAR), with formulas, thresholds, and literature references.
- **`extreme_precipitation_cases.md`** — Curated catalog of 10 extreme precipitation events with meteorological justification, return periods, and selection criteria.

---

## 3. Key Concepts and Conventions

### 3.1 Bounding Boxes and Scopes

The project defines 4 **prediction scopes** — different input spatial extents — all centered on the Mediterranean:

| Scope | Grid Size | Description |
|---|---|---|
| **Original** | 176×112 | The evaluation target region (also used for fine-tuning) |
| **Enlarged_Small** | 224×176 | ~6° centered padding around Original |
| **Enlarged_Large** | 320×240 | ~18° centered padding around Original |
| **Global** | 1440×721 | Full globe (0.25° resolution) |

All grid dimensions are multiples of 16 (required by Aurora's patch-based architecture).

**Evaluation is always on the Original box**, regardless of which scope was used for prediction. This isolates the effect of spatial context.

### 3.2 Aurora Model Variants

The project uses two Aurora model variants:

| Variant | Class | Checkpoint | Surface Vars | Use Case |
|---|---|---|---|---|
| **Aurora V1.5** | `AuroraV1p5` | `aurora-0.25-v1.5.ckpt` | 18 | Context comparison, extreme weather |
| **Aurora Pretrained** | `AuroraPretrained` | `aurora-0.25-pretrained.ckpt` | 4 | Fine-tuning, finetuned evaluation |

`AuroraPretrained` accepts fewer variables but supports native LoRA. The `pretrained_only` flag in `AuroraDataLoader` controls which variable set is downloaded.

### 3.3 Variable Naming

Aurora uses ECMWF shortNames, not CDS parameter names:

| CDS Name | Aurora Name | Description |
|---|---|---|
| `2m_temperature` / `t2m` | `2t` | 2-meter temperature |
| `10m_u_component_of_wind` / `u10` | `10u` | 10m u-wind |
| `mean_sea_level_pressure` / `msl` | `msl` | Mean sea level pressure |

See `dataloader.py` → `surf_name_map` and `atmos_name_map` for the complete mapping.

### 3.4 Lead Times

Standard lead times used across evaluation scripts: **6h, 24h, 72h, 120h**.
Extreme weather scripts also evaluate out to **240h**.
Aurora steps in 6-hour increments (e.g., 120h = 20 autoregressive steps).

---

## 4. Data Flow

```
CDS API  ──►  NetCDF cache  ──►  AuroraDataLoader  ──►  Batch objects  ──►  Aurora model
              (weather_data/)      (xarray → torch)      (surf_vars,         (forward pass /
                                                          atmos_vars,          rollout)
                                                          static_vars,
                                                          metadata)
```

For fine-tuning:
```
bulk_download_range()  ──►  Monthly NetCDF files  ──►  get_batches_from_bulk()
(2 CDS calls/month)        (.cache/era5/bulk/)         (random access by t0)
```

---

## 5. Output Files

| File | Source Script | Contents |
|---|---|---|
| `aurora_context_comparison_rmse.csv` | `compare_global_vs_med_aurora.py` | RMSE per (init_time, lead_time, scope) |
| `finetune_comparison_rmse.csv` | `evaluate_pretrained_vs_finetuned.py` | RMSE per (init_time, lead_time, model, scope) |
| `benchmark_aurora_resources.csv` | `benchmark_aurora_resources.py` | Resource usage per (scope, lead_time) |
| `aurora_forecast_skill_scores.csv` | `compute_forecast_skill.py` | Skill scores vs persistence/climatology |
| `heatwave_evaluations.csv` | `heatwave_evaluator.py` | 5 metrics × Aurora + HRES × all heatwave cases |
| `normal_evaluations.csv` | `normal_weather_evaluator.py` | Same metrics for normal-weather baselines |
| `precip_evaluations.csv` | `extreme_precip_evaluator.py` | Precipitation metrics for 10 curated events |
| `aurora-regional-lora-6h/` | `finetune_aurora_6h.py` | LoRA weights, training log |

---

## 6. Guidelines for AI Assistants

1. **Keep `src/` modular** — no experiment-specific logic. New experiments go in `scripts/`.
2. **Follow the variable naming conventions** — Aurora shortNames (`2t`, not `t2m`).
3. **Respect cache isolation** — different bounding boxes must use separate cache directories to prevent data collisions. See how `compare_global_vs_med_aurora.py` creates per-scope `AuroraDataLoader` instances.
4. **Grid dimensions must be multiples of 16** — Aurora's patch-based architecture requires this. Verify when defining new bounding boxes.
5. **`sys.path` convention** — scripts append the project root to `sys.path` for imports from `src/`. Follow this pattern for new scripts.
6. **Two model variants exist** — `AuroraV1p5` (full variables) and `AuroraPretrained` (4 surface vars, supports LoRA). Don't mix them. The `pretrained_only` flag in `AuroraDataLoader` must match the model being used.
7. **Longitude convention** — Aurora requires longitudes in [0, 360) and strictly increasing. Bounding boxes are designed to avoid crossing the prime meridian.
8. **Extreme scripts are a side investigation** — they are maintained but not the active focus. The primary direction is regional fine-tuning.
