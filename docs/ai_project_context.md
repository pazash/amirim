# AI Project Context: Amirim Earth Foundation Model Finetuning

## Project Overview
This project is a Bachelor's degree project in Computer Science and Earth Science. 
The core objective is to fine-tune an Earth foundation model (specifically the Aurora 1.5 model by Microsoft) to better predict **extreme weather events**. 
The training strategy involves **extreme oversampling**, meaning the dataset used for fine-tuning will heavily over-represent extreme cases (such as severe heatwaves) compared to normal weather conditions. The final step is to rigorously evaluate whether this fine-tuned model outperforms the base model in forecasting these extreme events.

## Repository Architecture

### 1. `src/` (Core Modules)
- **`dataloader.py`**: Defines `AuroraDataLoader`. This class is responsible for fetching ERA5 reanalysis data from the Copernicus Climate Data Store (CDS). It downloads surface and atmospheric variables, handles caching to `.nc` (NetCDF) files, and formats the data into `Batch` objects that are compatible with the Aurora 1.5 model. It calculates static variables, insolation, and manages geospatial slicing (bounding boxes).
- **`forecaster.py`**: Defines `AuroraForecaster`. This wraps the Aurora foundation model. It handles moving data to the correct device (GPU/CPU), executing single forward passes (`predict_step`), and running autoregressive rollouts (`predict_rollout`) to forecast multiple steps into the future.

### 2. `scripts/` (Execution Scripts)
- **`test.py`**: A simple sanity-check script. It loads the Aurora base model, fetches a single batch of weather data, runs a single-step prediction rollout, and calculates the Root Mean Squared Error (RMSE) and Mean Bias for the 2-meter temperature (`2t`) variable.
- **`finetune_test.py`**: The fine-tuning script. It leverages LoRA (Low-Rank Adaptation) via the `peft` library to inject trainable parameters into the massive Aurora model. It freezes the base weights, sets up a training loop using PyTorch, and optimizes the model specifically for the 2m temperature variable using Mean Squared Error (MSE) loss.
- **`extreme_evaluator.py`**: The evaluation script. It uses `extremeweatherbench` to evaluate the model's performance on known extreme weather cases (specifically heatwaves). It generates forecasts across multiple lead times covering the entire duration of a heatwave and calculates metrics like Spatial Intersection Over Union (IOU) and Peak Magnitude Bias against a dynamically calculated extreme threshold (e.g., the 95th percentile).

## Key Workflows
1. **Data Acquisition**: ERA5 data is downloaded via CDS API and cached in `weather_data/`.
2. **Fine-Tuning**: LoRA adapters are trained on specific oversampled extreme cases to adjust the model's weights without modifying the base model parameters.
3. **Evaluation**: Forecasts are generated and saved to `ewb_forecasts/` as NetCDF files. These are then evaluated against standard meteorological baselines like HRES.

## Note for AI Assistants
When modifying this codebase:
- Keep the `src/` modular and pure. Any new scripts or experiments should be added to `scripts/`.
- The Aurora model operates on specific naming conventions for variables (e.g., `2t` instead of `t2m`). See `dataloader.py` for mapping dictionaries.
- Scripts in the `scripts/` directory manipulate `sys.path` to import from `src`. Ensure new scripts follow this convention or that the project is pip-installed in editable mode in the future.
