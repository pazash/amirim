# Amirim Earth Foundation Model Finetuning

This repository contains the codebase for a bachelor's degree project in Computer Science and Earth Science. 
The core idea is to take an Earth foundation model (Aurora 1.5) and finetune it on a dataset with extreme oversampling to evaluate its performance in predicting extreme weather events, specifically heatwaves.

## Project Structure
- `src/`: Core reusable modules.
  - `dataloader.py`: Handles downloading, caching, and formatting ERA5 weather data into Aurora-compatible PyTorch batches.
  - `forecaster.py`: Manages the model forward passes and autoregressive rollouts.
- `scripts/`: Executable scripts for training and evaluation.
  - `test.py`: A basic script to verify the dataloader and forecaster can pull data and run a single step prediction.
  - `finetune_test.py`: Sets up a LoRA (Low-Rank Adaptation) fine-tuning loop on the Aurora foundation model.
  - `extreme_evaluator.py`: Uses ExtremeWeatherBench (EWB) to systematically evaluate the model against heatwave cases.
- `docs/`: Additional documentation.

## Setup
Ensure you have the necessary dependencies installed (e.g., `torch`, `xarray`, `extremeweatherbench`, `cdsapi`, etc.).
Run the scripts from the root directory of the project:
```bash
python scripts/test.py
```
