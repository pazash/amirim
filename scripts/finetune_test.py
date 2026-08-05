import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from peft import LoraConfig, get_peft_model

# Import your existing classes
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from aurora import AuroraV1p5
from src.dataloader import AuroraDataLoader
from src.forecaster import AuroraForecaster

def evaluate_batch(pred_batch, true_batch, var_name='2t', is_surface=True):
    """
    Evaluates Aurora Batch objects safely handling dimensional mismatches.
    """
    pred_data = pred_batch.surf_vars if is_surface else pred_batch.atmos_vars
    true_data = true_batch.surf_vars if is_surface else true_batch.atmos_vars
    
    pred_tensor = pred_data[var_name].detach().cpu().numpy()
    true_tensor = true_data[var_name].detach().cpu().numpy()
    
    # --- ALIGNMENT FIX ---
    min_lat = min(pred_tensor.shape[-2], true_tensor.shape[-2])
    min_lon = min(pred_tensor.shape[-1], true_tensor.shape[-1])
    
    pred_val = pred_tensor[..., :min_lat, :min_lon].flatten()
    true_val = true_tensor[..., :min_lat, :min_lon].flatten()
    # ---------------------
    
    diff = pred_val - true_val
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    bias = np.mean(diff)
    
    return {
        "RMSE": rmse, 
        "Mean Bias": bias
    }

if __name__ == "__main__":
    
    # 1. Setup Device & Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_pipeline = AuroraDataLoader(cache_dir=Path("weather_data"))
    
    t0_dt = datetime(2023, 1, 1, 6) 
    europe_bbox = {
        "lat_max": 65,
        "lat_min": 29.25,
        "lon_min": -10,
        "lon_max": 40.75
    }

    input_batch, true_target = data_pipeline.get_batches(
        t0_dt, 
        history_steps=1, 
        forecast_steps=1, 
        bbox=europe_bbox  # Pass the bounding box here!
    )
    
    # 2. Load Base Model
    print("\n--- LOADING BASE MODEL ---")
    aurora_model = AuroraV1p5(autocast=True)
    aurora_model.load_checkpoint("microsoft/aurora", "aurora-0.25-v1.5.ckpt", revision="main")
    aurora_model.to(device)
    aurora_model.configure_activation_checkpointing()
    # 3. Initial Pre-Trained Evaluation
    print("\n--- PRE-TRAINING EVALUATION ---")
    forecaster = AuroraForecaster(model=aurora_model, device=device)
    base_predictions = forecaster.predict_rollout(
        initial_batch=input_batch, steps=1, fine_lead_times=[6.0]
    )
    pre_train_metrics = evaluate_batch(base_predictions[0], true_target, var_name='2t')
    print(f"Base Model Metrics: {pre_train_metrics}")


    # 4. Apply LoRA Configuration
    print("\n--- INJECTING LORA ADAPTERS ---")

    # target_modules="all-linear" is a great trick when you don't know the exact 
    # internal layer names of a custom model. It finds all nn.Linear layers.
    lora_config = LoraConfig(
        r=4, 
        lora_alpha=8,
        target_modules="all-linear", 
        lora_dropout=0.05,
        bias="none",
    )
    
    # This freezes the base weights and adds the trainable LoRA matrices
    model_lora = get_peft_model(aurora_model, lora_config)
    model_lora.print_trainable_parameters()
    model_lora.train() # Set to training mode


    # 5. Fine-Tuning Loop (Overfitting a single sample)
    print("\n--- STARTING FINE-TUNING ---")
    optimizer = torch.optim.AdamW(model_lora.parameters(), lr=1e-4)
    epochs = 15
    
    # Move batches to device for training
    train_input = input_batch.to(device)
    train_target = true_target.to(device)
    torch.cuda.empty_cache()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass (Predict t+6 directly)
        # We don't use rollout() here because we need the computation graph for gradients
        batch_size = train_input.surf_vars['2t'].shape[0]
        lead_times_tensor = torch.full((batch_size,), 6.0, device=device)

        # Pass it to the forward call
        pred_batch = model_lora(train_input, lead_times=lead_times_tensor)
        
        # Extract the specific variable we want to train on (2m Temperature)
        pred_2t = pred_batch.surf_vars['2t']
        true_2t = train_target.surf_vars['2t']
        
        # Align spatial dimensions (Crop the South Pole if needed)
        min_lat = min(pred_2t.shape[-2], true_2t.shape[-2])
        min_lon = min(pred_2t.shape[-1], true_2t.shape[-1])
        
        pred_2t_cropped = pred_2t[..., :min_lat, :min_lon]
        true_2t_cropped = true_2t[..., :min_lat, :min_lon]
        
        # Calculate MSE Loss
        loss = F.mse_loss(pred_2t_cropped, true_2t_cropped)
        
        # Backward pass & Optimize
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss (MSE): {loss.item():.4f}")


    # 6. Post-Training Evaluation
    print("\n--- POST-TRAINING EVALUATION ---")
    # Wrap the LoRA model back into our forecaster class
    model_lora.eval()
    forecaster_lora = AuroraForecaster(model=model_lora, device=device)
    
    post_predictions = forecaster_lora.predict_rollout(
        initial_batch=input_batch, steps=1, fine_lead_times=[6.0]
    )
    post_train_metrics = evaluate_batch(post_predictions[0], true_target, var_name='2t')
    
    print(f"Base Model Metrics:   {pre_train_metrics}")
    print(f"LoRA Tuned Metrics:   {post_train_metrics}")