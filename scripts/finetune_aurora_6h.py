import torch
from torch.optim import AdamW
from aurora import AuroraPretrained
from pathlib import Path
from datetime import datetime, timedelta
import random
import sys
import os

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataloader import AuroraDataLoader

# Constants
CACHE_DIR = Path(".cache/era5")
OUTPUT_DIR = Path("aurora-regional-lora-6h")
EPOCHS = 3
SAMPLES_PER_EPOCH = 200
LEARNING_RATE = 1e-4

# We use the Original bbox from the compare script
PREDICTION_SCOPES = {
    "Original": { # 176x112 Grid
        "lat_min": 25.0,
        "lat_max": 52.75, 
        "lon_min": 5.0,
        "lon_max": 48.75
    }
}
BBOX = PREDICTION_SCOPES["Original"]

def calc_mae_loss_with_lat_weights(pred_batch, target_batch):
    """
    Calculates Mean Absolute Error (L1 Loss) applying latitude weighting 
    to compensate for smaller grid cell areas near the poles.
    """
    lat = pred_batch.metadata.lat # 1D tensor of latitudes
    device = lat.device
    
    # Calculate weights: cos(lat)
    weights = torch.cos(torch.deg2rad(lat)) # shape (H,)
    # Reshape to broadcast with (Batch, Time, Level, H, W) or (Batch, Time, H, W)
    # The last two dimensions are always H, W.
    weights = weights.view(-1, 1).to(device) # shape (H, 1)
    
    loss = 0.0
    num_vars = 0
    
    # Surface variables
    for k, v_pred in pred_batch.surf_vars.items():
        v_target = target_batch.surf_vars[k].to(device)
        # Apply weighting and take the mean
        weighted_error = weights * torch.abs(v_pred - v_target)
        loss += weighted_error.mean()
        num_vars += 1
        
    # Atmospheric variables
    for k, v_pred in pred_batch.atmos_vars.items():
        v_target = target_batch.atmos_vars[k].to(device)
        weighted_error = weights * torch.abs(v_pred - v_target)
        loss += weighted_error.mean()
        num_vars += 1
        
    return loss / num_vars

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize Model with native LoRA
    print("Loading AuroraPretrained with native LoRA enabled...")
    # autocast=True enables lower precision for the backbone, saving memory
    model = AuroraPretrained(
        use_lora=True, 
        lora_steps=1,
        lora_mode="single",
        autocast=True 
    )
    
    # Load the base model weights
    print("Loading base checkpoint...")
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", revision="main")
    model.configure_activation_checkpointing()
    model = model.to(device)
    model.train() # Set to train mode
    
    # Explicitly freeze all parameters except LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # Print trainable parameters to verify LoRA
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
    
    # 2. Setup DataLoader
    dataloader = AuroraDataLoader(cache_dir=CACHE_DIR)
    
    # Generate 200 random dates in 2019-2020
    # Start: Jan 1, 2019. End: Dec 31, 2020.
    start_dt = datetime(2019, 1, 1)
    end_dt = datetime(2020, 12, 31)
    total_hours = int((end_dt - start_dt).total_seconds() / 3600)
    
    # Generate random hour offsets (multiples of 6 to match ERA5 typical forecasting intervals)
    random.seed(42) # For reproducibility
    hour_offsets = random.sample([h for h in range(0, total_hours, 6)], SAMPLES_PER_EPOCH)
    train_dates = [start_dt + timedelta(hours=h) for h in hour_offsets]
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 3. Training Loop
    print(f"Starting fine-tuning for {EPOCHS} epochs on {SAMPLES_PER_EPOCH} dates...")
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        epoch_loss = 0.0
        
        # Shuffle dates each epoch
        random.shuffle(train_dates)
        
        for i, dt in enumerate(train_dates):
            print(f"[{i+1}/{SAMPLES_PER_EPOCH}] Processing date {dt}...", end=" ", flush=True)
            
            try:
                # get_batches returns t0 batch and target batch
                # Input: [t-6h, t] -> Target: [t+6h]
                input_batch, target_batch = dataloader.get_batches(
                    t0_dt=dt, 
                    history_steps=1, 
                    forecast_steps=1, 
                    bbox=BBOX
                )
                
                input_batch = input_batch.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                pred_batch = model.forward(input_batch)
                
                # Calculate Loss
                loss = calc_mae_loss_with_lat_weights(pred_batch, target_batch)
                
                # Backward pass
                loss.backward()
                
                # Simple gradient clipping to prevent exploding gradients (common in weather models)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                loss_val = loss.item()
                epoch_loss += loss_val
                print(f"Loss: {loss_val:.4f}")
                
            except Exception as e:
                print(f"Failed. Error: {e}")
                
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / SAMPLES_PER_EPOCH:.4f}")
        
    # 4. Save the LoRA weights
    # Extract only the parameters that require grad (the LoRA parameters)
    print("\nTraining complete. Saving LoRA adapter weights...")
    lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    save_path = OUTPUT_DIR / "lora_weights.pt"
    torch.save(lora_state_dict, save_path)
    print(f"Saved LoRA weights to {save_path}")

if __name__ == "__main__":
    main()
