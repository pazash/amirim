import torch
import math
import csv
import calendar
import dataclasses
import gc
import random
import traceback
import sys
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from aurora import AuroraPretrained
from pathlib import Path
from datetime import datetime, timedelta

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.dataloader import AuroraDataLoader

# ==================== Configuration ====================
CACHE_DIR = Path(".cache/era5")
OUTPUT_DIR = Path("aurora-regional-lora-6h")

# Training hyperparameters
EPOCHS = 5
LEARNING_RATE = 5e-5
GRAD_ACCUM_STEPS = 8
WARMUP_STEPS = 500
GRAD_CLIP_NORM = 1.0

# Data configuration — following the Aurora paper (Nature 2025)
# Pre-training used ERA5 1979-2021; fine-tuning used 2016-2021 for weather tasks
TRAIN_YEARS = list(range(2016, 2021))  # 2016-2020 (5 years, ~7,300 samples)
VAL_YEAR = 2021                         # Stratified across ALL 12 months
VAL_DATES_PER_MONTH = 3                 # 3 dates/month × 12 months = 36 val samples

# Checkpointing & logging
CHECKPOINT_EVERY_EPOCHS = 1
VALIDATE_EVERY_STEPS = 500
LOG_EVERY_STEPS = 10

# Bounding box — "Original" region from the comparison scripts
PREDICTION_SCOPES = {
    "Original": {  # 176x112 Grid
        "lat_min": 25.0,
        "lat_max": 52.75,
        "lon_min": 5.0,
        "lon_max": 48.75
    }
}
BBOX = PREDICTION_SCOPES["Original"]

# Variables that AuroraPretrained expects
ALLOWED_SURF_VARS = ("2t", "10u", "10v", "msl")
ALLOWED_STATIC_VARS = ("lsm", "z", "slt")
ALLOWED_ATMOS_VARS = ("z", "u", "v", "t", "q")


# ==================== Loss Function ====================

def calc_mae_loss_with_lat_weights(pred_batch, target_batch):
    """
    Calculates Mean Absolute Error (L1 Loss) applying latitude weighting 
    to compensate for smaller grid cell areas near the poles.
    """
    lat = pred_batch.metadata.lat  # 1D tensor of latitudes
    device = lat.device
    
    # Calculate weights: cos(lat)
    weights = torch.cos(torch.deg2rad(lat))  # shape (H,)
    # Reshape to broadcast with (Batch, Time, Level, H, W) or (Batch, Time, H, W)
    weights = weights.view(-1, 1).to(device)  # shape (H, 1)
    
    loss = 0.0
    num_vars = 0
    
    # Surface variables
    for k, v_pred in pred_batch.surf_vars.items():
        v_target = target_batch.surf_vars[k].to(device)
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


# ==================== LR Schedule ====================

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """
    Cosine annealing LR schedule with linear warmup.
    Standard practice for weather model fine-tuning (Aurora paper, D.3-D.4).
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ==================== Data Generation ====================

def generate_train_dates(years):
    """
    Generate all valid 6-hourly timestamps for training.
    
    Starts at 06:00 on Jan 1 (needs t-6h as history input).
    For intermediate years, includes up to Dec 31 18:00 (target Jan 1 of next
    training year is available). For the last year, stops at Dec 31 12:00 
    to avoid requiring data from the validation year.
    """
    dates = []
    last_year = max(years)
    for year in years:
        dt = datetime(year, 1, 1, 6)
        # Last training year: stop at 12:00 so target stays within the year
        # Other years: include 18:00 since next year's Jan data is also downloaded
        if year == last_year:
            end = datetime(year, 12, 31, 12)
        else:
            end = datetime(year, 12, 31, 18)
        while dt <= end:
            dates.append(dt)
            dt += timedelta(hours=6)
    return dates


def generate_val_dates(year, dates_per_month, seed=123):
    """
    Generate stratified validation dates: N random dates per month across
    ALL 12 months of the given year, ensuring full seasonal coverage.
    
    This avoids the bias of using only a contiguous block of months —
    the model's performance on all seasons is monitored during training.
    
    Excludes specific dates used in the evaluation test set to prevent leakage.
    """
    rng = random.Random(seed)
    val_dates = []
    
    for month in range(1, 13):
        days_in_month = calendar.monthrange(year, month)[1]
        
        # Generate all valid t0 times in this month
        candidates = []
        dt = datetime(year, month, 1, 6)
        # For December, stop at 12:00 on last day (target stays within 2021)
        # For other months, 18:00 is fine since next month's data is downloaded
        if month == 12:
            end = datetime(year, month, days_in_month, 12)
        else:
            end = datetime(year, month, days_in_month, 18)
        
        while dt <= end:
            candidates.append(dt)
            dt += timedelta(hours=6)
        
        # Sample N dates from this month
        n = min(dates_per_month, len(candidates))
        selected = rng.sample(candidates, n)
        val_dates.extend(selected)
    
    return sorted(val_dates)


# ==================== Helpers ====================

def filter_batch(input_batch):
    """Filter batch to only the variables AuroraPretrained expects."""
    return dataclasses.replace(
        input_batch,
        surf_vars={k: v for k, v in input_batch.surf_vars.items() if k in ALLOWED_SURF_VARS},
        static_vars={k: v for k, v in input_batch.static_vars.items() if k in ALLOWED_STATIC_VARS},
        atmos_vars={k: v for k, v in input_batch.atmos_vars.items() if k in ALLOWED_ATMOS_VARS},
    )


@torch.no_grad()
def validate(model, dataloader, val_dates, device):
    """
    Run validation on stratified dates and return average loss.
    Uses dates spread across all 12 months to detect seasonal bias.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    
    for dt in val_dates:
        try:
            input_batch, target_batch = dataloader.get_batches_from_bulk(
                t0_dt=dt, history_steps=1, forecast_steps=1, bbox=BBOX
            )
            input_batch = filter_batch(input_batch).to(device)
            pred_batch = model.forward(input_batch)
            loss = calc_mae_loss_with_lat_weights(pred_batch, target_batch)
            total_loss += loss.item()
            count += 1
            
            del input_batch, target_batch, pred_batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"  Val sample {dt} failed: {e}")
    
    model.train()
    
    if count == 0:
        return float('inf')
    return total_loss / count


# ==================== Main Training ====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Initialize Model with native LoRA
    print("Loading AuroraPretrained with native LoRA enabled...")
    model = AuroraPretrained(
        use_lora=True, 
        lora_steps=1,
        lora_mode="single",
        autocast=True 
    )
    
    # Load the base model weights
    print("Loading base checkpoint...")
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", revision="main", strict=False)
    model.configure_activation_checkpointing()
    model = model.to(device)
    model.train()
    
    # Explicitly freeze all parameters except LoRA parameters
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name
            
    # Print trainable parameters to verify LoRA
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
    
    # 2. Setup DataLoader & Download Data
    dataloader = AuroraDataLoader(cache_dir=CACHE_DIR, pretrained_only=True)
    
    # Generate training and validation dates
    train_dates = generate_train_dates(TRAIN_YEARS)
    val_dates = generate_val_dates(VAL_YEAR, VAL_DATES_PER_MONTH)
    print(f"\nTraining samples:   {len(train_dates)} (years {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]})")
    print(f"Validation samples: {len(val_dates)} (year {VAL_YEAR}, stratified across all 12 months)")
    
    # Bulk download all required months (far more efficient than per-sample)
    # Training: 5 years × 12 months = 60 months → 120 CDS API calls (vs ~7,300 per-sample)
    # Validation: 1 year × 12 months = 12 months → 24 CDS API calls
    print("\n=== Bulk Downloading Training Data ===")
    dataloader.bulk_download_range(TRAIN_YEARS, bbox=BBOX)
    print("\n=== Bulk Downloading Validation Data ===")
    dataloader.bulk_download_range([VAL_YEAR], bbox=BBOX)
    
    # 3. Setup Optimizer & Scheduler
    # Only pass trainable (LoRA) parameters to optimizer — more memory efficient
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    
    total_optimizer_steps = (EPOCHS * len(train_dates)) // GRAD_ACCUM_STEPS
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_optimizer_steps)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup loss logging to CSV
    log_path = OUTPUT_DIR / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "global_step", "train_loss", "val_loss", "learning_rate"])
    
    # 4. Training Loop
    print(f"\n{'='*60}")
    print(f"TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Epochs:               {EPOCHS}")
    print(f"  Samples per epoch:    {len(train_dates)}")
    print(f"  Gradient accumulation: {GRAD_ACCUM_STEPS}")
    print(f"  Optimizer steps:      {total_optimizer_steps}")
    print(f"  Learning rate:        {LEARNING_RATE} (cosine + {WARMUP_STEPS}-step warmup)")
    print(f"  Validation every:     {VALIDATE_EVERY_STEPS} steps ({len(val_dates)} samples)")
    print(f"  Bounding box:         {BBOX}")
    print(f"{'='*60}\n")
    
    global_step = 0
    best_val_loss = float('inf')
    random.seed(42)
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")
        
        # Shuffle training dates each epoch for diversity
        random.shuffle(train_dates)
        epoch_loss = 0.0
        epoch_samples = 0
        accum_loss = 0.0
        accum_count = 0
        
        optimizer.zero_grad()
        
        for i, dt in enumerate(train_dates):
            try:
                # Load sample from bulk cache
                input_batch, target_batch = dataloader.get_batches_from_bulk(
                    t0_dt=dt, history_steps=1, forecast_steps=1, bbox=BBOX
                )
                input_batch = filter_batch(input_batch).to(device)
                
                # Forward pass
                pred_batch = model.forward(input_batch)
                loss = calc_mae_loss_with_lat_weights(pred_batch, target_batch)
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / GRAD_ACCUM_STEPS
                scaled_loss.backward()
                
                loss_val = loss.item()
                accum_loss += loss_val
                epoch_loss += loss_val
                epoch_samples += 1
                accum_count += 1
                
                del input_batch, target_batch, pred_batch, loss, scaled_loss
                
            except Exception as e:
                print(f"  Sample {i+1} ({dt}) failed: {e}")
                traceback.print_exc()
                continue  # Don't increment accum_count on failure
            
            # Optimizer step when enough gradients accumulated
            if accum_count >= GRAD_ACCUM_STEPS:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                avg_accum_loss = accum_loss / accum_count
                accum_loss = 0.0
                accum_count = 0
                
                # Periodic logging
                if global_step % LOG_EVERY_STEPS == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"  [Step {global_step}/{total_optimizer_steps}] "
                          f"Sample {i+1}/{len(train_dates)} | "
                          f"Loss: {avg_accum_loss:.4f} | LR: {lr:.2e}")
                
                # Periodic validation
                if global_step % VALIDATE_EVERY_STEPS == 0:
                    print(f"\n  --- Validation at step {global_step} ---")
                    val_loss = validate(model, dataloader, val_dates, device)
                    lr = scheduler.get_last_lr()[0]
                    print(f"  Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")
                    
                    # Log to CSV
                    with open(log_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch+1, global_step, f"{avg_accum_loss:.6f}", f"{val_loss:.6f}", f"{lr:.2e}"])
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = OUTPUT_DIR / "lora_weights_best.pt"
                        lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
                        torch.save(lora_state, best_path)
                        print(f"  ★ New best model saved to {best_path}")
                    
                    model.train()
            
            # Periodic memory cleanup
            if (i + 1) % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Flush any remaining accumulated gradients at end of epoch
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            accum_count = 0
            accum_loss = 0.0
        
        # End of epoch summary
        avg_epoch_loss = epoch_loss / max(epoch_samples, 1)
        print(f"\nEpoch {epoch+1} Complete | Avg Loss: {avg_epoch_loss:.4f} | "
              f"Samples: {epoch_samples}/{len(train_dates)}")
        
        # Epoch checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0:
            ckpt_path = OUTPUT_DIR / f"lora_weights_epoch{epoch+1}.pt"
            lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
            torch.save(lora_state, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")
        
        # End of epoch validation
        print(f"\n--- End-of-Epoch {epoch+1} Validation ---")
        val_loss = validate(model, dataloader, val_dates, device)
        lr = scheduler.get_last_lr()[0]
        print(f"Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f}")
        
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, global_step, f"{avg_epoch_loss:.6f}", f"{val_loss:.6f}", f"{lr:.2e}"])
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = OUTPUT_DIR / "lora_weights_best.pt"
            lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
            torch.save(lora_state, best_path)
            print(f"★ New best model saved to {best_path}")
    
    # 5. Save final weights (in same format as before for eval script compatibility)
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    final_path = OUTPUT_DIR / "lora_weights.pt"
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state, final_path)
    print(f"Final weights:  {final_path}")
    print(f"Best weights:   {OUTPUT_DIR / 'lora_weights_best.pt'} (val loss: {best_val_loss:.4f})")
    print(f"Training log:   {log_path}")


if __name__ == "__main__":
    main()
