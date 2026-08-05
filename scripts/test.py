import urllib.request
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.dataloader import AuroraDataLoader
from src.forecaster import AuroraForecaster
from aurora import AuroraV1p5


# def add_world_coastlines(ax, extent):
#     """
#     A pure-Python alternative to Cartopy. Fetches a lightweight GeoJSON of 
#     the world's coastlines and plots them as standard Matplotlib lines.
#     """
#     url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_coastline.geojson"
#     try:
#         req = urllib.request.urlopen(url)
#         data = json.loads(req.read())
        
#         for feature in data['features']:
#             geom = feature['geometry']
#             lines = [geom['coordinates']] if geom['type'] == 'LineString' else geom['coordinates']
            
#             for coords in lines:
#                 lons, lats = zip(*coords)
#                 lons = np.array(lons)
#                 lats = np.array(lats)
                
#                 ax.plot(lons, lats, color='black', alpha=0.4, linewidth=0.8)
#                 ax.plot(lons + 360, lats, color='black', alpha=0.4, linewidth=0.8)
                
#         ax.set_xlim(extent[0], extent[1])
#         ax.set_ylim(extent[2], extent[3])
        
#     except Exception as e:
#         print(f"Could not load coastlines: {e}")


# def plot_temperature_comparison(predicted_batch, true_batch, title, batch_idx=0):
#     """
#     Extracts the 2m temperature ('2t') from Aurora Batch objects, converts 
#     to Celsius, adds light world map outlines, and plots it.
#     """
#     # 1. Extract the '2t' tensors and squeeze to a 2D grid
#     pred_t2m_k = predicted_batch.surf_vars["2t"][batch_idx].detach().cpu().numpy().squeeze()
#     true_t2m_k = true_batch.surf_vars["2t"][batch_idx].detach().cpu().numpy().squeeze()
    
#     # 2. Convert from Kelvin to Celsius
#     pred_t2m_c = pred_t2m_k - 273.15
#     true_t2m_c = true_t2m_k - 273.15
    
#     # 3. Calculate the difference (Forecast Error)
#     diff_t2m = pred_t2m_c - true_t2m_c

#     # 4. Dynamically extract the bounding box from the batch metadata
#     lats = predicted_batch.metadata.lat.detach().cpu().numpy()
#     lons = predicted_batch.metadata.lon.detach().cpu().numpy()
    
#     lat_min, lat_max = np.min(lats), np.max(lats)
#     lon_min, lon_max = np.min(lons), np.max(lons)
#     extent = [lon_min, lon_max, lat_min, lat_max]

#     # Find a reasonable global min/max for the colormap across both arrays
#     vmin = min(np.min(pred_t2m_c), np.min(true_t2m_c))
#     vmax = max(np.max(pred_t2m_c), np.max(true_t2m_c))

#     # 5. Set up the standard Matplotlib plot
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
#     # --- Map 1: Aurora Prediction ---
#     im0 = axes[0].imshow(pred_t2m_c, cmap="viridis", origin="upper", vmin=vmin, vmax=vmax, extent=extent)
#     add_world_coastlines(axes[0], extent)
#     axes[0].set_title("Aurora Prediction (2m Temp)")
#     fig.colorbar(im0, ax=axes[0], orientation="horizontal", pad=0.08, label="Temperature (°C)")
    
#     # --- Map 2: True ERA5 ---
#     im1 = axes[1].imshow(true_t2m_c, cmap="viridis", origin="upper", vmin=vmin, vmax=vmax, extent=extent)
#     add_world_coastlines(axes[1], extent)
#     axes[1].set_title("True ERA5 (2m Temp)")
#     fig.colorbar(im1, ax=axes[1], orientation="horizontal", pad=0.08, label="Temperature (°C)")
    
#     # --- Map 3: The Difference (Predicted - True) ---
#     max_err = max(abs(np.min(diff_t2m)), abs(np.max(diff_t2m)))
#     im2 = axes[2].imshow(diff_t2m, cmap="RdBu_r", origin="upper", vmin=-max_err, vmax=max_err, extent=extent)
#     add_world_coastlines(axes[2], extent)
#     axes[2].set_title("Difference (Aurora - ERA5)")
#     fig.colorbar(im2, ax=axes[2], orientation="horizontal", pad=0.08, label="Error (°C)")
    
#     for ax in axes:
#         ax.set_xticks([])
#         ax.set_yticks([])
        
#     plt.tight_layout()
#     plt.savefig(title)


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

    data_pipeline = AuroraDataLoader(cache_dir=Path("weather_data"))
    
    print("Loading Aurora 1.5 Model...")
    aurora_model = AuroraV1p5()
    aurora_model.load_checkpoint("microsoft/aurora", "aurora-0.25-v1.5.ckpt", revision="main")
    
    # Device is dynamically selected if not provided explicitly
    forecaster = AuroraForecaster(model=aurora_model)

    t0_dt = datetime(2023, 1, 1, 6) 
    
    # Generate 1 single step (which calculates from t-6 to t+6)
    print(f"Fetching data for t0 = {t0_dt}...")
    input_batch, true_target = data_pipeline.get_batches(
        t0_dt, 
        history_steps=1, 
        forecast_steps=1,
        bbox=None
    )

    # Use rollout to predict exactly 1 step (returning only the 6.0 hr mark)
    print("Executing forecast rollout...")
    predictions = forecaster.predict_rollout(
        initial_batch=input_batch, 
        steps=1, 
        fine_lead_times=[6.0], 
        save_path=Path("outputs/forecast_temp.pt")
    )
    
    # Rollout returns a list, grab the single 6-hour prediction object
    predicted_batch = predictions[0]

    # Evaluate the 2m Temperature (2t)
    print("Evaluating predictions vs true targets...")
    metrics = evaluate_batch(predicted_batch, true_target, var_name='2t', is_surface=True)
    print(f"Evaluation Metrics: {metrics}")