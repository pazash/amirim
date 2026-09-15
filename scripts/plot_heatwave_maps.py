import sys
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path
import warnings
import extremeweatherbench as ewb

# Ignore cartopy/shapely warnings if any
warnings.filterwarnings("ignore")

# Import evaluator functions
sys.path.append(str(Path(__file__).resolve().parent))
from heatwave_evaluator import load_era5_zarr, load_hres_zarr, standardize_longitude

def main():
    base_dir = Path(__file__).resolve().parent.parent
    forecast_dir = base_dir / "ewb_forecasts"
    
    if not forecast_dir.exists():
        print(f"Error: {forecast_dir} not found.")
        print("Please ensure you have generated the predictions first.")
        return

    plots_dir = base_dir / "plots" / "heatwave_maps"
    
    print("Loading EWB case metadata...")
    all_cases = ewb.load_cases()
    hw_cases = [c for c in all_cases if c.event_type == "heat_wave"]
    if not hw_cases:
        print("No heatwave cases found in EWB dataset.")
        return
        
    # Take the first heatwave event
    case = hw_cases[0]
    case_out_dir = plots_dir / f"event_{case.case_id_number}"
    case_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading ERA5 and HRES for Case {case.case_id_number}...")
    era5_ds = load_era5_zarr()
    hres_ds = load_hres_zarr()
    
    print(f"Slicing ERA5 for event period: {case.start_date} to {case.end_date}")
    era5_case = era5_ds.sel(time=slice(case.start_date, case.end_date))
    era5_case = case.location.mask(era5_case).compute()
    era5_case = standardize_longitude(era5_case)
    
    all_nc_files = sorted(list(forecast_dir.glob("*.nc")))
    if not all_nc_files:
        print(f"No Aurora forecast files found in {forecast_dir}")
        return
        
    print(f"Scanning {len(all_nc_files)} prediction files to find overlaps for Case {case.case_id_number}...")
    nc_files = []

    for nc_file in all_nc_files:
        try:
            aurora_fcst = xr.open_dataset(nc_file)
            init_t = aurora_fcst.init_time.values[0]
            lead_times = aurora_fcst.lead_time.values
            valid_times = init_t + lead_times
            start_dt = pd.to_datetime(case.start_date)
            end_dt = pd.to_datetime(case.end_date)
            if np.any((valid_times >= start_dt) & (valid_times <= end_dt)):
                nc_files.append(nc_file)
            aurora_fcst.close()
        except Exception as e:
            pass
            
    if not nc_files:
        print(f"No overlapping Aurora forecast files found for case {case.case_id_number}.")
        return
        
    print(f"Found {len(nc_files)} prediction files overlapping with Case {case.case_id_number}.")

    for nc_file in nc_files:
        print(f"Processing {nc_file.name}...")
        aurora_fcst = xr.open_dataset(nc_file)
        init_t = aurora_fcst.init_time.values[0]
        
        aurora_fcst = case.location.mask(aurora_fcst).compute()
        aurora_fcst = standardize_longitude(aurora_fcst)
        
        if "2t_aurora" in aurora_fcst.data_vars:
            aurora_var = "2t_aurora"
        else:
            aurora_var = [v for v in aurora_fcst.data_vars if v != "latitude" and v != "longitude"][0]
            
        try:
            hres_fcst = hres_ds.sel(init_time=init_t, method="pad")
            hres_init_t = hres_fcst.init_time.values
            hres_fcst = case.location.mask(hres_fcst).compute()
            hres_fcst = standardize_longitude(hres_fcst)
            has_hres = True
        except KeyError:
            has_hres = False
            
        for lt in aurora_fcst.lead_time.values:
            valid_time = init_t + lt
            lt_hours = int(pd.Timedelta(lt).total_seconds() / 3600)
            
            # Check if valid_time is within the event period
            if pd.to_datetime(case.start_date) <= pd.to_datetime(valid_time) <= pd.to_datetime(case.end_date):
                try:
                    tgt_2d = era5_case["surface_air_temperature"].sel(time=valid_time)
                except KeyError:
                    continue
                
                # Extract Aurora
                fcst_2d_aurora = aurora_fcst[aurora_var].sel(
                    init_time=init_t, 
                    lead_time=lt
                )
                
                # Convert from Kelvin to Celsius
                aurora_val = fcst_2d_aurora.values - 273.15
                tgt_val = tgt_2d.values - 273.15
                
                vmin, vmax = 0.0, 1.0 # Fallbacks
                if not np.isnan(aurora_val).all() and not np.isnan(tgt_val).all():
                    vmin = min(np.nanmin(aurora_val), np.nanmin(tgt_val))
                    vmax = max(np.nanmax(aurora_val), np.nanmax(tgt_val))
                elif not np.isnan(aurora_val).all():
                    vmin, vmax = np.nanmin(aurora_val), np.nanmax(aurora_val)
                elif not np.isnan(tgt_val).all():
                    vmin, vmax = np.nanmin(tgt_val), np.nanmax(tgt_val)
                
                if vmin == vmax:
                    vmax = vmin + 1.0

                # Plot setup
                fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': ccrs.PlateCarree()})
                plt.subplots_adjust(wspace=0.3)
                init_str = pd.to_datetime(init_t).strftime('%Y-%m-%d %H:%M')
                valid_str = pd.to_datetime(valid_time).strftime('%Y-%m-%d %H:%M')
                fig.suptitle(f"Heatwave Case {case.case_id_number} - Init: {init_str} | Valid: {valid_str} (Lead: {lt_hours}h)", fontsize=16)

                # Aurora
                ax = axes[0]
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                im0 = ax.pcolormesh(fcst_2d_aurora.longitude, fcst_2d_aurora.latitude, aurora_val, 
                                    cmap='coolwarm', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
                ax.set_title("Aurora (°C)")
                plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
                
                # HRES
                ax = axes[1]
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                if has_hres:
                    hres_lt = valid_time - hres_init_t
                    hres_lt_hours = int(pd.Timedelta(hres_lt).total_seconds() / 3600)
                    try:
                        fcst_2d_hres = hres_fcst["surface_air_temperature"].sel(lead_time=hres_lt_hours)
                        hres_val = fcst_2d_hres.values - 273.15
                        vmin_hres = vmin
                        vmax_hres = vmax
                        if not np.isnan(hres_val).all():
                            vmin_hres = min(vmin, np.nanmin(hres_val))
                            vmax_hres = max(vmax, np.nanmax(hres_val))
                        im1 = ax.pcolormesh(fcst_2d_hres.longitude, fcst_2d_hres.latitude, hres_val, 
                                            cmap='coolwarm', vmin=vmin_hres, vmax=vmax_hres, transform=ccrs.PlateCarree())
                        ax.set_title("HRES (°C)")
                        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
                    except KeyError:
                        ax.set_title("HRES (Data Unavailable)")
                else:
                    ax.set_title("HRES (Data Unavailable)")

                # ERA5
                ax = axes[2]
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                im2 = ax.pcolormesh(tgt_2d.longitude, tgt_2d.latitude, tgt_val, 
                                    cmap='coolwarm', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
                ax.set_title("ERA5 (°C)")
                plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
                
                # Set extent for all axes
                for ax in axes:
                    ax.set_extent([case.location.longitude_min, case.location.longitude_max,
                                   case.location.latitude_min, case.location.latitude_max], crs=ccrs.PlateCarree())
                
                # Save plot
                out_name = f"init_{pd.to_datetime(init_t).strftime('%Y%m%d%H')}_lt_{lt_hours:03d}.png"
                plt.savefig(case_out_dir / out_name, bbox_inches='tight', dpi=150)
                plt.close()
                print(f"Saved {out_name}")

if __name__ == '__main__':
    main()
