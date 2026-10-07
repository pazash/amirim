import sys
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path
import warnings

# Ignore cartopy/shapely warnings if any
warnings.filterwarnings("ignore")

# Import evaluator functions
sys.path.append(str(Path(__file__).resolve().parent))
from extreme_precip_evaluator import PRECIP_CASES, load_era5_zarr, load_hres_zarr, standardize_longitude

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    forecast_dir = base_dir / "ewb_precip_forecasts"
    
    if not forecast_dir.exists():
        print(f"Error: {forecast_dir} not found.")
        print("Please ensure you have generated the predictions first by running extreme_precip_evaluator.py")
        return

    plots_dir = base_dir / "plots" / "precip_maps"
    
    # Take the first event
    case = PRECIP_CASES[0]
    case_out_dir = plots_dir / f"event_{case.case_id_number}_{case.title.replace(' ', '_')}"
    case_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading ERA5 and HRES for {case.title}...")
    era5_ds = load_era5_zarr()
    hres_ds = load_hres_zarr()
    
    if "total_precipitation_6hr" in hres_ds.data_vars:
        hres_var = "total_precipitation_6hr"
    elif "total_precipitation" in hres_ds.data_vars:
        hres_var = "total_precipitation"
    else:
        hres_var = None

    print(f"Slicing ERA5 for event period: {case.start_date} to {case.end_date}")
    era5_case = era5_ds.sel(time=slice(case.start_date, case.end_date))
    era5_case = case.location.mask(era5_case).compute()
    era5_case = standardize_longitude(era5_case)
    
    nc_files = sorted(list(forecast_dir.glob(f"*_{case.case_id_number}.nc")))
    if not nc_files:
        print(f"No Aurora forecast files found for case {case.case_id_number} in {forecast_dir}")
        return

    print(f"Found {len(nc_files)} prediction files for {case.title}.")

    for nc_file in nc_files:
        print(f"Processing {nc_file.name}...")
        aurora_fcst = xr.open_dataset(nc_file)
        init_t = aurora_fcst.init_time.values[0]
        
        aurora_fcst = case.location.mask(aurora_fcst).compute()
        aurora_fcst = standardize_longitude(aurora_fcst)
        
        try:
            hres_fcst = hres_ds.sel(init_time=init_t, method="pad")
            hres_init_t = hres_fcst.init_time.values
            hres_fcst = case.location.mask(hres_fcst).compute()
            hres_fcst = standardize_longitude(hres_fcst)
            has_hres = True and hres_var is not None
        except KeyError:
            has_hres = False
            
        for lt in aurora_fcst.lead_time.values:
            valid_time = init_t + lt
            lt_hours = int(pd.Timedelta(lt).total_seconds() / 3600)
            
            # Check if valid_time is within the event period
            if pd.to_datetime(case.start_date) <= pd.to_datetime(valid_time) <= pd.to_datetime(case.end_date):
                try:
                    tgt_2d = era5_case["total_precipitation"].sel(time=valid_time)
                except KeyError:
                    continue
                
                # Extract Aurora
                fcst_2d_aurora = aurora_fcst["scaled_tp_1h"].sel(
                    init_time=init_t, 
                    lead_time=pd.to_timedelta(f"{lt_hours}h")
                )
                
                # Calculate max value for colorbar normalization
                vmax = 0.1
                if not np.isnan(fcst_2d_aurora.values).all():
                    vmax = max(vmax, np.nanmax(fcst_2d_aurora.values * 1000.0))
                if not np.isnan(tgt_2d.values).all():
                    vmax = max(vmax, np.nanmax(tgt_2d.values * 1000.0))

                # Plot setup
                fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={'projection': ccrs.PlateCarree()})
                plt.subplots_adjust(wspace=0.3)
                init_str = pd.to_datetime(init_t).strftime('%Y-%m-%d %H:%M')
                valid_str = pd.to_datetime(valid_time).strftime('%Y-%m-%d %H:%M')
                fig.suptitle(f"{case.title} - Init: {init_str} | Valid: {valid_str} (Lead: {lt_hours}h)", fontsize=16)

                vmin = 0

                # Aurora
                ax = axes[0]
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                aurora_val = fcst_2d_aurora.values * 1000.0
                im0 = ax.pcolormesh(fcst_2d_aurora.longitude, fcst_2d_aurora.latitude, aurora_val, 
                                    cmap='Blues', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
                ax.set_title("Aurora (mm/h)")
                plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
                
                # HRES
                ax = axes[1]
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                if has_hres:
                    hres_lt = valid_time - hres_init_t
                    hres_lt_hours = int(pd.Timedelta(hres_lt).total_seconds() / 3600)
                    try:
                        fcst_2d_hres = hres_fcst[hres_var].sel(lead_time=hres_lt_hours)
                        hres_val = (fcst_2d_hres.values / 6.0) * 1000.0 # Convert to mm/h
                        vmax_hres = vmax
                        if not np.isnan(hres_val).all():
                            vmax_hres = max(vmax, np.nanmax(hres_val))
                        im1 = ax.pcolormesh(fcst_2d_hres.longitude, fcst_2d_hres.latitude, hres_val, 
                                            cmap='Blues', vmin=vmin, vmax=vmax_hres, transform=ccrs.PlateCarree())
                        ax.set_title("HRES (mm/h)")
                        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
                    except KeyError:
                        ax.set_title("HRES (Data Unavailable)")
                else:
                    ax.set_title("HRES (Data Unavailable)")

                # ERA5
                ax = axes[2]
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                era5_val = tgt_2d.values * 1000.0
                im2 = ax.pcolormesh(tgt_2d.longitude, tgt_2d.latitude, era5_val, 
                                    cmap='Blues', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
                ax.set_title("ERA5 (mm/h)")
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
