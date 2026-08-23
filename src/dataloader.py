import pickle
import numpy as np
import torch
import xarray as xr
import cdsapi
from pathlib import Path
from datetime import datetime, timedelta
from huggingface_hub import hf_hub_download
from aurora import Batch, Metadata
from aurora.insolation import insolation


class AuroraDataLoader:
    """
    A dedicated class to handle downloading, caching, and formatting 
    ERA5 weather data into Aurora 1.5-compatible PyTorch Batch objects.
    """
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.levels = [
            50, 100, 150, 200, 250, 300, 400, 
            500, 600, 700, 850, 925, 1000
        ]
        
        self.surf_request_vars = [
            "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
            "mean_sea_level_pressure", "2m_dewpoint_temperature", "total_column_water_vapour",
            "total_cloud_cover", "100m_u_component_of_wind", "100m_v_component_of_wind",
            "surface_pressure", "low_cloud_cover", "medium_cloud_cover", "high_cloud_cover",
            "skin_temperature", "soil_temperature_level_1", "volumetric_soil_water_layer_1",
            "sea_ice_cover", "snow_depth"
        ]
        
        self.atmos_request_vars = [
            "temperature", "u_component_of_wind", "v_component_of_wind",
            "specific_humidity", "geopotential"
        ]

        self.surf_name_map = {
            "t2m": "2t", "u10": "10u", "v10": "10v", "msl": "msl",
            "d2m": "2d", "tcwv": "tcwv", "tcc": "tcc", "u100": "100u",
            "v100": "100v", "sp": "sp", "lcc": "lcc", "mcc": "mcc",
            "hcc": "hcc", "skt": "skt", "stl1": "stl1", "swvl1": "swvl1",
            "siconc": "ci", "sd": "scaled_sd",
        }

        self.atmos_name_map = {
            "t": "t", "u": "u", "v": "v", "q": "q", "z": "z"
        }

        print("Loading Aurora 1.5 static variables...")
        self.static_path = hf_hub_download(
            repo_id="microsoft/aurora",
            filename="aurora-0.25-v1.5-static.pickle",
        )
        with open(self.static_path, "rb") as f:
            self.static_raw = pickle.load(f)
            
        self.global_lat = np.linspace(90, -90, 721)
        self.global_lon = np.linspace(0, 359.75, 1440)

    def _fetch_era5_combined(self, required_dts: list[datetime], bbox=None) -> xr.Dataset:
        """
        [INTERNAL] Downloads required timesteps in a single CDS request.
        """
        years = sorted(list(set(dt.strftime("%Y") for dt in required_dts)))
        months = sorted(list(set(dt.strftime("%m") for dt in required_dts)))
        days = sorted(list(set(dt.strftime("%d") for dt in required_dts)))
        times = sorted(list(set(dt.strftime("%H:00") for dt in required_dts)))
        
        start_str = required_dts[0].strftime("%Y%m%d_%H")
        end_str = required_dts[-1].strftime("%Y%m%d_%H")
        
        surf_path = self.cache_dir / f"era5_surf_{start_str}_to_{end_str}.nc"
        atmos_path = self.cache_dir / f"era5_atmos_{start_str}_to_{end_str}.nc"
        
        c = cdsapi.Client()

        if not surf_path.exists():
            print(f"Downloading Surface variables for {start_str} to {end_str}...")
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": self.surf_request_vars,
                    "year": years, "month": months, "day": days, "time": times,
                    "data_format": "netcdf",
                },
                str(surf_path),
            )

        if not atmos_path.exists():
            print(f"Downloading Atmospheric variables for {start_str} to {end_str}...")
            c.retrieve(
                "reanalysis-era5-pressure-levels",
                {
                    "product_type": "reanalysis",
                    "variable": self.atmos_request_vars,
                    "pressure_level": [str(l) for l in self.levels],
                    "year": years, "month": months, "day": days, "time": times,
                    "data_format": "netcdf",
                },
                str(atmos_path),
            )

        # .load() immediately pulls the NetCDF into memory, removing the need for .compute() later
        ds_surf = xr.open_dataset(surf_path, engine="netcdf4").load()
        ds_atmos = xr.open_dataset(atmos_path, engine="netcdf4").load()
        ds = xr.merge([ds_surf, ds_atmos], compat="override")
        if "valid_time" in ds.coords or "valid_time" in ds.dims:
            ds = ds.rename({"valid_time": "time"})
        if "pressure_level" in ds.coords or "pressure_level" in ds.dims:
            ds = ds.rename({"pressure_level": "level"})
        exact_times = [dt.strftime("%Y-%m-%dT%H:00:00") for dt in required_dts]
        ds = ds.sel(time=exact_times)

        if bbox is not None:
            l_min = bbox["lon_min"]
            l_max = bbox["lon_max"]

            if l_min < 0:
                # Negative lon_min (e.g. -3.0): fetch western part from 0-360 range,
                # then shift to negative coords to keep longitudes strictly increasing.
                part1 = ds.sel(longitude=slice(l_min + 360, 359.75))
                part2 = ds.sel(longitude=slice(0, l_max))
                part1 = part1.assign_coords(longitude=part1.longitude - 360)
                ds = xr.concat([part1, part2], dim="longitude")
            elif l_min > l_max:
                # Anti-meridian crossing (lon_min in 0-360 format, e.g. 357 > 52)
                part1 = ds.sel(longitude=slice(l_min, 360))
                part2 = ds.sel(longitude=slice(0, l_max))
                part1 = part1.assign_coords(longitude=part1.longitude - 360)
                ds = xr.concat([part1, part2], dim="longitude")
            else:
                ds = ds.sel(longitude=slice(l_min, l_max))

            ds = ds.sel(latitude=slice(bbox["lat_max"], bbox["lat_min"]))

        return ds

    def _format_single_sample(
        self, ds: xr.Dataset, input_dts: list[datetime], target_dts: list[datetime], bbox=None
    ) -> tuple[Batch, Batch]:
        """
        [INTERNAL] Converts xarray slice into input and target Aurora 1.5 Batch objects.
        """
        input_time_strs = [dt.strftime("%Y-%m-%dT%H:00:00") for dt in input_dts]
        target_time_strs = [dt.strftime("%Y-%m-%dT%H:00:00") for dt in target_dts]

        input_ds = ds.sel(time=input_time_strs)
        target_ds = ds.sel(time=target_time_strs)

        lat = ds.latitude.values.astype(np.float32)
        lon = ds.longitude.values.astype(np.float32)

        def build_tensors(sub_ds, time_list):
            surf = {}
            for nc_name, aurora_name in self.surf_name_map.items():
                data = sub_ds[nc_name].values 
                surf[aurora_name] = torch.from_numpy(np.nan_to_num(data, nan=0.0).astype(np.float32))

            sol = np.stack(
                [insolation([t], lat, lon, enforce_2d=True)[0] for t in time_list], axis=0,
            )
            surf["insolation"] = torch.from_numpy(sol.astype(np.float32))

            atmos = {}
            for nc_name, aurora_name in self.atmos_name_map.items():
                arr = sub_ds[nc_name].transpose("time", "level", "latitude", "longitude").values
                atmos[aurora_name] = torch.from_numpy(np.nan_to_num(arr, nan=0.0).astype(np.float32))

            return surf, atmos

        input_surf, input_atmos = build_tensors(input_ds, input_dts)
        if target_dts:
            target_surf, target_atmos = build_tensors(target_ds, target_dts)
        else:
            target_surf, target_atmos = {}, {}

        # Static Variables: Only calculate indices and slice if a bbox was actually provided
        static_inputs = {}
        if bbox is not None:
            lat_indices = [int(np.argmin(np.abs(self.global_lat - l))) for l in ds.latitude.values]
            lon_indices = [int(np.argmin(np.abs(self.global_lon - (l % 360)))) for l in ds.longitude.values]

            for k, v in self.static_raw.items():
                tensor_v = torch.from_numpy(v).float()
                if tensor_v.shape[-2:] == (721, 1440):
                    tensor_v = tensor_v[..., lat_indices, :][..., :, lon_indices]
                static_inputs[k] = tensor_v
        else:
            # Fast path for global data
            for k, v in self.static_raw.items():
                static_inputs[k] = torch.from_numpy(v).float()

        lon_vals = ds.longitude.values.astype(np.float64)
        # NOTE: Do NOT convert negative longitudes to 0-360 here.
        # After cross-meridian slicing, coords are e.g. [-3, ..., 52] (strictly increasing).
        # Converting back to [357, ..., 0, ..., 52] would break monotonicity,
        # which Aurora rejects. The static variable lookup already handles
        # negatives via `l % 360` (line 175).

        lat_tensor = torch.from_numpy(ds.latitude.values.copy())
        lon_tensor = torch.from_numpy(lon_vals)
        levels_tuple = tuple(int(lvl) for lvl in ds.level.values)

        t0_dt = input_dts[-1]
        
        input_metadata = Metadata(
            lat=lat_tensor, lon=lon_tensor, atmos_levels=levels_tuple, time=(t0_dt,)
        )
        target_metadata = Metadata(
            lat=lat_tensor, lon=lon_tensor, atmos_levels=levels_tuple, time=tuple(target_dts)
        )

        input_surf = {k: v.unsqueeze(0) for k, v in input_surf.items()}
        input_atmos = {k: v.unsqueeze(0) for k, v in input_atmos.items()}
        target_surf = {k: v.unsqueeze(0) for k, v in target_surf.items()}
        target_atmos = {k: v.unsqueeze(0) for k, v in target_atmos.items()}

        input_batch = Batch(
            surf_vars=input_surf, atmos_vars=input_atmos,
            static_vars=static_inputs, metadata=input_metadata,
        )
        if target_dts:
            target_batch = Batch(
                surf_vars=target_surf, atmos_vars=target_atmos,
                static_vars=static_inputs, metadata=target_metadata,
            )
        else:
            target_batch = None
            
        return input_batch, target_batch

    def get_batches(
        self,
        t0_dt: datetime,
        history_steps: int = 1,
        forecast_steps: int = 1,
        bbox=None,
    ) -> tuple[Batch, Batch]:
        """
        [PUBLIC] Returns a single model-ready input and ground-truth target batch.

        Args:
            t0_dt: The reference forecast datetime (t0).
            history_steps: How many 6-hour steps backward to include in input.
            forecast_steps: How many 6-hour steps forward to include in target.
            bbox: Optional bounding box dict. Defaults to None (full globe).
        """
        input_dts = [
            t0_dt - timedelta(hours=6 * i) for i in range(history_steps, -1, -1)
        ]
        target_dts = [
            t0_dt + timedelta(hours=6 * i) for i in range(1, forecast_steps + 1)
        ]
        
        all_required_dts = sorted(list(set(input_dts + target_dts)))
        
        ds = self._fetch_era5_combined(all_required_dts, bbox=bbox)
        
        return self._format_single_sample(ds, input_dts, target_dts, bbox=bbox)