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
import calendar


class AuroraDataLoader:
    """
    A dedicated class to handle downloading, caching, and formatting 
    ERA5 weather data into Aurora 1.5-compatible PyTorch Batch objects.
    """
    def __init__(self, cache_dir: Path, pretrained_only: bool = False):
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pretrained_only = pretrained_only
        self.nc_suffix = "_pretrained.nc" if pretrained_only else ".nc"
        
        self.levels = [
            50, 100, 150, 200, 250, 300, 400, 
            500, 600, 700, 850, 925, 1000
        ]
        
        if self.pretrained_only:
            self.surf_request_vars = [
                "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
                "mean_sea_level_pressure"
            ]
        else:
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
        
        c = cdsapi.Client()
        
        # Build requests, adding bounding box area if provided
        surf_req = {
            "product_type": "reanalysis",
            "variable": self.surf_request_vars,
            "year": years, "month": months, "day": days, "time": times,
            "data_format": "netcdf",
        }
        atmos_req = {
            "product_type": "reanalysis",
            "variable": self.atmos_request_vars,
            "pressure_level": [str(l) for l in self.levels],
            "year": years, "month": months, "day": days, "time": times,
            "data_format": "netcdf",
        }
        
        # Determine cache path and apply area filter if bbox is provided
        if bbox is not None:
            # Create a completely separate cache directory for regional data
            regional_cache = self.cache_dir / "regional"
            regional_cache.mkdir(exist_ok=True)
            
            surf_path = regional_cache / f"era5_surf_{start_str}_to_{end_str}{self.nc_suffix}"
            atmos_path = regional_cache / f"era5_atmos_{start_str}_to_{end_str}{self.nc_suffix}"
            
            # CDS area format: [North, West, South, East]
            area = [bbox["lat_max"], bbox["lon_min"], bbox["lat_min"], bbox["lon_max"]]
            surf_req["area"] = area
            atmos_req["area"] = area
        else:
            surf_path = self.cache_dir / f"era5_surf_{start_str}_to_{end_str}{self.nc_suffix}"
            atmos_path = self.cache_dir / f"era5_atmos_{start_str}_to_{end_str}{self.nc_suffix}"

        if not surf_path.exists():
            print(f"Downloading Surface variables for {start_str} to {end_str}...")
            c.retrieve("reanalysis-era5-single-levels", surf_req, str(surf_path))

        if not atmos_path.exists():
            print(f"Downloading Atmospheric variables for {start_str} to {end_str}...")
            c.retrieve("reanalysis-era5-pressure-levels", atmos_req, str(atmos_path))

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
            if bbox["lon_min"] > bbox["lon_max"]:
                part1 = ds.sel(longitude=slice(bbox["lon_min"], 360))
                part2 = ds.sel(longitude=slice(0, bbox["lon_max"]))
                part1 = part1.assign_coords(longitude=part1.longitude - 360)
                ds = xr.concat([part1, part2], dim="longitude")
            else:
                ds = ds.sel(longitude=slice(bbox["lon_min"], bbox["lon_max"]))

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
        lon_vals = np.where(lon_vals < 0, lon_vals + 360, lon_vals)

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

    # ================================================================
    # Bulk Download Methods — for efficient large-scale training
    # ================================================================

    def _bulk_cache_dir(self, bbox=None) -> Path:
        """[INTERNAL] Returns the base cache directory for bulk monthly downloads."""
        if bbox is not None:
            bulk_dir = self.cache_dir / "bulk" / "regional"
        else:
            bulk_dir = self.cache_dir / "bulk"
        bulk_dir.mkdir(parents=True, exist_ok=True)
        return bulk_dir

    def bulk_download_month(self, year: int, month: int, bbox=None) -> None:
        """
        [PUBLIC] Downloads all 6-hourly ERA5 data for a single month in one CDS API
        request. Data is cached as monthly NetCDF files for efficient bulk training.

        This is vastly more efficient than per-sample downloads: 2 API calls per month
        vs. ~120 per month if downloading per-sample.

        Args:
            year: Year to download (e.g., 2019).
            month: Month to download (1-12).
            bbox: Optional bounding box dict for regional downloads.
        """
        days_in_month = calendar.monthrange(year, month)[1]

        bulk_dir = self._bulk_cache_dir(bbox)
        surf_path = bulk_dir / f"era5_surf_{year}_{month:02d}{self.nc_suffix}"
        atmos_path = bulk_dir / f"era5_atmos_{year}_{month:02d}{self.nc_suffix}"

        if surf_path.exists() and atmos_path.exists():
            print(f"  Bulk cache hit: {year}-{month:02d}")
            return

        c = cdsapi.Client()
        days = [f"{d:02d}" for d in range(1, days_in_month + 1)]
        times = ["00:00", "06:00", "12:00", "18:00"]

        surf_req = {
            "product_type": "reanalysis",
            "variable": self.surf_request_vars,
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": times,
            "data_format": "netcdf",
        }
        atmos_req = {
            "product_type": "reanalysis",
            "variable": self.atmos_request_vars,
            "pressure_level": [str(l) for l in self.levels],
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": times,
            "data_format": "netcdf",
        }

        if bbox is not None:
            area = [bbox["lat_max"], bbox["lon_min"], bbox["lat_min"], bbox["lon_max"]]
            surf_req["area"] = area
            atmos_req["area"] = area

        if not surf_path.exists():
            print(f"  Downloading surface vars for {year}-{month:02d}...")
            c.retrieve("reanalysis-era5-single-levels", surf_req, str(surf_path))

        if not atmos_path.exists():
            print(f"  Downloading atmospheric vars for {year}-{month:02d}...")
            c.retrieve("reanalysis-era5-pressure-levels", atmos_req, str(atmos_path))

        print(f"  ✓ Downloaded: {year}-{month:02d}")

    def bulk_download_range(self, years, bbox=None) -> None:
        """
        [PUBLIC] Downloads all months for the given years using bulk monthly CDS requests.
        Much more efficient than per-sample downloads for large-scale training.

        Args:
            years: Iterable of years (e.g., range(2016, 2021) for 2016-2020).
            bbox: Optional bounding box dict for regional downloads.
        """
        years_list = list(years)
        total_months = len(years_list) * 12
        count = 0
        for year in years_list:
            for month in range(1, 13):
                count += 1
                print(f"[{count}/{total_months}] Bulk downloading {year}-{month:02d}...")
                self.bulk_download_month(year, month, bbox)

    def get_batches_from_bulk(
        self,
        t0_dt: datetime,
        history_steps: int = 1,
        forecast_steps: int = 1,
        bbox=None,
    ) -> tuple:
        """
        [PUBLIC] Loads batches from pre-downloaded bulk monthly data.

        Functionally identical to get_batches(), but reads from the bulk monthly
        cache instead of downloading per-sample. Call bulk_download_range() first.

        Args:
            t0_dt: The reference forecast datetime (t0).
            history_steps: How many 6-hour steps backward to include in input.
            forecast_steps: How many 6-hour steps forward to include in target.
            bbox: Optional bounding box dict. Must match the bbox used during download.
        """
        input_dts = [
            t0_dt - timedelta(hours=6 * i) for i in range(history_steps, -1, -1)
        ]
        target_dts = [
            t0_dt + timedelta(hours=6 * i) for i in range(1, forecast_steps + 1)
        ]
        all_required_dts = sorted(list(set(input_dts + target_dts)))

        # Determine which months we need to load
        required_months = sorted(set((dt.year, dt.month) for dt in all_required_dts))

        bulk_dir = self._bulk_cache_dir(bbox)

        # Load and merge required months (lazily opened, only selected timestamps loaded)
        datasets = []
        for year, month in required_months:
            surf_path = bulk_dir / f"era5_surf_{year}_{month:02d}{self.nc_suffix}"
            atmos_path = bulk_dir / f"era5_atmos_{year}_{month:02d}{self.nc_suffix}"

            if not surf_path.exists() or not atmos_path.exists():
                raise FileNotFoundError(
                    f"Bulk data for {year}-{month:02d} not found at {bulk_dir}. "
                    f"Run bulk_download_month({year}, {month}) first."
                )

            ds_surf = xr.open_dataset(surf_path, engine="netcdf4")
            ds_atmos = xr.open_dataset(atmos_path, engine="netcdf4")
            ds_month = xr.merge([ds_surf, ds_atmos], compat="override")

            # Normalize coordinate names before potential concatenation
            if "valid_time" in ds_month.coords or "valid_time" in ds_month.dims:
                ds_month = ds_month.rename({"valid_time": "time"})
            if "pressure_level" in ds_month.coords or "pressure_level" in ds_month.dims:
                ds_month = ds_month.rename({"pressure_level": "level"})

            datasets.append(ds_month)

        # Merge across months if the sample spans a month boundary
        if len(datasets) == 1:
            ds = datasets[0]
        else:
            ds = xr.concat(datasets, dim="time")

        # Select exact timestamps and load into memory
        exact_times = [dt.strftime("%Y-%m-%dT%H:00:00") for dt in all_required_dts]
        ds = ds.sel(time=exact_times).load()

        # Apply bbox slicing (consistent with _fetch_era5_combined)
        if bbox is not None:
            if bbox["lon_min"] > bbox["lon_max"]:
                part1 = ds.sel(longitude=slice(bbox["lon_min"], 360))
                part2 = ds.sel(longitude=slice(0, bbox["lon_max"]))
                part1 = part1.assign_coords(longitude=part1.longitude - 360)
                ds = xr.concat([part1, part2], dim="longitude")
            else:
                ds = ds.sel(longitude=slice(bbox["lon_min"], bbox["lon_max"]))

            ds = ds.sel(latitude=slice(bbox["lat_max"], bbox["lat_min"]))

        return self._format_single_sample(ds, input_dts, target_dts, bbox=bbox)