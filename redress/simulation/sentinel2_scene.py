#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulate a georeferenced Sentinel-2 snow scene with REDRESS.

The functions in this module connect a projected DEM, a spatially uniform
Sentinel-2 geometry, snow optical properties and a prescribed atmosphere to
the REDRESS forward model. The output follows CF raster conventions and can
be written either as NetCDF or as a multiband GeoTIFF.

The DEM should contain a terrain buffer around the area that will be analysed.
Without that buffer, horizon and terrain-exchange calculations near the image
boundary omit terrain outside the raster.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from redress.inputs.products import Dem, Model


DEFAULT_BANDS = (
    "B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12",
)


@dataclass(frozen=True)
class Sentinel2Geometry:
    """Scene-mean illumination and viewing geometry, in degrees."""

    solar_zenith_deg: float
    solar_azimuth_deg: float
    view_zenith_deg: float = 0.0
    view_azimuth_deg: float = 0.0

    def validate(self) -> None:
        """Validate angle ranges accepted by the forward model."""

        values = (
            self.solar_zenith_deg,
            self.solar_azimuth_deg,
            self.view_zenith_deg,
            self.view_azimuth_deg,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("illumination and viewing angles must be finite")
        if not 0.0 <= self.solar_zenith_deg < 90.0:
            raise ValueError("solar_zenith_deg must be in [0, 90)")
        if not 0.0 <= self.view_zenith_deg < 90.0:
            raise ValueError("view_zenith_deg must be in [0, 90)")
        for name, value in (
            ("solar_azimuth_deg", self.solar_azimuth_deg),
            ("view_azimuth_deg", self.view_azimuth_deg),
        ):
            if not 0.0 <= value < 360.0:
                raise ValueError(f"{name} must be in [0, 360)")


@dataclass(frozen=True)
class Sentinel2Atmosphere:
    """Atmospheric inputs in the units expected by Py6S."""

    aerosol: str | Mapping[str, float]
    aod550: float
    water_vapour_g_cm2: float
    ozone_cm_atm: float
    month: int
    day: int
    atmospheric_profile: str | None = None

    def validate(self) -> None:
        """Validate the scalar atmospheric quantities."""

        for name, value in (
            ("aod550", self.aod550),
            ("water_vapour_g_cm2", self.water_vapour_g_cm2),
            ("ozone_cm_atm", self.ozone_cm_atm),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 1 <= int(self.month) <= 12:
            raise ValueError("month must be in [1, 12]")
        if not 1 <= int(self.day) <= 31:
            raise ValueError("day must be in [1, 31]")


@dataclass(frozen=True)
class SnowSurface:
    """Spatially uniform snow optical properties."""

    specific_surface_area_m2_kg: float = 20.0
    black_carbon_ng_g: float = 0.0
    dust_ug_g: float = 0.0
    dust_density_kg_m3: float = 2600.0
    dust_model: str = "muller2011"

    def validate(self) -> None:
        """Validate the snow and impurity properties."""

        if (
            not np.isfinite(self.specific_surface_area_m2_kg)
            or self.specific_surface_area_m2_kg <= 0.0
        ):
            raise ValueError(
                "specific_surface_area_m2_kg must be positive and finite"
            )
        for name, value in (
            ("black_carbon_ng_g", self.black_carbon_ng_g),
            ("dust_ug_g", self.dust_ug_g),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


def _raster_coordinates(geotransform, shape):
    """Return one-dimensional centre coordinates for a north-up raster."""

    rows, columns = shape
    x = geotransform[0] + (np.arange(columns) + 0.5) * geotransform[1]
    y = geotransform[3] + (np.arange(rows) + 0.5) * geotransform[5]
    return x, y


def read_dem(path: str | Path) -> Dem:
    """Read a projected, north-up, square-pixel DEM without resampling it."""

    from osgeo import gdal, osr

    path = Path(path).expanduser().resolve()
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise OSError(f"GDAL cannot read DEM: {path}")
    if dataset.RasterCount != 1:
        raise ValueError("the DEM must contain exactly one raster band")

    geotransform = dataset.GetGeoTransform()
    if not np.isclose(geotransform[2], 0.0) or not np.isclose(
        geotransform[4], 0.0
    ):
        raise ValueError("the DEM must be north-up without affine rotation")
    if not np.isclose(
        abs(geotransform[1]), abs(geotransform[5]), rtol=1e-6
    ):
        raise ValueError("the DEM must have square pixels")

    spatial_reference = osr.SpatialReference(wkt=dataset.GetProjection())
    if not spatial_reference.IsProjected():
        raise ValueError("the DEM CRS must be projected")
    linear_units = float(spatial_reference.GetLinearUnits())
    if not np.isclose(linear_units, 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("the DEM CRS must use metres")

    band = dataset.GetRasterBand(1)
    altitude = band.ReadAsArray().astype(np.float64)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        altitude[np.isclose(altitude, nodata, equal_nan=True)] = np.nan
    if not np.all(np.isfinite(altitude)):
        missing = int(np.count_nonzero(~np.isfinite(altitude)))
        raise ValueError(
            f"the DEM contains {missing} missing cells; fill the buffered "
            "DEM before calculating horizons"
        )

    x, y = _raster_coordinates(geotransform, altitude.shape)
    dem = Dem()
    dem.meta["path"] = str(path)
    dem.bands["altitude"] = xr.DataArray(
        altitude,
        dims=("x", "y"),
        coords={
            "easting": ("y", x),
            "northing": ("x", y),
        },
        attrs={"long_name": "surface altitude", "units": "m"},
    )
    dem.geotransform = tuple(geotransform)
    dem.pixel_size = (geotransform[1], geotransform[5])
    dem.crs_wkt = dataset.GetProjection()
    return dem


def _constant_angles(dem: Dem, geometry: Sentinel2Geometry) -> xr.Dataset:
    """Create DEM-aligned angle fields in radians."""

    geometry.validate()
    shape = dem.bands["altitude"].shape
    coords = dem.bands["altitude"].coords
    fields = {
        "SZA": geometry.solar_zenith_deg,
        "SAA": geometry.solar_azimuth_deg,
        "VZA": geometry.view_zenith_deg,
        "VAA": geometry.view_azimuth_deg,
    }
    angles = xr.Dataset(coords=coords)
    for name, value in fields.items():
        angles[name] = xr.DataArray(
            np.full(shape, np.deg2rad(value), dtype=np.float64),
            dims=("x", "y"),
            coords=coords,
            attrs={"units": "radian"},
        )
    return angles


def _output_dataset(model: Model, dem: Dem) -> xr.Dataset:
    """Collect model fields in a CF-oriented band, y, x dataset."""

    bands = tuple(model.meta["bandlist"])
    reflectance = np.stack(
        [model.toa_reflectance[band].values for band in bands]
    ).astype(np.float32)
    radiance = np.stack(
        [model.toa_rad[band].values for band in bands]
    ).astype(np.float32)
    x, y = _raster_coordinates(dem.geotransform, reflectance.shape[1:])

    output = xr.Dataset(
        data_vars={
            "toa_reflectance": (("band", "y", "x"), reflectance),
            "toa_radiance": (("band", "y", "x"), radiance),
        },
        coords={
            "band": np.asarray(bands, dtype=str),
            "wavelength_nm": (
                "band", np.asarray(model.meta["wavelengths"], dtype=float)
            ),
            "x": x,
            "y": y,
        },
        attrs={
            "title": "REDRESS synthetic Sentinel-2 snow scene",
            "Conventions": "CF-1.8",
            "platform": model.meta["platform"],
            "sensor": "MSI",
        },
    )
    output["toa_reflectance"].attrs.update(
        long_name="top-of-atmosphere reflectance factor", units="1"
    )
    output["toa_radiance"].attrs.update(
        long_name="top-of-atmosphere spectral radiance",
        units="W m-2 sr-1 um-1",
    )
    output["x"].attrs.update(
        standard_name="projection_x_coordinate", units="m", axis="X"
    )
    output["y"].attrs.update(
        standard_name="projection_y_coordinate", units="m", axis="Y"
    )
    output["wavelength_nm"].attrs.update(
        long_name="response-weighted effective wavelength", units="nm"
    )

    topographic_fields = (
        "altitude", "slope", "aspect", "eff_sza", "eff_vza", "vt", "ct",
        "beam_visibility", "view_ground",
    )
    for name in topographic_fields:
        if name in dem.bands:
            output[name] = xr.DataArray(
                np.asarray(dem.bands[name].values, dtype=np.float32),
                dims=("y", "x"),
                coords={"x": x, "y": y},
                attrs=dict(dem.bands[name].attrs),
            )

    return output


def simulate_sentinel2_scene(
    dem: Dem,
    geometry: Sentinel2Geometry,
    atmosphere: Sentinel2Atmosphere,
    snow: SnowSurface,
    *,
    platform: str = "S2B",
    band_names: Sequence[str] = DEFAULT_BANDS,
    case_number: int = 5,
    horizon_half_directions: int = 32,
    terrain_configuration_method: str = "hemispheric_closure",
    terrain_reflection_method: str = "neighbour_exitance",
    terrain_window_width_m: float = 1500.0,
    atmosphere_window_width_m: float = 2100.0,
    atmosphere_altitude_step_m: float | None = None,
    terrain_visibility_radius_m: float | None = None,
    verbose: bool = True,
) -> xr.Dataset:
    """Run REDRESS for every selected Sentinel-2 band over a DEM.

    ``neighbour_exitance`` is the practical default for image-sized rasters.
    ``visible_exitance`` is more geometrically explicit but currently traces
    the same terrain rays again for every band and radiance iteration, making
    it suitable mainly for small tiles and method experiments.
    """

    atmosphere.validate()
    snow.validate()
    if "altitude" not in dem.bands:
        raise ValueError("dem must contain an altitude band")

    dem.angles = _constant_angles(dem, geometry)
    dem.compute_horizon(N=horizon_half_directions)
    dem.compute_topo_bands(
        terrain_configuration_method=terrain_configuration_method
    )

    model = Model()
    model.set_sentinel2(platform, band_names)
    model.import_rt_model("python_6S")
    model.ssa = snow.specific_surface_area_m2_kg
    model.set_lap(
        black_carbon_ng_g=snow.black_carbon_ng_g,
        dust_ug_g=snow.dust_ug_g,
        dust_density_kg_m3=snow.dust_density_kg_m3,
        dust_model=snow.dust_model,
    )
    model.angles = dem.angles
    model.topo_bands = dem.bands
    model.set_rt_options(
        aero=atmosphere.aerosol,
        aod=atmosphere.aod550,
        water=atmosphere.water_vapour_g_cm2,
        ozone=atmosphere.ozone_cm_atm,
        atmos=atmosphere.atmospheric_profile,
        atcor=False,
        month=atmosphere.month,
        day=atmosphere.day,
    )
    model.simulate_TOA_radiance(
        case_number,
        verbose=verbose,
        pixel_resolution_m=abs(float(dem.pixel_size[0])),
        terrain_window_width_m=terrain_window_width_m,
        atmosphere_window_width_m=atmosphere_window_width_m,
        atmosphere_altitude_step_m=atmosphere_altitude_step_m,
        weight_terrain_by_neighbour_beam_visibility=False,
        terrain_reflection_method=terrain_reflection_method,
        terrain_visibility_radius_m=terrain_visibility_radius_m,
    )
    output = _output_dataset(model, dem)
    output.attrs.update(
        terrain_configuration_method=terrain_configuration_method,
        terrain_reflection_method=terrain_reflection_method,
        snow_specific_surface_area_m2_kg=snow.specific_surface_area_m2_kg,
        aod550=atmosphere.aod550,
        water_vapour_g_cm2=atmosphere.water_vapour_g_cm2,
        ozone_cm_atm=atmosphere.ozone_cm_atm,
    )
    return output


def _with_cf_grid_mapping(scene: xr.Dataset, dem: Dem) -> xr.Dataset:
    """Attach a GDAL-compatible CF grid-mapping variable."""

    output = scene.copy()
    geotransform = " ".join(str(value) for value in dem.geotransform)
    output["spatial_ref"] = xr.DataArray(
        0,
        attrs={
            "spatial_ref": dem.crs_wkt,
            "crs_wkt": dem.crs_wkt,
            "GeoTransform": geotransform,
        },
    )
    for name, variable in output.data_vars.items():
        if {"x", "y"}.issubset(variable.dims):
            output[name].attrs["grid_mapping"] = "spatial_ref"
    return output


def write_scene_netcdf(scene: xr.Dataset, dem: Dem, path: str | Path) -> Path:
    """Write a compressed CF NetCDF file that opens directly in QGIS."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    output = _with_cf_grid_mapping(scene, dem)
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for name, variable in output.data_vars.items()
        if variable.dtype.kind == "f"
    }
    output.to_netcdf(path, engine="netcdf4", encoding=encoding)
    return path


def write_multiband_geotiff(
    scene: xr.Dataset,
    dem: Dem,
    path: str | Path,
    variable: str = "toa_reflectance",
) -> Path:
    """Write one band-by-band floating-point GeoTIFF."""

    from osgeo import gdal

    if variable not in scene:
        raise KeyError(f"scene does not contain {variable!r}")
    values = scene[variable].transpose("band", "y", "x").values
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    raster = driver.Create(
        str(path),
        values.shape[2],
        values.shape[1],
        values.shape[0],
        gdal.GDT_Float32,
        options=("TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"),
    )
    if raster is None:
        raise OSError(f"GDAL could not create {path}")
    raster.SetGeoTransform(dem.geotransform)
    raster.SetProjection(dem.crs_wkt)
    for index, band_name in enumerate(scene["band"].values, start=1):
        band = raster.GetRasterBand(index)
        band.WriteArray(values[index - 1].astype(np.float32))
        band.SetDescription(str(band_name))
        band.SetNoDataValue(np.nan)
    raster.FlushCache()
    raster = None
    return path
