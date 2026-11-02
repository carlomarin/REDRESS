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
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import xarray as xr

from redress.inputs.products import Dem, Model
from redress.inputs.sentinel2 import (
    SENTINEL2_BAND_NAMES,
    normalize_sentinel2_band_name,
)
from redress.readers.sentinel2_atmosphere import (
    Sentinel2AtmosphericInputs,
    Sentinel2Site,
    read_sentinel2_l1c_atmosphere,
)


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


@dataclass(frozen=True)
class Sentinel2AuxiliaryData:
    """Geometry, acquisition time and atmosphere sampled for one DEM pixel."""

    geometry: Sentinel2Geometry
    atmosphere: Sentinel2Atmosphere
    site: Sentinel2Site
    atmospheric_inputs: Sentinel2AtmosphericInputs
    acquisition_time: datetime


def _local_name(element) -> str:
    """Return an XML element name without its namespace."""

    return element.tag.rsplit("}", maxsplit=1)[-1]


def _child_float(element, name: str) -> float:
    """Read one finite floating-point value from a direct XML child."""

    matches = [
        child for child in element if _local_name(child) == name
    ]
    if len(matches) != 1 or matches[0].text is None:
        raise ValueError(
            f"expected one {name} value inside {_local_name(element)}"
        )
    value = float(matches[0].text)
    if not np.isfinite(value):
        raise ValueError(f"{name} contains a non-finite value")
    return value


def _mean_viewing_direction(viewing_angles):
    """Average viewing directions on the unit sphere."""

    zenith = np.deg2rad([item[0] for item in viewing_angles])
    azimuth = np.deg2rad([item[1] for item in viewing_angles])
    east = np.mean(np.sin(zenith) * np.sin(azimuth))
    north = np.mean(np.sin(zenith) * np.cos(azimuth))
    upward = np.mean(np.cos(zenith))
    norm = np.sqrt(east ** 2 + north ** 2 + upward ** 2)
    view_zenith = np.rad2deg(np.arccos(np.clip(upward / norm, -1.0, 1.0)))
    view_azimuth = np.mod(np.rad2deg(np.arctan2(east, north)), 360.0)
    # A tiny negative angle just west of north rounds to exactly 360.
    if view_azimuth >= 360.0:
        view_azimuth = 0.0
    return float(view_zenith), float(view_azimuth)


def read_mtd_tl_geometry(
    mtd_tl_file: str | Path,
    band_names: Sequence[str] = DEFAULT_BANDS,
) -> tuple[Sentinel2Geometry, datetime]:
    """Read mean solar and selected-band viewing geometry from ``MTD_TL.xml``.

    The current REDRESS model accepts one viewing geometry for all bands.
    The band-specific mean directions stored in the metadata are therefore
    averaged as three-dimensional unit vectors. Solar geometry is common to
    every MSI band.
    """

    mtd_tl_file = Path(mtd_tl_file).expanduser().resolve()
    if not mtd_tl_file.is_file():
        raise FileNotFoundError(f"MTD_TL metadata not found: {mtd_tl_file}")
    root = ET.parse(mtd_tl_file).getroot()

    sensing_times = [
        (element.text or "").strip()
        for element in root.iter()
        if _local_name(element) == "SENSING_TIME"
    ]
    if len(sensing_times) != 1:
        raise ValueError(
            f"{mtd_tl_file}: expected one SENSING_TIME; "
            f"found {len(sensing_times)}"
        )
    acquisition_time = datetime.fromisoformat(
        sensing_times[0].replace("Z", "+00:00")
    )

    mean_sun = [
        element for element in root.iter()
        if _local_name(element) == "Mean_Sun_Angle"
    ]
    if len(mean_sun) != 1:
        raise ValueError(
            f"{mtd_tl_file}: expected one Mean_Sun_Angle; "
            f"found {len(mean_sun)}"
        )
    solar_zenith = _child_float(mean_sun[0], "ZENITH_ANGLE")
    solar_azimuth = _child_float(mean_sun[0], "AZIMUTH_ANGLE")

    requested_bands = tuple(
        normalize_sentinel2_band_name(name) for name in band_names
    )
    band_id_to_name = dict(enumerate(SENTINEL2_BAND_NAMES))
    viewing_by_band = {}
    for element in root.iter():
        if _local_name(element) != "Mean_Viewing_Incidence_Angle":
            continue
        band_id = element.attrib.get("bandId")
        if band_id is None:
            continue
        try:
            band_name = band_id_to_name[int(band_id)]
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"invalid Sentinel-2 bandId {band_id!r} in {mtd_tl_file}"
            ) from error
        viewing_by_band[band_name] = (
            _child_float(element, "ZENITH_ANGLE"),
            _child_float(element, "AZIMUTH_ANGLE"),
        )

    missing = set(requested_bands).difference(viewing_by_band)
    if missing:
        raise ValueError(
            f"{mtd_tl_file}: missing mean viewing angles for "
            + ", ".join(sorted(missing))
        )
    viewing_angles = [viewing_by_band[name] for name in requested_bands]
    view_zenith, view_azimuth = _mean_viewing_direction(viewing_angles)
    geometry = Sentinel2Geometry(
        solar_zenith_deg=solar_zenith,
        solar_azimuth_deg=solar_azimuth,
        view_zenith_deg=view_zenith,
        view_azimuth_deg=view_azimuth,
    )
    geometry.validate()
    return geometry, acquisition_time


def _raster_coordinates(geotransform, shape):
    """Return one-dimensional centre coordinates for a north-up raster."""

    rows, columns = shape
    x = geotransform[0] + (np.arange(columns) + 0.5) * geotransform[1]
    y = geotransform[3] + (np.arange(rows) + 0.5) * geotransform[5]
    return x, y


def fill_missing_dem_cells(
    altitude: np.ndarray,
    pixel_size_m: float,
    maximum_fill_distance_m: float,
) -> tuple[np.ndarray, int, float]:
    """Fill small DEM gaps from the nearest finite elevation.

    Nearest-neighbour filling preserves the local elevation range and avoids
    the artificial cliffs produced by replacing NoData with zero. The maximum
    distance prevents the procedure from inventing terrain across a large
    internal void or outside a strongly irregular DEM footprint.
    """

    from scipy.ndimage import distance_transform_edt

    values = np.asarray(altitude, dtype=np.float64)
    missing = ~np.isfinite(values)
    missing_count = int(np.count_nonzero(missing))
    if missing_count == 0:
        return values.copy(), 0, 0.0
    if missing_count == values.size:
        raise ValueError("the DEM contains no finite elevation cells")
    if (
        not np.isfinite(maximum_fill_distance_m)
        or maximum_fill_distance_m <= 0.0
    ):
        raise ValueError(
            "maximum_fill_distance_m must be positive and finite"
        )

    distance_m, nearest = distance_transform_edt(
        missing,
        sampling=float(pixel_size_m),
        return_distances=True,
        return_indices=True,
    )
    maximum_distance = float(np.max(distance_m[missing]))
    if maximum_distance > maximum_fill_distance_m:
        raise ValueError(
            f"the DEM contains {missing_count} missing cells and the largest "
            f"gap is {maximum_distance:.1f} m from valid terrain; this "
            f"exceeds maximum_fill_distance_m={maximum_fill_distance_m:g}. "
            "Use a DEM with a complete rectangular terrain buffer or raise "
            "the limit only after inspecting the NoData pattern."
        )

    filled = values.copy()
    filled[missing] = values[nearest[0, missing], nearest[1, missing]]
    return filled, missing_count, maximum_distance


def read_dem(
    path: str | Path,
    *,
    fill_missing: bool = True,
    maximum_fill_distance_m: float = 200.0,
) -> Dem:
    """Read a projected, north-up, square-pixel DEM without resampling it.

    Small NoData gaps are filled from the nearest valid elevation by default.
    The number of filled cells and maximum fill distance are recorded in the
    DEM metadata for reporting and reproducibility.
    """

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
    missing_count = int(np.count_nonzero(~np.isfinite(altitude)))
    maximum_fill_distance = 0.0
    if missing_count and fill_missing:
        altitude, missing_count, maximum_fill_distance = (
            fill_missing_dem_cells(
                altitude,
                pixel_size_m=abs(float(geotransform[1])),
                maximum_fill_distance_m=maximum_fill_distance_m,
            )
        )
    elif missing_count:
        raise ValueError(
            f"the DEM contains {missing_count} missing cells and "
            "fill_missing=False"
        )

    x, y = _raster_coordinates(geotransform, altitude.shape)
    dem = Dem()
    dem.meta["path"] = str(path)
    dem.meta["filled_missing_cells"] = missing_count
    dem.meta["maximum_fill_distance_m"] = maximum_fill_distance
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


def central_pixel_indices(dem: Dem) -> tuple[int, int]:
    """Return the row and column of the DEM central pixel."""

    rows, columns = dem.bands["altitude"].shape
    return rows // 2, columns // 2


def central_pixel_site(dem: Dem, name: str = "central_pixel") -> Sentinel2Site:
    """Return the WGS84 position and DEM elevation of the central pixel."""

    from pyproj import CRS, Transformer

    row, column = central_pixel_indices(dem)
    x = (
        dem.geotransform[0]
        + (column + 0.5) * dem.geotransform[1]
        + (row + 0.5) * dem.geotransform[2]
    )
    y = (
        dem.geotransform[3]
        + (column + 0.5) * dem.geotransform[4]
        + (row + 0.5) * dem.geotransform[5]
    )
    transformer = Transformer.from_crs(
        CRS.from_wkt(dem.crs_wkt), CRS.from_epsg(4326), always_xy=True
    )
    longitude, latitude = transformer.transform(x, y)
    elevation = float(dem.bands["altitude"].values[row, column])
    site = Sentinel2Site(
        latitude_deg=float(latitude),
        longitude_deg=float(longitude),
        elevation_m=elevation,
        name=name,
    )
    site.validate()
    return site


def read_sentinel2_auxiliary_data(
    safe_folder: str | Path,
    mtd_tl_file: str | Path,
    dem: Dem,
    band_names: Sequence[str] = DEFAULT_BANDS,
    *,
    legacy_aod550: float | None = None,
    legacy_aerosol_model: str = "Continental",
) -> Sentinel2AuxiliaryData:
    """Prepare central-pixel geometry and Py6S atmosphere from an L1C SAFE.

    Water vapour and aerosol optical depth are corrected from the CAMS model
    surface to the central DEM elevation. Ozone is left unchanged. The result
    should be used with a scene-mean 6S altitude, because the corrected columns
    describe this central pixel rather than every elevation in the DEM.
    """

    geometry, acquisition_time = read_mtd_tl_geometry(
        mtd_tl_file, band_names=band_names
    )
    site = central_pixel_site(dem)
    inputs = read_sentinel2_l1c_atmosphere(
        safe_folder,
        site,
        apply_elevation_correction=True,
        legacy_aod550=legacy_aod550,
        legacy_aerosol_model=legacy_aerosol_model,
    )
    aerosol = (
        dict(inputs.aerosol_components)
        if inputs.aerosol_components is not None
        else inputs.aerosol_model
    )
    atmosphere = Sentinel2Atmosphere(
        aerosol=aerosol,
        aod550=inputs.aod550,
        water_vapour_g_cm2=inputs.water_vapour_g_cm2,
        ozone_cm_atm=inputs.ozone_cm_atm,
        month=acquisition_time.month,
        day=acquisition_time.day,
    )
    atmosphere.validate()
    return Sentinel2AuxiliaryData(
        geometry=geometry,
        atmosphere=atmosphere,
        site=site,
        atmospheric_inputs=inputs,
        acquisition_time=acquisition_time,
    )


def validate_central_pixel_context(
    dem: Dem,
    maximum_window_width_m: float,
) -> None:
    """Check that the central pixel is unaffected by moving-window edges."""

    pixel_size = abs(float(dem.pixel_size[0]))
    if not np.isclose(pixel_size, 20.0, rtol=0.0, atol=1e-3):
        raise ValueError(
            f"this workflow requires a 20 m DEM; found {pixel_size:g} m"
        )
    window_pixels = int(np.ceil(maximum_window_width_m / pixel_size))
    if window_pixels % 2 == 0:
        window_pixels += 1
    rows, columns = dem.bands["altitude"].shape
    if rows < window_pixels or columns < window_pixels:
        raise ValueError(
            "the DEM is too small to simulate an edge-independent central "
            f"pixel: at least {window_pixels} by {window_pixels} cells are "
            f"required, found {rows} by {columns}"
        )


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


def central_pixel_spectrum(scene: xr.Dataset, dem: Dem) -> xr.Dataset:
    """Extract the simulated spectrum and diagnostics at the DEM centre."""

    row, column = central_pixel_indices(dem)
    central = scene.isel(y=row, x=column, drop=False).copy()
    central.attrs.update(
        source_row=row,
        source_column=column,
        spatial_selection="central DEM pixel",
    )
    return central


def write_central_pixel_csv(
    spectrum: xr.Dataset,
    path: str | Path,
) -> Path:
    """Write the central Sentinel-2 spectrum and its coordinates as CSV."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    table = spectrum[["toa_reflectance", "toa_radiance"]].to_dataframe()
    table.reset_index().to_csv(path, index=False)
    return path


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
