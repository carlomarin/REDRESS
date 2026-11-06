#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulate a georeferenced Sentinel-2 snow scene with REDRESS.

The functions in this module connect a projected DEM, Sentinel-2 solar and
band-mean viewing geometry, snow optical properties and a prescribed
atmosphere to the REDRESS forward model. The output follows CF raster
conventions and can be written either as NetCDF or as a multiband GeoTIFF.

The DEM should contain a terrain buffer around the area that will be analysed.
Without that buffer, horizon and terrain-exchange calculations near the image
boundary omit terrain outside the raster.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import xarray as xr

from redress.inputs.products import Dem, Model
from redress.optics.snow import SNOW_ALBEDO_MODELS
from redress.topography import dem_products
from redress.inputs.sentinel2 import (
    SENTINEL2_BAND_NAMES,
    normalize_sentinel2_band_name,
)
from redress.rtmodel.atmosphere import AtmosphericColumnProfile
from redress.readers.sentinel2_atmosphere import (
    Sentinel2AtmosphericInputs,
    Sentinel2Site,
    read_sentinel2_l1c_atmosphere,
)


DEFAULT_BANDS = (
    "B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12",
)


@dataclass(frozen=True)
class Sentinel2BandViewingGeometry:
    """Mean viewing direction of one Sentinel-2 band, in degrees."""

    zenith_deg: float
    azimuth_deg: float

    def validate(self) -> None:
        """Validate the viewing direction."""

        if not np.isfinite(self.zenith_deg) or not np.isfinite(
            self.azimuth_deg
        ):
            raise ValueError("viewing angles must be finite")
        if not 0.0 <= self.zenith_deg < 90.0:
            raise ValueError("viewing zenith must be in [0, 90) degrees")
        if not 0.0 <= self.azimuth_deg < 360.0:
            raise ValueError("viewing azimuth must be in [0, 360) degrees")


@dataclass(frozen=True)
class Sentinel2Geometry:
    """Solar geometry and mean Sentinel-2 viewing geometry, in degrees.

    ``view_zenith_deg`` and ``view_azimuth_deg`` provide a scene-level
    summary and a backward-compatible fallback. Geometry read from
    ``MTD_TL.xml`` also contains the mean direction of every selected band in
    ``viewing_by_band``.
    """

    solar_zenith_deg: float
    solar_azimuth_deg: float
    view_zenith_deg: float = 0.0
    view_azimuth_deg: float = 0.0
    viewing_by_band: Mapping[
        str, Sentinel2BandViewingGeometry
    ] = field(default_factory=dict)

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
        for band_name, viewing in self.viewing_by_band.items():
            if normalize_sentinel2_band_name(band_name) != band_name:
                raise ValueError(
                    "viewing_by_band keys must use canonical Sentinel-2 "
                    "band names"
                )
            if not isinstance(viewing, Sentinel2BandViewingGeometry):
                raise TypeError(
                    "viewing_by_band values must be "
                    "Sentinel2BandViewingGeometry instances"
                )
            viewing.validate()

    def viewing_for_band(self, band_name: str) -> tuple[float, float]:
        """Return the band mean direction or the scene-level fallback."""

        band_name = normalize_sentinel2_band_name(band_name)
        viewing = self.viewing_by_band.get(band_name)
        if viewing is None:
            return self.view_zenith_deg, self.view_azimuth_deg
        return viewing.zenith_deg, viewing.azimuth_deg


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
    column_profile: AtmosphericColumnProfile | None = None

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
    # "art" or "tartes"; see redress.optics.snow.SNOW_ALBEDO_MODELS.
    albedo_model: str = "art"

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
        if self.albedo_model not in SNOW_ALBEDO_MODELS:
            raise ValueError(
                "albedo_model must be one of: "
                + ", ".join(SNOW_ALBEDO_MODELS)
            )


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

    Every selected band retains its own mean viewing direction. Their vector
    mean is also stored as the scene-level summary and backward-compatible
    fallback. Solar geometry is common to every MSI band.

    TODO: interpolate the per-band, per-detector viewing-angle grids and use
    the detector footprint of each pixel when this extra spatial detail is
    required.
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
        viewing_by_band={
            name: Sentinel2BandViewingGeometry(*viewing_by_band[name])
            for name in requested_bands
        },
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
    surface to the central DEM elevation for the reported site values. The
    original model-surface columns and scale heights are also retained so the
    forward model can construct target-level columns at every altitude node.
    Ozone is left unchanged.
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
        column_profile=inputs.column_profile(),
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


def _constant_angles(
    dem: Dem,
    geometry: Sentinel2Geometry,
    band_name: str | None = None,
) -> xr.Dataset:
    """Create DEM-aligned angle fields in radians.

    When ``band_name`` is supplied, its mean viewing direction is used. A
    manually constructed geometry without per-band values retains the
    scene-level viewing direction as a backward-compatible fallback.
    """

    geometry.validate()
    if band_name is None:
        view_zenith = geometry.view_zenith_deg
        view_azimuth = geometry.view_azimuth_deg
    else:
        view_zenith, view_azimuth = geometry.viewing_for_band(band_name)
    shape = dem.bands["altitude"].shape
    coords = dem.bands["altitude"].coords
    fields = {
        "SZA": geometry.solar_zenith_deg,
        "SAA": geometry.solar_azimuth_deg,
        "VZA": view_zenith,
        "VAA": view_azimuth,
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


def _view_dependent_topography(
    dem: Dem,
    angles: xr.Dataset,
) -> xr.Dataset:
    """Return topography with viewing terms evaluated for one band.

    Altitude, slope, illumination, shadows, sky view and terrain
    configuration are independent of the MSI band viewing direction and are
    shared without recomputation. Effective viewing zenith, local relative
    azimuth and sensor visibility are recalculated for the supplied angles.
    """

    required = {"altitude", "slope", "aspect", "eff_sza", "vt", "ct"}
    missing = required.difference(dem.bands)
    if missing:
        raise ValueError(
            "common topographic products are missing: "
            + ", ".join(sorted(missing))
        )
    if "horizon_ele" not in dem.bands:
        raise ValueError("horizon_ele is required for sensor visibility")

    view_dependent = {
        "eff_vza",
        "local_relative_azimuth",
        "local_art_relative_azimuth",
        "view_ground",
    }
    spatial_variables = {
        name: variable
        for name, variable in dem.bands.data_vars.items()
        if name not in view_dependent
        and set(variable.dims).issubset({"x", "y"})
    }
    topography = xr.Dataset(spatial_variables)
    slope = dem.bands["slope"].data
    aspect = dem.bands["aspect"].data
    effective_viewing_zenith = dem_products.effective_zenith_angle(
        angles["VZA"].data,
        angles["VAA"].data,
        slope,
        aspect,
    )
    local_relative_azimuth = dem_products.local_relative_azimuth(
        angles["SZA"].data,
        angles["SAA"].data,
        angles["VZA"].data,
        angles["VAA"].data,
        slope,
        aspect,
    )
    local_art_relative_azimuth = dem_products.art_relative_azimuth(
        local_relative_azimuth
    )
    view_ground = dem_products.terrain_visibility(
        dem.bands["horizon_ele"].data,
        angles["VZA"].data,
        angles["VAA"].data,
        effective_viewing_zenith,
        min_cosine=0.0,
    )
    valid_surface = np.isfinite(slope) & np.isfinite(aspect)
    view_ground = np.where(valid_surface, view_ground, np.nan)

    coords = {
        name: coordinate
        for name, coordinate in dem.bands.coords.items()
        if set(coordinate.dims).issubset({"x", "y"})
    }
    for name, values in (
        ("eff_vza", effective_viewing_zenith),
        ("local_relative_azimuth", local_relative_azimuth),
        ("local_art_relative_azimuth", local_art_relative_azimuth),
        ("view_ground", view_ground),
    ):
        topography[name] = xr.DataArray(
            values,
            dims=("x", "y"),
            coords=coords,
        )
    for name in (
        "eff_vza", "local_relative_azimuth", "local_art_relative_azimuth",
    ):
        topography[name].attrs["units"] = "radian"
    topography["view_ground"].attrs.update(
        units="1",
        long_name="binary terrain visibility toward the sensor",
    )
    return topography


def _output_dataset(model: Model, dem: Dem) -> xr.Dataset:
    """Collect model fields in a CF-oriented band, y, x dataset."""

    bands = tuple(model.meta["bandlist"])
    reflectance = np.stack(
        [model.toa_reflectance[band].values for band in bands]
    ).astype(np.float32)
    radiance = np.stack(
        [model.toa_rad[band].values for band in bands]
    ).astype(np.float32)
    viewing_zenith = []
    viewing_azimuth = []
    for band in bands:
        angles, _ = model._geometry_for_band(band)
        viewing_zenith.append(np.rad2deg(np.nanmean(angles["VZA"].data)))
        viewing_azimuth.append(np.rad2deg(np.nanmean(angles["VAA"].data)))
    reference_angles, _ = model._geometry_for_band(bands[0])
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
            "viewing_zenith_deg": (
                "band", np.asarray(viewing_zenith, dtype=float)
            ),
            "viewing_azimuth_deg": (
                "band", np.asarray(viewing_azimuth, dtype=float)
            ),
            "x": x,
            "y": y,
        },
        attrs={
            "title": "REDRESS synthetic Sentinel-2 snow scene",
            "Conventions": "CF-1.8",
            "platform": model.meta["platform"],
            "sensor": "MSI",
            "viewing_geometry": "mean angle of each Sentinel-2 band",
            "solar_zenith_deg": float(np.rad2deg(
                np.nanmean(reference_angles["SZA"].data)
            )),
            "solar_azimuth_deg": float(np.rad2deg(
                np.nanmean(reference_angles["SAA"].data)
            )),
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
    output["viewing_zenith_deg"].attrs.update(
        long_name="band mean viewing zenith angle", units="degree"
    )
    output["viewing_azimuth_deg"].attrs.update(
        long_name="band mean viewing azimuth angle", units="degree"
    )

    common_topographic_fields = (
        "altitude", "slope", "aspect", "eff_sza", "vt", "ct",
        "beam_visibility",
    )
    for name in common_topographic_fields:
        if name in dem.bands:
            output[name] = xr.DataArray(
                np.asarray(dem.bands[name].values, dtype=np.float32),
                dims=("y", "x"),
                coords={"x": x, "y": y},
                attrs=dict(dem.bands[name].attrs),
            )

    for name in ("eff_vza", "view_ground"):
        values = np.stack([
            model._geometry_for_band(band)[1][name].values
            for band in bands
        ]).astype(np.float32)
        source = model._geometry_for_band(bands[0])[1][name]
        output[name] = xr.DataArray(
            values,
            dims=("band", "y", "x"),
            coords={"band": output["band"], "x": x, "y": y},
            attrs=dict(source.attrs),
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

    Mean viewing zenith and azimuth are selected independently for each band
    when present in ``geometry.viewing_by_band``. The expensive solar and
    terrain products are shared; only view-dependent products are rebuilt.

    ``neighbour_exitance`` is the practical default for image-sized rasters.
    ``visible_exitance`` is more geometrically explicit but currently traces
    the same terrain rays again for every band and radiance iteration, making
    it suitable mainly for small tiles and method experiments.
    """

    atmosphere.validate()
    snow.validate()
    geometry.validate()
    if "altitude" not in dem.bands:
        raise ValueError("dem must contain an altitude band")

    selected_bands = tuple(
        normalize_sentinel2_band_name(name) for name in band_names
    )
    if not selected_bands:
        raise ValueError("band_names cannot be empty")
    if len(selected_bands) != len(set(selected_bands)):
        raise ValueError("band_names contains duplicate Sentinel-2 bands")

    # Solar and terrain geometry are common to all bands. Compute these
    # expensive products once using the first band, then update only the
    # view-dependent terms for each band mean direction.
    dem.angles = _constant_angles(dem, geometry, selected_bands[0])
    dem.compute_horizon(N=horizon_half_directions)
    dem.compute_topo_bands(
        terrain_configuration_method=terrain_configuration_method
    )
    angles_by_band = {
        band: _constant_angles(dem, geometry, band)
        for band in selected_bands
    }
    topography_by_band = {
        band: _view_dependent_topography(dem, angles_by_band[band])
        for band in selected_bands
    }

    model = Model()
    model.set_sentinel2(platform, selected_bands)
    model.import_rt_model("python_6S")
    model.ssa = snow.specific_surface_area_m2_kg
    model.snow_albedo_model = snow.albedo_model
    model.set_lap(
        black_carbon_ng_g=snow.black_carbon_ng_g,
        dust_ug_g=snow.dust_ug_g,
        dust_density_kg_m3=snow.dust_density_kg_m3,
        dust_model=snow.dust_model,
    )
    model.angles = angles_by_band[selected_bands[0]]
    model.angles_by_band = angles_by_band
    model.topo_bands = topography_by_band[selected_bands[0]]
    model.topo_bands_by_band = topography_by_band
    model.set_rt_options(
        aero=atmosphere.aerosol,
        aod=atmosphere.aod550,
        water=atmosphere.water_vapour_g_cm2,
        ozone=atmosphere.ozone_cm_atm,
        atmos=atmosphere.atmospheric_profile,
        atcor=False,
        month=atmosphere.month,
        day=atmosphere.day,
        column_profile=atmosphere.column_profile,
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
    if atmosphere.column_profile is not None:
        output.attrs.update(
            atmospheric_column_reference_elevation_m=(
                atmosphere.column_profile.reference_elevation_m
            ),
            water_scale_height_m=(
                atmosphere.column_profile.water_scale_height_m
            ),
            aerosol_scale_height_m=(
                atmosphere.column_profile.aerosol_scale_height_m
            ),
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


def _scene_geotransform(scene: xr.Dataset, dem: Dem) -> tuple[float, ...]:
    """Return the GDAL geotransform of a full or cropped scene.

    The scene coordinates are pixel centres on the DEM grid, so a scene
    cropped with :func:`crop_scene` keeps its own origin.
    """

    geotransform = dem.geotransform
    return (
        float(scene["x"].values[0]) - 0.5 * geotransform[1],
        geotransform[1],
        geotransform[2],
        float(scene["y"].values[0]) - 0.5 * geotransform[5],
        geotransform[4],
        geotransform[5],
    )


def _with_cf_grid_mapping(scene: xr.Dataset, dem: Dem) -> xr.Dataset:
    """Attach a GDAL-compatible CF grid-mapping variable."""

    output = scene.copy()
    geotransform = " ".join(
        str(value) for value in _scene_geotransform(scene, dem)
    )
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
    raster.SetGeoTransform(_scene_geotransform(scene, dem))
    raster.SetProjection(dem.crs_wkt)
    for index, band_name in enumerate(scene["band"].values, start=1):
        band = raster.GetRasterBand(index)
        band.WriteArray(values[index - 1].astype(np.float32))
        band.SetDescription(str(band_name))
        band.SetNoDataValue(np.nan)
    raster.FlushCache()
    raster = None
    return path


# ---------------------------------------------------------------------------
# Comparison with an observed L1C product
# ---------------------------------------------------------------------------

SENTINEL2_NATIVE_RESOLUTION_M = {
    "B01": 60.0, "B02": 10.0, "B03": 10.0, "B04": 10.0, "B05": 20.0,
    "B06": 20.0, "B07": 20.0, "B08": 10.0, "B8A": 20.0, "B09": 60.0,
    "B10": 60.0, "B11": 20.0, "B12": 20.0,
}


@dataclass(frozen=True)
class Sentinel2TileGrid:
    """Upper-left corner and CRS of one Sentinel-2 tile resolution."""

    upper_left_x: float
    upper_left_y: float
    resolution_m: float
    epsg: int


def read_mtd_tl_grid(
    mtd_tl_file: str | Path,
    resolution_m: float = 20.0,
) -> Sentinel2TileGrid:
    """Read the tile grid of one resolution from ``MTD_TL.xml``."""

    mtd_tl_file = Path(mtd_tl_file).expanduser().resolve()
    root = ET.parse(mtd_tl_file).getroot()
    codes = [
        (element.text or "").strip()
        for element in root.iter()
        if _local_name(element) == "HORIZONTAL_CS_CODE"
    ]
    if len(codes) != 1 or not codes[0].upper().startswith("EPSG:"):
        raise ValueError(f"{mtd_tl_file}: expected one EPSG code")
    for element in root.iter():
        if (
            _local_name(element) == "Geoposition"
            and np.isclose(float(element.attrib["resolution"]), resolution_m)
        ):
            return Sentinel2TileGrid(
                upper_left_x=_child_float(element, "ULX"),
                upper_left_y=_child_float(element, "ULY"),
                resolution_m=float(resolution_m),
                epsg=int(codes[0].split(":", maxsplit=1)[1]),
            )
    raise ValueError(
        f"{mtd_tl_file}: no Geoposition at {resolution_m:g} m resolution"
    )


def resample_dem_to_sentinel2_grid(
    source_dem: str | Path,
    output_dem: str | Path,
    grid: Sentinel2TileGrid,
) -> Path:
    """Average a finer DEM onto the Sentinel-2 pixel grid.

    The output covers the largest extent of whole Sentinel-2 pixels that
    lies inside the source DEM, so every simulated pixel matches one
    observed pixel exactly.
    """

    from osgeo import gdal

    source_dem = Path(source_dem).expanduser().resolve()
    output_dem = Path(output_dem).expanduser().resolve()
    output_dem.parent.mkdir(parents=True, exist_ok=True)
    dataset = gdal.Open(str(source_dem), gdal.GA_ReadOnly)
    if dataset is None:
        raise OSError(f"GDAL cannot read DEM: {source_dem}")
    geotransform = dataset.GetGeoTransform()
    if not np.isclose(geotransform[2], 0.0) or not np.isclose(
        geotransform[4], 0.0
    ):
        raise ValueError("the source DEM must be north-up")
    west = geotransform[0]
    north = geotransform[3]
    east = west + dataset.RasterXSize * geotransform[1]
    south = north + dataset.RasterYSize * geotransform[5]
    resolution = grid.resolution_m
    # Snap inward so no output pixel is only partly covered.
    west = grid.upper_left_x + np.ceil(
        (west - grid.upper_left_x) / resolution
    ) * resolution
    east = grid.upper_left_x + np.floor(
        (east - grid.upper_left_x) / resolution
    ) * resolution
    north = grid.upper_left_y - np.ceil(
        (grid.upper_left_y - north) / resolution
    ) * resolution
    south = grid.upper_left_y - np.floor(
        (grid.upper_left_y - south) / resolution
    ) * resolution
    if east <= west or north <= south:
        raise ValueError("the source DEM is smaller than one tile pixel")
    nodata = dataset.GetRasterBand(1).GetNoDataValue()
    result = gdal.Warp(
        str(output_dem),
        dataset,
        dstSRS=f"EPSG:{grid.epsg}",
        outputBounds=(west, south, east, north),
        xRes=resolution,
        yRes=resolution,
        resampleAlg="average",
        srcNodata=nodata,
        dstNodata=-9999.0,
        outputType=gdal.GDT_Float32,
        creationOptions=("COMPRESS=DEFLATE", "TILED=YES"),
    )
    if result is None:
        raise OSError(f"GDAL could not resample {source_dem}")
    result.FlushCache()
    return output_dem


def edge_independent_window(
    dem: Dem,
    maximum_window_width_m: float,
    margin_m: float | None = None,
) -> tuple[slice, slice]:
    """Return the central rows and columns kept away from the DEM edges.

    By default a pixel qualifies when the widest moving window centred on
    it lies entirely inside the DEM. ``margin_m`` sets a smaller distance
    instead, accepting truncated windows and horizons near the edges.
    """

    pixel_size = abs(float(dem.pixel_size[0]))
    if margin_m is None:
        window_pixels = int(np.ceil(maximum_window_width_m / pixel_size))
        if window_pixels % 2 == 0:
            window_pixels += 1
        margin = window_pixels // 2
    else:
        if not margin_m >= 0.0:
            raise ValueError("margin_m must be non-negative")
        margin = int(np.ceil(margin_m / pixel_size))
    rows, columns = dem.bands["altitude"].shape
    if rows <= 2 * margin or columns <= 2 * margin:
        raise ValueError(
            "the DEM has no pixel at least "
            f"{margin * pixel_size:g} m from its edges"
        )
    return slice(margin, rows - margin), slice(margin, columns - margin)


def crop_scene(
    scene: xr.Dataset,
    rows: slice,
    columns: slice,
) -> xr.Dataset:
    """Return the scene restricted to the given DEM rows and columns."""

    cropped = scene.isel(y=rows, x=columns).copy()
    cropped.attrs.update(
        source_row_start=int(rows.start),
        source_column_start=int(columns.start),
    )
    return cropped


def _l1c_radiometric_scaling(safe_folder: Path):
    """Return the L1C quantification value and per-band offsets."""

    product_files = sorted(safe_folder.glob("MTD_MSIL1C.xml"))
    if len(product_files) != 1:
        raise FileNotFoundError(f"MTD_MSIL1C.xml not found in {safe_folder}")
    root = ET.parse(product_files[0]).getroot()
    quantification = [
        float(element.text)
        for element in root.iter()
        if _local_name(element) == "QUANTIFICATION_VALUE"
    ]
    if len(quantification) != 1:
        raise ValueError("expected one L1C QUANTIFICATION_VALUE")
    # Processing baselines before 04.00 have no offsets.
    offsets = {
        SENTINEL2_BAND_NAMES[int(element.attrib["band_id"])]: float(
            element.text
        )
        for element in root.iter()
        if _local_name(element) == "RADIO_ADD_OFFSET"
    }
    return quantification[0], offsets


def _jp2_precision(path: Path) -> int:
    """Return the component bit depth from a JPEG 2000 codestream header."""

    # The codestream starts with SOC (FF4F) followed directly by SIZ
    # (FF51); the first component's Ssiz byte lies 40 bytes after SIZ.
    marker = b"\xff\x4f\xff\x51"
    with path.open("rb") as stream:
        head = stream.read(1 << 20)
    start = head.find(marker)
    if start < 0 or start + 42 >= len(head):
        raise ValueError(f"no JPEG 2000 SIZ marker found in {path}")
    return (head[start + 42] & 0x7F) + 1


def _read_jp2_counts(path: Path, box: tuple[int, int, int, int]) -> np.ndarray:
    """Read a window of digital numbers at their stored bit depth.

    Pillow scales lower-precision 16-bit codestreams to the full 16-bit
    range, so the 15-bit L1C counts are shifted back.
    """

    from PIL import Image

    shift = 16 - _jp2_precision(path)
    with Image.open(path) as image:
        counts = np.asarray(image.crop(box), dtype=np.uint16)
    return np.right_shift(counts, shift).astype(np.float64)


def read_l1c_toa_reflectance(
    safe_folder: str | Path,
    grid: Sentinel2TileGrid,
    x: np.ndarray,
    y: np.ndarray,
    band_names: Sequence[str] = DEFAULT_BANDS,
) -> xr.Dataset:
    """Read observed L1C TOA reflectance on simulated pixel centres.

    ``x`` and ``y`` are pixel centres on the ``grid`` resolution. Finer
    bands are block-averaged onto that grid and coarser bands are repeated.
    No-data pixels become NaN.
    """

    from PIL import Image

    safe_folder = Path(safe_folder).expanduser().resolve()
    quantification, offsets = _l1c_radiometric_scaling(safe_folder)
    resolution = grid.resolution_m
    column_start = int(round(
        (x[0] - 0.5 * resolution - grid.upper_left_x) / resolution
    ))
    row_start = int(round(
        (grid.upper_left_y - (y[0] + 0.5 * resolution)) / resolution
    ))
    rows, columns = len(y), len(x)

    bands = tuple(normalize_sentinel2_band_name(name) for name in band_names)
    reflectance = np.full((len(bands), rows, columns), np.nan, np.float32)
    previous_limit = Image.MAX_IMAGE_PIXELS
    # 10 m bands exceed Pillow's decompression-bomb guard.
    Image.MAX_IMAGE_PIXELS = None
    try:
        for index, band in enumerate(bands):
            paths = sorted(
                safe_folder.glob(f"GRANULE/*/IMG_DATA/*_{band}.jp2")
            )
            if len(paths) != 1:
                raise FileNotFoundError(
                    f"expected one {band} image in {safe_folder}"
                )
            native = SENTINEL2_NATIVE_RESOLUTION_M[band]
            if native <= resolution:
                factor = int(round(resolution / native))
                box = (
                    column_start * factor,
                    row_start * factor,
                    (column_start + columns) * factor,
                    (row_start + rows) * factor,
                )
                counts = _read_jp2_counts(paths[0], box)
                counts[counts == 0] = np.nan
                counts = counts.reshape(
                    rows, factor, columns, factor
                ).mean(axis=(1, 3))
            else:
                factor = int(round(native / resolution))
                first_row = row_start // factor
                first_column = column_start // factor
                last_row = (row_start + rows - 1) // factor + 1
                last_column = (column_start + columns - 1) // factor + 1
                coarse = _read_jp2_counts(
                    paths[0],
                    (first_column, first_row, last_column, last_row),
                )
                coarse[coarse == 0] = np.nan
                row_index = (
                    np.arange(row_start, row_start + rows) // factor
                    - first_row
                )
                column_index = (
                    np.arange(column_start, column_start + columns)
                    // factor
                    - first_column
                )
                counts = coarse[np.ix_(row_index, column_index)]
            reflectance[index] = (
                counts + offsets.get(band, 0.0)
            ) / quantification
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    output = xr.Dataset(
        data_vars={
            "toa_reflectance": (("band", "y", "x"), reflectance),
        },
        coords={"band": np.asarray(bands, dtype=str), "x": x, "y": y},
        attrs={
            "title": "Sentinel-2 L1C TOA reflectance on the REDRESS grid",
            "source": str(safe_folder),
        },
    )
    output["toa_reflectance"].attrs.update(
        long_name="top-of-atmosphere reflectance factor", units="1"
    )
    return output


def read_snow_fraction_on_grid(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Read a fractional snow cover raster at the given pixel centres.

    The raster must lie on the same 20 m grid as the centres, in the same
    UTM zone; ETRS89 and WGS84 zones differ by well under a pixel. Values
    above one are taken as percentages. Pixels outside the raster or
    without a value are NaN.
    """

    from osgeo import gdal, osr

    path = Path(path).expanduser().resolve()
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None:
        raise OSError(f"GDAL cannot read snow fraction: {path}")
    reference = osr.SpatialReference(wkt=dataset.GetProjection())
    if reference.GetUTMZone() == 0:
        raise ValueError("the snow fraction raster must use a UTM CRS")
    geotransform = dataset.GetGeoTransform()
    resolution = float(geotransform[1])
    columns = (x - 0.5 * resolution - geotransform[0]) / resolution
    rows = (geotransform[3] - (y + 0.5 * resolution)) / -geotransform[5]
    if not (
        np.allclose(columns, np.round(columns), atol=1e-3)
        and np.allclose(rows, np.round(rows), atol=1e-3)
        and np.isclose(-geotransform[5], resolution)
    ):
        raise ValueError(
            "the snow fraction raster is not on the simulated pixel grid"
        )
    band = dataset.GetRasterBand(1)
    values = band.ReadAsArray().astype(np.float64)
    nodata = band.GetNoDataValue()
    if nodata is not None and np.isfinite(nodata):
        values[values == nodata] = np.nan
    if np.nanmax(values) > 1.0:
        values = values / 100.0

    rows = np.round(rows).astype(int)
    columns = np.round(columns).astype(int)
    inside_rows = (rows >= 0) & (rows < values.shape[0])
    inside_columns = (columns >= 0) & (columns < values.shape[1])
    fraction = np.full((len(y), len(x)), np.nan)
    fraction[np.ix_(inside_rows, inside_columns)] = values[
        np.ix_(rows[inside_rows], columns[inside_columns])
    ]
    return fraction


def reflectance_comparison_table(
    simulated: xr.Dataset,
    observed: xr.Dataset,
    mask: np.ndarray | None = None,
):
    """Summarise simulated minus observed TOA reflectance by band.

    ``mask`` restricts the statistics to the (y, x) pixels where it is
    true, for example pure snow.
    """

    import pandas as pd

    rows = []
    for band in simulated["band"].values:
        model = simulated["toa_reflectance"].sel(band=band).values.ravel()
        measured = observed["toa_reflectance"].sel(band=band).values.ravel()
        valid = np.isfinite(model) & np.isfinite(measured)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool).ravel()
        difference = model[valid] - measured[valid]
        rows.append({
            "band": str(band),
            "pixels": int(valid.sum()),
            "simulated_mean": float(np.mean(model[valid])),
            "observed_mean": float(np.mean(measured[valid])),
            "bias": float(np.mean(difference)),
            "rmse": float(np.sqrt(np.mean(difference ** 2))),
            "correlation": float(
                np.corrcoef(model[valid], measured[valid])[0, 1]
            ),
        })
    return pd.DataFrame(rows)
