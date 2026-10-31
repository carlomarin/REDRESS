#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read atmospheric inputs embedded in a Sentinel-2 L1C SAFE product.

Sentinel-2 products contain ECMWF total-column water vapour and ozone. From
processing baseline 04.00 onwards, they also contain CAMS aerosol optical
depths. This module samples those fields at a site, converts them to the units
expected by 6S, and optionally adjusts low-altitude constituents from the
model surface to the physical site elevation.

The aerosol mixture is an approximation. CAMS species AOD fractions are used
as proxies for the volume fractions required by the four-component 6S aerosol
model. The total CAMS AOD at 550 nm remains the optical-depth constraint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np


STANDARD_GRAVITY_M_S2 = 9.80665
WATER_VAPOUR_SCALE_HEIGHT_M = 2000.0
AEROSOL_SCALE_HEIGHT_M = 1500.0
CAMS_BASELINE = (4, 0)

_ECMWF_FIELDS = {
    "water": ("tcwv", "totalcolumnwatervapour", "totalcolumnwater"),
    "ozone": ("tco3", "totalcolumnozone"),
}
_CAMS_FIELDS = {
    "aod550": ("aod550", "totalaerosolopticaldepthat550nm"),
    "surface_geopotential": ("z", "geopotential", "surfacegeopotential"),
    "black_carbon": ("bcaod550", "blackcarbonaerosolopticaldepthat550nm"),
    "dust": ("duaod550", "dustaerosolopticaldepthat550nm"),
    "organic_matter": (
        "omaod550", "organicmatteraerosolopticaldepthat550nm",
    ),
    "sea_salt": ("ssaod550", "seasaltaerosolopticaldepthat550nm"),
    "sulphate": ("suaod550", "sulphateaerosolopticaldepthat550nm"),
}


@dataclass(frozen=True)
class Sentinel2Site:
    """Geodetic position and elevation of one atmospheric sampling site."""

    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    name: str = "site"

    def validate(self) -> None:
        """Validate the finite coordinate ranges used by the reader."""

        values = (self.latitude_deg, self.longitude_deg, self.elevation_m)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("site coordinates and elevation must be finite")
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("site latitude must be in [-90, 90] degrees")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError("site longitude must be in [-180, 180] degrees")


@dataclass(frozen=True)
class Sentinel2AtmosphericInputs:
    """Atmospheric quantities at the target elevation, ready for Py6S."""

    processing_baseline: str
    site: Sentinel2Site
    water_vapour_g_cm2: float
    ozone_cm_atm: float
    aod550: float
    aerosol_model: str
    aerosol_components: Mapping[str, float] | None
    model_surface_elevation_m: float | None
    raw_water_vapour_g_cm2: float
    raw_aod550: float
    water_elevation_factor: float
    aerosol_elevation_factor: float
    aerosol_source: str
    auxiliary_files: tuple[Path, ...]
    notes: tuple[str, ...] = ()

    def py6s_settings(self) -> dict[str, Any]:
        """Return native Py6S atmosphere and aerosol parameter objects.

        The target altitude must still be assigned to ``SixS.altitudes``.
        User water and ozone columns represent the atmosphere above that
        target; 6S does not expect sea-level totals in this configuration.
        """

        from Py6S import AeroProfile, AtmosProfile

        if self.aerosol_components is None:
            try:
                aerosol_type = getattr(AeroProfile, self.aerosol_model)
            except AttributeError as error:
                raise ValueError(
                    f"Py6S has no aerosol model {self.aerosol_model!r}"
                ) from error
            aerosol_profile = AeroProfile.PredefinedType(aerosol_type)
        else:
            aerosol_profile = AeroProfile.User(**self.aerosol_components)

        return {
            "atmos_profile": AtmosProfile.UserWaterAndOzone(
                self.water_vapour_g_cm2,
                self.ozone_cm_atm,
            ),
            "aero_profile": aerosol_profile,
            "aot550": self.aod550,
            "target_altitude_km": self.site.elevation_m / 1000.0,
        }

    def redress_rt_options(self) -> dict[str, Any]:
        """Return keyword arguments accepted by ``rtmodel.run``."""

        aerosol = (
            dict(self.aerosol_components)
            if self.aerosol_components is not None
            else self.aerosol_model
        )
        return {
            "alt": self.site.elevation_m,
            "aero": aerosol,
            "aod": self.aod550,
            "water": self.water_vapour_g_cm2,
            "ozone": self.ozone_cm_atm,
        }


@dataclass(frozen=True)
class _Sample:
    value: float
    unit: str
    path: Path


def _normalise_text(value: str) -> str:
    """Return a compact lowercase identifier for metadata comparisons."""

    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _parse_baseline(value: str) -> tuple[int, int]:
    """Parse a Sentinel-2 processing baseline such as ``05.11``."""

    match = re.fullmatch(r"\s*(\d{1,2})\.(\d{1,2})\s*", value)
    if match is None:
        raise ValueError(f"invalid Sentinel-2 processing baseline {value!r}")
    return int(match.group(1)), int(match.group(2))


def _metadata_root(safe_path: Path) -> Path:
    """Locate the product-level Sentinel-2 L1C metadata document."""

    candidates = sorted(safe_path.glob("MTD_MSIL1C*.xml"))
    if len(candidates) != 1:
        raise ValueError(
            f"{safe_path}: expected one MTD_MSIL1C XML file; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _read_processing_baseline(metadata_path: Path) -> str:
    """Read the processing baseline without depending on the XML namespace."""

    root = ET.parse(metadata_path).getroot()
    product_type = None
    baseline = None

    for element in root.iter():
        name = element.tag.rsplit("}", maxsplit=1)[-1]
        text = (element.text or "").strip()
        if name == "PRODUCT_TYPE":
            product_type = text
        elif name == "PROCESSING_BASELINE":
            baseline = text

    accepted_types = ("MSIL1C", "S2MSI1C")
    if product_type and not any(
        name in product_type.upper() for name in accepted_types
    ):
        raise ValueError(
            f"{metadata_path}: expected a Sentinel-2 L1C product; "
            f"found {product_type!r}"
        )
    if baseline is None:
        name_match = re.search(r"_N(\d{2})(\d{2})_", metadata_path.parent.name)
        if name_match is not None:
            baseline = f"{name_match.group(1)}.{name_match.group(2)}"
    if baseline is None:
        raise ValueError(f"{metadata_path}: PROCESSING_BASELINE is missing")

    _parse_baseline(baseline)
    return baseline


def _find_auxiliary_files(safe_path: Path, stem: str) -> tuple[Path, ...]:
    """Return auxiliary GRIB files for all granules in a SAFE product."""

    paths = tuple(
        path for path in sorted(safe_path.rglob(f"{stem}*"))
        if path.is_file()
    )
    if not paths:
        raise FileNotFoundError(
            f"{safe_path}: no {stem} auxiliary file was found"
        )
    return paths


def _band_identifiers(band) -> tuple[str, ...]:
    """Collect normalized GRIB identifiers exposed by GDAL."""

    metadata = band.GetMetadata() or {}
    values = [
        band.GetDescription() or "",
        metadata.get("GRIB_ELEMENT", ""),
        metadata.get("GRIB_COMMENT", ""),
        metadata.get("GRIB_SHORT_NAME", ""),
        metadata.get("long_name", ""),
        metadata.get("short_name", ""),
    ]
    return tuple(_normalise_text(value) for value in values if value)


def _band_unit(band) -> str:
    """Return the physical unit recorded for a GDAL GRIB band."""

    metadata = band.GetMetadata() or {}
    values = (
        band.GetUnitType() or "",
        metadata.get("GRIB_UNIT", ""),
        metadata.get("units", ""),
    )
    return next((value.strip() for value in values if value.strip()), "")


def _matching_bands(dataset, aliases: Sequence[str]):
    """Yield GDAL bands whose metadata match a requested field."""

    aliases = tuple(_normalise_text(alias) for alias in aliases)
    for index in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(index)
        identifiers = _band_identifiers(band)
        if any(
            alias == identifier
            or (len(alias) > 3 and alias in identifier)
            for alias in aliases
            for identifier in identifiers
        ):
            yield band


def _site_in_dataset_coordinates(dataset, site: Sentinel2Site) -> tuple[float, float]:
    """Transform site longitude and latitude to raster coordinates."""

    from osgeo import osr

    projection = dataset.GetProjection()
    if not projection:
        return site.longitude_deg, site.latitude_deg

    source = osr.SpatialReference()
    source.SetWellKnownGeogCS("WGS84")
    target = osr.SpatialReference()
    target.ImportFromWkt(projection)
    if hasattr(source, "SetAxisMappingStrategy"):
        strategy = osr.OAMS_TRADITIONAL_GIS_ORDER
        source.SetAxisMappingStrategy(strategy)
        target.SetAxisMappingStrategy(strategy)

    transform = osr.CoordinateTransformation(source, target)
    x_coordinate, y_coordinate, _ = transform.TransformPoint(
        site.longitude_deg,
        site.latitude_deg,
    )
    return x_coordinate, y_coordinate


def _bilinear_sample(
    array: np.ndarray,
    geotransform: Sequence[float],
    x_coordinate: float,
    y_coordinate: float,
    nodata: float | None = None,
) -> float:
    """Sample a georeferenced array bilinearly at one map coordinate."""

    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("auxiliary raster must be two-dimensional")

    transform = np.asarray(
        [[geotransform[1], geotransform[2]],
         [geotransform[4], geotransform[5]]],
        dtype=np.float64,
    )
    offset = np.asarray(
        [x_coordinate - geotransform[0], y_coordinate - geotransform[3]],
        dtype=np.float64,
    )
    column_edge, row_edge = np.linalg.solve(transform, offset)
    column = column_edge - 0.5
    row = row_edge - 0.5

    if not (-0.5 <= column <= values.shape[1] - 0.5):
        raise ValueError("site lies outside the auxiliary raster longitude extent")
    if not (-0.5 <= row <= values.shape[0] - 0.5):
        raise ValueError("site lies outside the auxiliary raster latitude extent")

    column = float(np.clip(column, 0.0, values.shape[1] - 1.0))
    row = float(np.clip(row, 0.0, values.shape[0] - 1.0))
    column0 = int(np.floor(column))
    row0 = int(np.floor(row))
    column1 = min(column0 + 1, values.shape[1] - 1)
    row1 = min(row0 + 1, values.shape[0] - 1)

    samples = np.asarray([
        values[row0, column0],
        values[row0, column1],
        values[row1, column0],
        values[row1, column1],
    ])
    weights = np.asarray([
        (1.0 - (column - column0)) * (1.0 - (row - row0)),
        (column - column0) * (1.0 - (row - row0)),
        (1.0 - (column - column0)) * (row - row0),
        (column - column0) * (row - row0),
    ])
    valid = np.isfinite(samples)
    if nodata is not None and np.isfinite(nodata):
        valid &= samples != nodata
    if not np.any(valid):
        raise ValueError("all auxiliary samples surrounding the site are missing")

    valid_weights = weights[valid]
    if np.sum(valid_weights) <= 0.0:
        raise ValueError("the weighted auxiliary sample is missing")
    return float(np.average(samples[valid], weights=valid_weights))


def _sample_field(
    paths: Sequence[Path],
    aliases: Sequence[str],
    site: Sentinel2Site,
) -> _Sample:
    """Sample a named field from the granule covering the site."""

    from osgeo import gdal

    errors = []
    for path in paths:
        dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
        if dataset is None:
            errors.append(f"{path}: GDAL could not open the file")
            continue
        bands = tuple(_matching_bands(dataset, aliases))
        if len(bands) != 1:
            errors.append(
                f"{path}: expected one field matching {tuple(aliases)}; "
                f"found {len(bands)}"
            )
            continue

        band = bands[0]
        try:
            x_coordinate, y_coordinate = _site_in_dataset_coordinates(
                dataset, site
            )
            value = _bilinear_sample(
                band.ReadAsArray(),
                dataset.GetGeoTransform(),
                x_coordinate,
                y_coordinate,
                band.GetNoDataValue(),
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            errors.append(f"{path}: {error}")
            continue
        return _Sample(value=value, unit=_band_unit(band), path=path)

    detail = "\n".join(errors)
    raise ValueError(
        f"site {site.name!r} is not represented by a valid auxiliary field:\n"
        f"{detail}"
    )


def _unit_key(unit: str) -> str:
    """Normalize common unit notation while retaining dimensional meaning."""

    return (
        unit.lower()
        .replace("**", "^")
        .replace("−", "-")
        .replace("·", "")
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
    )


def water_to_g_cm2(value: float, unit: str = "kg m-2") -> float:
    """Convert total-column water vapour to grams per square centimetre."""

    key = _unit_key(unit)
    if key in ("", "kg/m^2", "kgm^-2", "kgm-2", "mm"):
        result = value * 0.1
    elif key in ("g/cm^2", "gcm^-2", "gcm-2"):
        result = value
    else:
        raise ValueError(f"unsupported total-column water unit {unit!r}")
    return float(result)


def ozone_to_cm_atm(value: float, unit: str = "kg m-2") -> float:
    """Convert total-column ozone to centimetre-atmospheres.

    One Dobson unit is 2.1415e-5 kg m-2 and one cm-atm is 1000 DU.
    """

    key = _unit_key(unit)
    if key in ("", "kg/m^2", "kgm^-2", "kgm-2"):
        result = value / 0.021415
    elif key in ("du", "dobsonunit", "dobsonunits"):
        result = value / 1000.0
    elif key in ("cm-atm", "cmatm"):
        result = value
    else:
        raise ValueError(f"unsupported total-column ozone unit {unit!r}")
    return float(result)


def geopotential_to_elevation_m(value: float, unit: str = "m2 s-2") -> float:
    """Convert surface geopotential to geopotential height in metres."""

    key = _unit_key(unit)
    if key in ("", "m^2/s^2", "m2s-2", "m^2s^-2"):
        result = value / STANDARD_GRAVITY_M_S2
    elif key in ("m", "metre", "meter"):
        result = value
    else:
        raise ValueError(f"unsupported surface-geopotential unit {unit!r}")
    return float(result)


def elevation_factor(
    site_elevation_m: float,
    model_elevation_m: float,
    scale_height_m: float,
) -> float:
    """Return an exponential column correction from model to site elevation."""

    if not np.isfinite(scale_height_m) or scale_height_m <= 0.0:
        raise ValueError("scale height must be finite and greater than zero")
    difference_m = site_elevation_m - model_elevation_m
    return float(np.exp(-difference_m / scale_height_m))


def cams_aerosol_components(
    species_aod550: Mapping[str, float],
) -> dict[str, float]:
    """Map CAMS species AOD shares to the four 6S aerosol components."""

    required = {
        "black_carbon", "dust", "organic_matter", "sea_salt", "sulphate",
    }
    missing = required.difference(species_aod550)
    if missing:
        raise ValueError(
            "missing CAMS aerosol species: " + ", ".join(sorted(missing))
        )
    values = {name: float(species_aod550[name]) for name in required}
    if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("CAMS aerosol species AODs must be finite and non-negative")

    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("the sum of CAMS aerosol species AODs is zero")
    return {
        "dust": values["dust"] / total,
        "water": (values["organic_matter"] + values["sulphate"]) / total,
        "oceanic": values["sea_salt"] / total,
        "soot": values["black_carbon"] / total,
    }


def read_sentinel2_l1c_atmosphere(
    safe_folder: str | Path,
    site: Sentinel2Site,
    *,
    apply_elevation_correction: bool = True,
    water_scale_height_m: float = WATER_VAPOUR_SCALE_HEIGHT_M,
    aerosol_scale_height_m: float = AEROSOL_SCALE_HEIGHT_M,
    legacy_aod550: float | None = None,
    legacy_aerosol_model: str = "Continental",
) -> Sentinel2AtmosphericInputs:
    """Read site-specific atmospheric inputs from a Sentinel-2 L1C SAFE.

    Parameters
    ----------
    safe_folder
        Root of an unpacked Sentinel-2 ``MSIL1C`` SAFE product.
    site
        WGS84 coordinates and physical target elevation.
    apply_elevation_correction
        Correct TCWV and AOD from the CAMS model surface to the site using an
        exponential scale-height approximation. Ozone is not corrected.
    water_scale_height_m, aerosol_scale_height_m
        Scale heights used for the approximate correction.
    legacy_aod550, legacy_aerosol_model
        Explicit aerosol fallback for baselines before 04.00, where CAMS
        aerosol fields are not embedded in the product.
    """

    safe_path = Path(safe_folder).expanduser().resolve()
    if not safe_path.is_dir():
        raise NotADirectoryError(f"Sentinel-2 SAFE folder not found: {safe_path}")
    if "MSIL1C" not in safe_path.name.upper():
        raise ValueError(
            f"{safe_path}: expected a Sentinel-2 MSIL1C SAFE folder"
        )
    site.validate()

    metadata_path = _metadata_root(safe_path)
    baseline = _read_processing_baseline(metadata_path)
    ecmwf_paths = _find_auxiliary_files(safe_path, "AUX_ECMWFT")
    water_sample = _sample_field(ecmwf_paths, _ECMWF_FIELDS["water"], site)
    ozone_sample = _sample_field(ecmwf_paths, _ECMWF_FIELDS["ozone"], site)
    raw_water = water_to_g_cm2(water_sample.value, water_sample.unit)
    ozone = ozone_to_cm_atm(ozone_sample.value, ozone_sample.unit)

    notes = []
    auxiliary_files = {water_sample.path, ozone_sample.path}
    model_elevation = None
    water_factor = 1.0
    aerosol_factor = 1.0

    if _parse_baseline(baseline) >= CAMS_BASELINE:
        cams_paths = _find_auxiliary_files(safe_path, "AUX_CAMS")
        total_sample = _sample_field(
            cams_paths, _CAMS_FIELDS["aod550"], site
        )
        geopotential_sample = _sample_field(
            cams_paths, _CAMS_FIELDS["surface_geopotential"], site
        )
        raw_aod550 = float(total_sample.value)
        model_elevation = geopotential_to_elevation_m(
            geopotential_sample.value,
            geopotential_sample.unit,
        )
        auxiliary_files.update((total_sample.path, geopotential_sample.path))

        species = {}
        for name in (
            "black_carbon", "dust", "organic_matter", "sea_salt", "sulphate",
        ):
            sample = _sample_field(cams_paths, _CAMS_FIELDS[name], site)
            species[name] = sample.value
            auxiliary_files.add(sample.path)
        components = cams_aerosol_components(species)
        aerosol_model = "User"
        aerosol_source = "CAMS species AOD fractions"
        notes.append(
            "CAMS species AOD fractions are proxies for the volume fractions "
            "required by the four-component 6S aerosol model"
        )

        species_sum = sum(species.values())
        if raw_aod550 > 0.0 and not np.isclose(
            species_sum, raw_aod550, rtol=0.05, atol=1e-4
        ):
            notes.append(
                "CAMS species AOD sum differs from total AOD550 by more than "
                "5%; species were normalized only to define the 6S mixture"
            )
    else:
        if legacy_aod550 is None:
            raise ValueError(
                f"processing baseline {baseline} predates embedded CAMS "
                "aerosols; provide legacy_aod550 and legacy_aerosol_model"
            )
        if not np.isfinite(legacy_aod550) or legacy_aod550 < 0.0:
            raise ValueError("legacy_aod550 must be finite and non-negative")
        raw_aod550 = float(legacy_aod550)
        aerosol_model = legacy_aerosol_model
        components = None
        aerosol_source = "explicit legacy fallback"
        notes.append(
            "No CAMS species are available before processing baseline 04.00"
        )

    if apply_elevation_correction:
        if model_elevation is None:
            notes.append(
                "Elevation correction was skipped because the legacy product "
                "does not contain CAMS surface geopotential"
            )
        else:
            water_factor = elevation_factor(
                site.elevation_m, model_elevation, water_scale_height_m
            )
            aerosol_factor = elevation_factor(
                site.elevation_m, model_elevation, aerosol_scale_height_m
            )
            if site.elevation_m < model_elevation:
                notes.append(
                    "The site is below the CAMS model surface; the exponential "
                    "correction extrapolates water vapour and AOD downward"
                )

    water = raw_water * water_factor
    aod550 = raw_aod550 * aerosol_factor
    if any(value < 0.0 or not np.isfinite(value) for value in (
        water, ozone, aod550,
    )):
        raise ValueError("converted atmospheric quantities are not physical")

    return Sentinel2AtmosphericInputs(
        processing_baseline=baseline,
        site=site,
        water_vapour_g_cm2=water,
        ozone_cm_atm=ozone,
        aod550=aod550,
        aerosol_model=aerosol_model,
        aerosol_components=components,
        model_surface_elevation_m=model_elevation,
        raw_water_vapour_g_cm2=raw_water,
        raw_aod550=raw_aod550,
        water_elevation_factor=water_factor,
        aerosol_elevation_factor=aerosol_factor,
        aerosol_source=aerosol_source,
        auxiliary_files=tuple(sorted(auxiliary_files)),
        notes=tuple(notes),
    )
