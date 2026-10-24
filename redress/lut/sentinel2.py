#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sentinel-2 look-up tables generated with REDRESS snow optics."""

from collections.abc import Iterable

import numpy as np
import xarray as xr

from redress.inputs.products import Model


def _coordinate(values: Iterable[float], name: str) -> np.ndarray:
    """Return a validated, one-dimensional floating-point coordinate."""

    coordinate = np.asarray(tuple(values), dtype=np.float64)
    if coordinate.ndim != 1 or coordinate.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(coordinate)):
        raise ValueError(f"{name} contains a non-finite value")
    if np.unique(coordinate).size != coordinate.size:
        raise ValueError(f"{name} contains duplicate values")
    return coordinate


def _geometry(
    solar_zenith_deg: float,
    view_zenith_deg: float,
    relative_azimuth_deg: float,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build a one-pixel flat-terrain geometry for the REDRESS model."""

    solar_zenith = np.deg2rad(solar_zenith_deg)
    view_zenith = np.deg2rad(view_zenith_deg)
    relative_azimuth = np.deg2rad(relative_azimuth_deg)

    shape = (1, 1)
    angles = xr.Dataset(
        {
            "SZA": (("x", "y"), np.full(shape, solar_zenith)),
            "SAA": (("x", "y"), np.full(shape, relative_azimuth)),
            "VZA": (("x", "y"), np.full(shape, view_zenith)),
            "VAA": (("x", "y"), np.zeros(shape)),
        }
    )
    topography = xr.Dataset(
        {
            "eff_sza": (("x", "y"), np.full(shape, solar_zenith)),
            "eff_vza": (("x", "y"), np.full(shape, view_zenith)),
        }
    )
    return angles, topography


def create_sentinel2_snow_brf_lut(
    platform: str,
    band_names: Iterable[str],
    specific_surface_area: Iterable[float],
    solar_zenith_angle: Iterable[float],
    view_zenith_angle: Iterable[float],
    relative_azimuth_angle: Iterable[float],
) -> xr.Dataset:
    """Create a band-integrated Sentinel-2 snow BRF look-up table.

    This first LUT represents a locally flat, fully illuminated snow surface.
    It isolates the effects of snow specific surface area and observation
    geometry. Atmospheric attenuation, terrain re-illumination and cast
    shadows are outside its scope.

    Parameters
    ----------
    platform
        Sentinel-2 platform, currently ``S2A`` or ``S2B``.
    band_names
        Ordered MSI bands to include.
    specific_surface_area
        Snow specific surface area values in m2 kg-1.
    solar_zenith_angle
        Solar zenith angles in degrees.
    view_zenith_angle
        Sensor viewing zenith angles in degrees.
    relative_azimuth_angle
        Solar-minus-view azimuth angles in degrees.

    Returns
    -------
    xarray.Dataset
        Dataset containing the dimensionless band reflectance factor ``brf``.
    """

    ssa = _coordinate(specific_surface_area, "specific_surface_area")
    sza = _coordinate(solar_zenith_angle, "solar_zenith_angle")
    vza = _coordinate(view_zenith_angle, "view_zenith_angle")
    raa = _coordinate(relative_azimuth_angle, "relative_azimuth_angle")

    if np.any(ssa <= 0):
        raise ValueError("specific_surface_area must be greater than zero")
    if np.any((sza < 0) | (sza >= 90)):
        raise ValueError("solar_zenith_angle must be in [0, 90) degrees")
    if np.any((vza < 0) | (vza >= 90)):
        raise ValueError("view_zenith_angle must be in [0, 90) degrees")
    if np.any((raa < 0) | (raa > 360)):
        raise ValueError("relative_azimuth_angle must be in [0, 360] degrees")

    model = Model()
    model.set_sentinel2(platform, band_names)
    band_count = len(model.meta["bandlist"])

    shape = (ssa.size, sza.size, vza.size, raa.size, band_count)
    brf = np.empty(shape, dtype=np.float32)

    for ssa_index, ssa_value in enumerate(ssa):
        model.ssa = float(ssa_value)
        for sza_index, sza_value in enumerate(sza):
            for vza_index, vza_value in enumerate(vza):
                for raa_index, raa_value in enumerate(raa):
                    model.angles, model.topo_bands = _geometry(
                        sza_value, vza_value, raa_value
                    )
                    model.compute_brf()
                    brf[ssa_index, sza_index, vza_index, raa_index, :] = [
                        np.asarray(value[1]).item() for value in model.brf
                    ]

    dataset = xr.Dataset(
        data_vars={
            "brf": (
                (
                    "specific_surface_area",
                    "solar_zenith_angle",
                    "view_zenith_angle",
                    "relative_azimuth_angle",
                    "band",
                ),
                brf,
            )
        },
        coords={
            "specific_surface_area": ssa,
            "solar_zenith_angle": sza,
            "view_zenith_angle": vza,
            "relative_azimuth_angle": raa,
            "band": model.meta["bandlist"],
            "wavelength_nm": (
                "band", np.asarray(model.meta["wavelengths"])
            ),
            "native_resolution_m": (
                "band", np.asarray(model.meta["native_resolution_m"])
            ),
        },
        attrs={
            "title": "REDRESS Sentinel-2 snow BRF look-up table",
            "platform": model.meta["platform"],
            "sensor": "MSI",
            "surface_geometry": "locally flat and fully illuminated",
            "spectral_integration": "Py6S MSI relative spectral response",
        },
    )

    dataset["brf"].attrs.update(
        long_name="bidirectional reflectance factor",
        units="1",
    )
    dataset["specific_surface_area"].attrs.update(
        long_name="snow specific surface area",
        units="m2 kg-1",
    )
    for coordinate_name in (
        "solar_zenith_angle",
        "view_zenith_angle",
        "relative_azimuth_angle",
    ):
        dataset[coordinate_name].attrs["units"] = "degree"
    dataset["wavelength_nm"].attrs.update(
        long_name="response-weighted effective wavelength",
        units="nm",
    )
    dataset["native_resolution_m"].attrs.update(
        long_name="native Sentinel-2 spatial resolution",
        units="m",
    )

    return dataset
