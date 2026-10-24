#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sentinel-2 MSI band definitions used by REDRESS.

The module keeps the spatial and spectral properties of Sentinel-2 bands in
one place. Spectral response functions are obtained from the predefined Py6S
filters so that atmospheric simulations can use the complete band response
rather than a single nominal wavelength.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


SENTINEL2_BAND_NAMES = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
)

SENTINEL2_NATIVE_RESOLUTION_M = {
    "B01": 60,
    "B02": 10,
    "B03": 10,
    "B04": 10,
    "B05": 20,
    "B06": 20,
    "B07": 20,
    "B08": 10,
    "B8A": 20,
    "B09": 60,
    "B10": 60,
    "B11": 20,
    "B12": 20,
}

_PLATFORM_ALIASES = {
    "S2A": "S2A",
    "SENTINEL-2A": "S2A",
    "SENTINEL2A": "S2A",
    "S2B": "S2B",
    "SENTINEL-2B": "S2B",
    "SENTINEL2B": "S2B",
}


def normalize_sentinel2_platform(platform: str) -> str:
    """Return the canonical Sentinel-2 platform name.

    Py6S 1.9.2 provides MSI response functions for Sentinel-2A and
    Sentinel-2B. Other platforms are rejected explicitly to prevent using a
    response function measured for a different instrument.
    """

    key = platform.strip().upper().replace("_", "-")
    try:
        return _PLATFORM_ALIASES[key]
    except KeyError as error:
        supported = ", ".join(sorted(set(_PLATFORM_ALIASES.values())))
        raise ValueError(
            f"Unsupported Sentinel-2 platform {platform!r}. "
            f"Supported platforms: {supported}."
        ) from error


def normalize_sentinel2_band_name(band_name: str) -> str:
    """Return a canonical MSI band name such as ``B02`` or ``B8A``."""

    name = band_name.strip().upper()
    if not name.startswith("B"):
        name = f"B{name}"

    if name != "B8A" and name[1:].isdigit():
        name = f"B{int(name[1:]):02d}"

    if name not in SENTINEL2_NATIVE_RESOLUTION_M:
        valid = ", ".join(SENTINEL2_BAND_NAMES)
        raise ValueError(f"Unknown Sentinel-2 band {band_name!r}. Valid bands: {valid}.")

    return name


@dataclass(frozen=True)
class Sentinel2Band:
    """Spectral and spatial definition of one Sentinel-2 MSI band."""

    platform: str
    name: str
    native_resolution_m: int
    py6s_filter_name: str

    @property
    def py6s_filter(self):
        """Return the Py6S predefined spectral response tuple."""

        from Py6S import PredefinedWavelengths

        try:
            return getattr(PredefinedWavelengths, self.py6s_filter_name)
        except AttributeError as error:
            raise RuntimeError(
                f"Py6S does not provide {self.py6s_filter_name}. "
                "REDRESS Sentinel-2 support requires Py6S 1.9.2 or newer."
            ) from error

    @property
    def wavelength_nm(self) -> np.ndarray:
        """Wavelength coordinates of the Py6S response function, in nm."""

        _, lower_um, upper_um, response = self.py6s_filter
        return np.linspace(lower_um * 1000.0, upper_um * 1000.0,
                           num=len(response), dtype=np.float64)

    @property
    def spectral_response(self) -> np.ndarray:
        """Dimensionless relative spectral response."""

        _, lower_um, upper_um, response = self.py6s_filter
        response = np.asarray(response, dtype=np.float64)
        expected_size = round((upper_um - lower_um) / 0.0025) + 1

        if response.ndim != 1 or response.size != expected_size:
            raise ValueError(
                f"Invalid spectral response for {self.platform} {self.name}: "
                f"expected {expected_size} samples, found {response.size}."
            )
        if not np.all(np.isfinite(response)) or np.any(response < 0):
            raise ValueError(
                f"Invalid spectral response values for {self.platform} {self.name}."
            )

        return response.copy()

    @property
    def effective_wavelength_nm(self) -> float:
        """Response-weighted central wavelength, in nm."""

        wavelength_nm = self.wavelength_nm
        response = self.spectral_response
        response_integral = np.trapezoid(response, wavelength_nm)

        if response_integral <= 0:
            raise ValueError(
                f"The spectral response integral is zero for "
                f"{self.platform} {self.name}."
            )

        weighted_integral = np.trapezoid(
            wavelength_nm * response, wavelength_nm
        )
        return float(weighted_integral / response_integral)

    def integrate_spectral_values(
        self,
        values: np.ndarray,
        axis: int = 0,
    ) -> np.ndarray:
        """Integrate spectral values using the MSI response function.

        Parameters
        ----------
        values
            Values evaluated at :attr:`wavelength_nm`. The length of
            ``axis`` must equal the number of response samples.
        axis
            Spectral axis in ``values``.
        """

        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 0:
            raise ValueError("values must contain a spectral axis")
        if not -values.ndim <= axis < values.ndim:
            raise ValueError(
                f"axis {axis} is invalid for an array with "
                f"{values.ndim} dimensions"
            )
        axis %= values.ndim
        response = self.spectral_response

        if values.shape[axis] != response.size:
            raise ValueError(
                f"Spectral axis for {self.platform} {self.name} contains "
                f"{values.shape[axis]} samples; expected {response.size}."
            )

        response_shape = [1] * values.ndim
        response_shape[axis] = response.size
        weighted_values = values * response.reshape(response_shape)

        numerator = np.trapezoid(
            weighted_values, self.wavelength_nm, axis=axis
        )
        denominator = np.trapezoid(response, self.wavelength_nm)
        return numerator / denominator


def sentinel2_bands(
    platform: str,
    band_names: Iterable[str] | None = None,
) -> tuple[Sentinel2Band, ...]:
    """Build ordered MSI band definitions for one Sentinel-2 platform."""

    platform = normalize_sentinel2_platform(platform)
    if band_names is None:
        names = SENTINEL2_BAND_NAMES
    else:
        names = tuple(normalize_sentinel2_band_name(name) for name in band_names)

    if not names:
        raise ValueError("Sentinel-2 band selection cannot be empty.")
    if len(names) != len(set(names)):
        raise ValueError("Sentinel-2 band selection contains duplicate bands.")

    definitions = []
    for name in names:
        suffix = name[1:]
        filter_name = f"{platform}_MSI_{suffix}"
        definitions.append(
            Sentinel2Band(
                platform=platform,
                name=name,
                native_resolution_m=SENTINEL2_NATIVE_RESOLUTION_M[name],
                py6s_filter_name=filter_name,
            )
        )

    return tuple(definitions)


def sentinel2_metadata(
    platform: str,
    band_names: Iterable[str] | None = None,
) -> dict:
    """Return Sentinel-2 metadata accepted by :class:`products.Model`.

    The numeric ``wavelengths`` entry is retained for compatibility and for
    concise labels. The forward model uses ``band_definitions`` to access the
    full spectral response of each band.
    """

    definitions = sentinel2_bands(platform, band_names)
    return {
        "sensor": "MSI",
        "platform": definitions[0].platform,
        "bandlist": [band.name for band in definitions],
        "wavelengths": [band.effective_wavelength_nm for band in definitions],
        "native_resolution_m": [
            band.native_resolution_m for band in definitions
        ],
        "band_definitions": definitions,
    }
