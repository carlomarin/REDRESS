#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vertical atmospheric-column models used by REDRESS and Py6S."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AtmosphericColumns:
    """Integrated atmospheric columns above one target elevation."""

    altitude_m: float
    water_vapour_g_cm2: float
    ozone_cm_atm: float
    aod550: float

    def validate(self) -> None:
        """Validate the scalar columns accepted by Py6S."""

        values = (
            self.altitude_m,
            self.water_vapour_g_cm2,
            self.ozone_cm_atm,
            self.aod550,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("atmospheric altitude and columns must be finite")
        if any(value < 0.0 for value in values[1:]):
            raise ValueError("atmospheric columns must be non-negative")


@dataclass(frozen=True)
class AtmosphericColumnProfile:
    """Scale integrated columns from a reference surface to target altitude.

    Water vapour and aerosol optical depth follow independent exponential
    profiles. Ozone is retained because its column is concentrated primarily
    in the stratosphere and is effectively insensitive to mountain relief.

    The reference columns must describe the atmosphere above
    reference_elevation_m. Values returned by at_altitude therefore describe
    the atmosphere above the requested 6S target, as required for an
    elevated-target simulation.
    """

    reference_elevation_m: float
    water_vapour_g_cm2: float
    ozone_cm_atm: float
    aod550: float
    water_scale_height_m: float = 2000.0
    aerosol_scale_height_m: float = 1500.0

    def __post_init__(self) -> None:
        """Validate reference columns and scale heights at construction."""

        values = (
            self.reference_elevation_m,
            self.water_vapour_g_cm2,
            self.ozone_cm_atm,
            self.aod550,
            self.water_scale_height_m,
            self.aerosol_scale_height_m,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("atmospheric profile parameters must be finite")
        if any(value < 0.0 for value in values[1:4]):
            raise ValueError("reference atmospheric columns must be non-negative")
        if self.water_scale_height_m <= 0.0:
            raise ValueError("water scale height must be greater than zero")
        if self.aerosol_scale_height_m <= 0.0:
            raise ValueError("aerosol scale height must be greater than zero")

    def at_altitude(self, altitude_m: float) -> AtmosphericColumns:
        """Return Py6S-ready columns above one target altitude."""

        altitude_m = float(altitude_m)
        if not np.isfinite(altitude_m):
            raise ValueError("target altitude must be finite")
        elevation_difference_m = altitude_m - self.reference_elevation_m
        columns = AtmosphericColumns(
            altitude_m=altitude_m,
            water_vapour_g_cm2=self.water_vapour_g_cm2 * np.exp(
                -elevation_difference_m / self.water_scale_height_m
            ),
            ozone_cm_atm=self.ozone_cm_atm,
            aod550=self.aod550 * np.exp(
                -elevation_difference_m / self.aerosol_scale_height_m
            ),
        )
        columns.validate()
        return columns

    def at_altitudes(self, altitude_m) -> dict[str, np.ndarray]:
        """Return columns for an array of target-altitude nodes."""

        altitude = np.asarray(altitude_m, dtype=np.float64)
        if not np.all(np.isfinite(altitude)):
            raise ValueError("target altitude nodes must be finite")
        elevation_difference_m = altitude - self.reference_elevation_m
        return {
            "altitude_m": altitude.copy(),
            "water_vapour_g_cm2": self.water_vapour_g_cm2 * np.exp(
                -elevation_difference_m / self.water_scale_height_m
            ),
            "ozone_cm_atm": np.full_like(altitude, self.ozone_cm_atm),
            "aod550": self.aod550 * np.exp(
                -elevation_difference_m / self.aerosol_scale_height_m
            ),
        }
