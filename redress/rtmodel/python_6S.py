#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
from collections.abc import Mapping

import numpy as np
import Py6S as ps


def _aerosol_profile(aerosol):
    """Build a predefined or four-component Py6S aerosol profile."""

    predefined = {
        "Continental": ps.AeroProfile.Continental,
        "BiomassBurning": ps.AeroProfile.BiomassBurning,
        "Desert": ps.AeroProfile.Desert,
        "Maritime": ps.AeroProfile.Maritime,
        "Stratospheric": ps.AeroProfile.Stratospheric,
        "Urban": ps.AeroProfile.Urban,
        "None": ps.AeroProfile.NoAerosols,
    }
    if isinstance(aerosol, str):
        try:
            return ps.AeroProfile.PredefinedType(predefined[aerosol])
        except KeyError as error:
            valid = ", ".join(predefined)
            raise ValueError(
                f"Unknown aerosol profile {aerosol!r}. Valid profiles: {valid}."
            ) from error

    if isinstance(aerosol, Mapping):
        components = {name: float(value) for name, value in aerosol.items()}
        valid_components = {"dust", "water", "oceanic", "soot"}
        unexpected = set(components).difference(valid_components)
        if unexpected:
            raise ValueError(
                "Unknown aerosol components: " + ", ".join(sorted(unexpected))
            )
        if not np.isclose(sum(components.values()), 1.0, atol=1e-8):
            raise ValueError("aerosol component fractions must sum to one")
        if any(value < 0.0 for value in components.values()):
            raise ValueError("aerosol component fractions cannot be negative")
        return ps.AeroProfile.User(**components)

    raise TypeError("aero must be a profile name or component mapping")


class rtmodel(object):
    """
    """
    def __init__(self):
        self.outputs = None

    def run(self, sza, saa, vza, vaa, wavelength, alt, aero,
            aod=0.1, refl=0.99, water=0.05, ozone=0.30, atmo=None, atcor=False,
            month=None, day=None,
            ):
        """Run Py6S for one wavelength or one spectral response.

        ``wavelength`` may be a scalar wavelength in nanometers or a Py6S
        predefined wavelength tuple. Angles are expressed in radians and
        altitude in meters.
        """

        alt_km = alt / 1000

        # Generate a 6S class
        s = ps.SixS()

        # Add geometry
        s.geometry = ps.Geometry.User()

        # Convert angles in radians to degrees
        s.geometry.solar_z = np.rad2deg(sza)
        s.geometry.view_z = np.rad2deg(vza)
        s.geometry.solar_a = np.rad2deg(saa)
        s.geometry.view_a = np.rad2deg(vaa)

        if (month is None) != (day is None):
            raise ValueError("month and day must be provided together")
        if month is not None:
            s.geometry.month = int(month)
            s.geometry.day = int(day)

        # Set altitudes
        s.altitudes = ps.Altitudes()
        s.altitudes.set_target_custom_altitude(alt_km)
        s.altitudes.set_sensor_satellite_level()

        if np.isscalar(wavelength):
            s.wavelength = ps.Wavelength(float(wavelength) / 1000.0)
        else:
            try:
                s.wavelength = ps.Wavelength(wavelength)
            except (TypeError, ValueError, IndexError) as error:
                raise TypeError(
                    "wavelength must be a scalar value in nanometers or a "
                    "Py6S predefined wavelength tuple"
                ) from error

        # Atmosphere parameters
        if atmo is None:

            # If no standard atmospheric profile is specified, use water and
            # ozone.
            s.atmos_profile = ps.AtmosProfile.UserWaterAndOzone(water, ozone)
        else:

            # Build atmo dictionnary
            atmo_profiles = {
                "Mid_lat_summer": ps.AtmosProfile.MidlatitudeSummer,
                "Mid_lat_winter": ps.AtmosProfile.MidlatitudeWinter,
                "Sub_arctic_summer": ps.AtmosProfile.SubarcticSummer,
                "Sub_arctic_winter": ps.AtmosProfile.SubarcticWinter,
                "Tropical": ps.AtmosProfile.Tropical,
                "None": ps.AtmosProfile.NoGaseousAbsorption,
                }
            # Run a standard atmospheric profile
            s.atmos_profile = ps.AtmosProfile.PredefinedType(
                atmo_profiles[atmo])

        # Aerosol parameters
        s.aero_profile = _aerosol_profile(aero)

        s.aot550 = aod

        # According to switch, perform atmospheric correction or not
        if atcor:
            s.atmos_corr = ps.AtmosCorr.AtmosCorrLambertianFromReflectance(
                refl)
        else:
            s.ground_reflectance = ps.GroundReflectance.HomogeneousLambertian(
                refl
            )

        s.run()  # Run Py6S

        self.outputs = s.outputs

