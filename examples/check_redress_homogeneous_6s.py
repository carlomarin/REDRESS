#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare REDRESS and 6S over homogeneous Lambertian bright ground.

The configuration removes DEM and BRDF effects. REDRESS reconstructs the
surface-atmosphere coupling from atmosphere-only 6S quantities, while the
reference 6S run calculates the coupled homogeneous-ground signal directly.
"""

import numpy as np
import xarray as xr

from redress.rtmodel.python_6S import rtmodel
from redress.topography.forward_model import Cases, iterative_radiance


GROUND_REFLECTANCE = 0.9
SOLAR_ZENITH_DEGREE = 45.0
SOLAR_AZIMUTH_DEGREE = 0.0
VIEW_ZENITH_DEGREE = 0.0
VIEW_AZIMUTH_DEGREE = 0.0
WAVELENGTH_NM = 560.0
ALTITUDE_M = 2000.0
AEROSOL_MODEL = "Continental"
AOD550 = 0.10
WATER_VAPOUR_G_CM2 = 1.0
OZONE_CM_ATM = 0.30
SCENE_SIZE = 9
RELATIVE_TOLERANCE = 0.01


def radiative_transfer_options():
    """Return the common atmospheric configuration."""

    return {
        "aerosol_model": AEROSOL_MODEL,
        "aod": AOD550,
        "refl": 0.0,
        "water": WATER_VAPOUR_G_CM2,
        "ozone": OZONE_CM_ATM,
        "atmo_model": None,
        "atcor": False,
    }


def flat_scene():
    """Build a uniform horizontal scene with an unobstructed sky."""

    shape = (SCENE_SIZE, SCENE_SIZE)
    solar_zenith = np.deg2rad(SOLAR_ZENITH_DEGREE)
    view_zenith = np.deg2rad(VIEW_ZENITH_DEGREE)
    zeros = np.zeros(shape)
    angles = xr.Dataset({
        "SZA": (("x", "y"), np.full(shape, solar_zenith)),
        "SAA": (("x", "y"), np.full(
            shape, np.deg2rad(SOLAR_AZIMUTH_DEGREE)
        )),
        "VZA": (("x", "y"), np.full(shape, view_zenith)),
        "VAA": (("x", "y"), np.full(
            shape, np.deg2rad(VIEW_AZIMUTH_DEGREE)
        )),
    })
    topography = xr.Dataset({
        "altitude": (("x", "y"), np.full(shape, ALTITUDE_M)),
        "eff_sza": (("x", "y"), np.full(shape, solar_zenith)),
        "eff_vza": (("x", "y"), np.full(shape, view_zenith)),
        "vt": (("x", "y"), np.ones(shape)),
        "ct": (("x", "y"), zeros.copy()),
        "all_shadows": (("x", "y"), np.ones(shape)),
    })
    return angles, topography


def direct_6s_reflectance():
    """Run 6S with the bright Lambertian ground included internally."""

    model = rtmodel()
    model.run(
        sza=np.deg2rad(SOLAR_ZENITH_DEGREE),
        saa=np.deg2rad(SOLAR_AZIMUTH_DEGREE),
        vza=np.deg2rad(VIEW_ZENITH_DEGREE),
        vaa=np.deg2rad(VIEW_AZIMUTH_DEGREE),
        wavelength=WAVELENGTH_NM,
        alt=ALTITUDE_M,
        aero=AEROSOL_MODEL,
        aod=AOD550,
        refl=GROUND_REFLECTANCE,
        water=WATER_VAPOUR_G_CM2,
        ozone=OZONE_CM_ATM,
        atcor=False,
    )
    return model.outputs.apparent_reflectance


def redress_reflectance():
    """Reconstruct the coupled TOA signal with REDRESS."""

    angles, topography = flat_scene()
    shape = topography["altitude"].shape
    case = Cases()
    case.create_cases(5)
    _, reflectance = iterative_radiance(
        topo_bands=topography,
        angles=angles,
        wavelength=WAVELENGTH_NM,
        hdr=np.full(shape, GROUND_REFLECTANCE),
        bhr=np.full(shape, GROUND_REFLECTANCE),
        rt_model=rtmodel(),
        rt_options=radiative_transfer_options(),
        brf=np.full(shape, GROUND_REFLECTANCE),
        case=case,
        dif_anis=False,
        return_reflectance=True,
        verbose=False,
    )
    center = SCENE_SIZE // 2
    return float(reflectance[center, center])


def main():
    """Run and report the homogeneous-ground comparison."""

    reference = direct_6s_reflectance()
    simulated = redress_reflectance()
    relative_difference = (simulated - reference) / reference

    print(f"Direct 6S TOA reflectance: {reference:.8f}")
    print(f"REDRESS TOA reflectance:   {simulated:.8f}")
    print(f"Relative difference:       {relative_difference:.3%}")

    if not np.isclose(
        simulated,
        reference,
        rtol=RELATIVE_TOLERANCE,
        atol=1e-5,
    ):
        raise AssertionError(
            "REDRESS and homogeneous-ground 6S differ by more than "
            f"{RELATIVE_TOLERANCE:.1%}"
        )


if __name__ == "__main__":
    main()
