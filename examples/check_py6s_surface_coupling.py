#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check how Py6S separates atmospheric and environmental irradiance.

The experiment changes only the homogeneous Lambertian ground reflectance.
Atmosphere-only direct and diffuse irradiance should remain unchanged, while
the environmental irradiance should increase for the bright surface.
"""

import numpy as np

from redress.rtmodel.python_6S import rtmodel


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
GROUND_REFLECTANCES = (0.0, 0.9)


def run_case(ground_reflectance):
    """Run one 6S simulation and return its ground-level irradiances."""

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
        refl=ground_reflectance,
        water=WATER_VAPOUR_G_CM2,
        ozone=OZONE_CM_ATM,
        atcor=False,
    )
    outputs = model.outputs
    return {
        "ground_reflectance": ground_reflectance,
        "direct_solar_irradiance": outputs.direct_solar_irradiance,
        "diffuse_solar_irradiance": outputs.diffuse_solar_irradiance,
        "environmental_irradiance": outputs.environmental_irradiance,
    }


def main():
    """Run the reflectance comparison and verify the expected separation."""

    results = [run_case(reflectance) for reflectance in GROUND_REFLECTANCES]
    columns = tuple(results[0])
    print(",".join(columns))
    for result in results:
        print(",".join(f"{result[column]:.8f}" for column in columns))

    dark, bright = results
    for field in (
        "direct_solar_irradiance",
        "diffuse_solar_irradiance",
    ):
        if not np.isclose(dark[field], bright[field], rtol=1e-6, atol=1e-8):
            raise AssertionError(
                f"{field} changed with ground reflectance: "
                f"{dark[field]} versus {bright[field]}"
            )
    if (
        bright["environmental_irradiance"]
        <= dark["environmental_irradiance"]
    ):
        raise AssertionError(
            "environmental irradiance did not increase for bright ground"
        )

    print("6S separates atmospheric diffuse and environmental irradiance.")


if __name__ == "__main__":
    main()
