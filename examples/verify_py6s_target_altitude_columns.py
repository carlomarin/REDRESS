#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify user atmospheric columns at custom 6S target altitudes.

The experiment separates two effects:

1. fixed-column runs pass identical water, ozone and AOD values at every
   target altitude;
2. target-column runs construct water and AOD above every altitude node from
   one reference surface.

The 6S output is checked to ensure that the reported user columns match the
values supplied by Py6S. The resulting transmittances show the radiative
effect of using target-level columns instead of repeating one column over
the complete DEM elevation range.
"""

from pathlib import Path
import csv
import re

import numpy as np
from Py6S import (
    AeroProfile,
    Altitudes,
    AtmosProfile,
    Geometry,
    GroundReflectance,
    SixS,
    Wavelength,
)

from redress.rtmodel.atmosphere import AtmosphericColumnProfile


ALTITUDE_NODES_M = np.asarray([1000.0, 2000.0, 3000.0])
REFERENCE_PROFILE = AtmosphericColumnProfile(
    reference_elevation_m=1000.0,
    water_vapour_g_cm2=2.0,
    ozone_cm_atm=0.3,
    aod550=0.2,
    water_scale_height_m=2000.0,
    aerosol_scale_height_m=1500.0,
)
OUTPUT_CSV = Path("outputs/py6s_target_altitude_verification.csv")

SOLAR_ZENITH_DEG = 40.0
SOLAR_AZIMUTH_DEG = 150.0
VIEW_ZENITH_DEG = 5.0
VIEW_AZIMUTH_DEG = 20.0
MONTH = 7
DAY = 15


def _reported_user_column(fulltext, constituent):
    """Extract a user water or ozone column from the textual 6S output."""

    patterns = {
        "water": r"user defined water content\s*:\s*uh2o=\s*([0-9.Ee+-]+)",
        "ozone": r"user defined ozone content\s*:\s*uo3\s*=\s*([0-9.Ee+-]+)",
    }
    match = re.search(patterns[constituent], fulltext, flags=re.IGNORECASE)
    return np.nan if match is None else float(match.group(1))


def run_6s(altitude_m, water_g_cm2, ozone_cm_atm, aod550, wavelength_um):
    """Run one atmosphere-only 6S simulation."""

    model = SixS()
    model.geometry = Geometry.User()
    model.geometry.solar_z = SOLAR_ZENITH_DEG
    model.geometry.solar_a = SOLAR_AZIMUTH_DEG
    model.geometry.view_z = VIEW_ZENITH_DEG
    model.geometry.view_a = VIEW_AZIMUTH_DEG
    model.geometry.month = MONTH
    model.geometry.day = DAY

    model.altitudes = Altitudes()
    model.altitudes.set_target_custom_altitude(altitude_m / 1000.0)
    model.altitudes.set_sensor_satellite_level()
    model.atmos_profile = AtmosProfile.UserWaterAndOzone(
        water_g_cm2,
        ozone_cm_atm,
    )
    model.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    model.aot550 = aod550
    model.wavelength = Wavelength(wavelength_um)
    model.ground_reflectance = (
        GroundReflectance.HomogeneousLambertian(0.0)
    )
    model.run()
    return model.outputs


def evaluate_case(mode, altitude_m):
    """Evaluate fixed or altitude-adjusted columns at one node."""

    if mode == "fixed":
        columns = REFERENCE_PROFILE.at_altitude(
            REFERENCE_PROFILE.reference_elevation_m
        )
    elif mode == "target":
        columns = REFERENCE_PROFILE.at_altitude(altitude_m)
    else:
        raise ValueError(f"unknown column mode {mode!r}")

    visible = run_6s(
        altitude_m,
        columns.water_vapour_g_cm2,
        columns.ozone_cm_atm,
        columns.aod550,
        wavelength_um=0.55,
    )
    water_band = run_6s(
        altitude_m,
        columns.water_vapour_g_cm2,
        columns.ozone_cm_atm,
        columns.aod550,
        wavelength_um=0.94,
    )
    ozone_band = run_6s(
        altitude_m,
        columns.water_vapour_g_cm2,
        columns.ozone_cm_atm,
        columns.aod550,
        wavelength_um=0.60,
    )

    return {
        "mode": mode,
        "altitude_m": altitude_m,
        "water_input_g_cm2": columns.water_vapour_g_cm2,
        "water_reported_g_cm2": _reported_user_column(
            visible.fulltext, "water"
        ),
        "ozone_input_cm_atm": columns.ozone_cm_atm,
        "ozone_reported_cm_atm": _reported_user_column(
            visible.fulltext, "ozone"
        ),
        "aod550_input": columns.aod550,
        "aod550_reported": visible.aot550,
        "ground_altitude_reported_km": visible.ground_altitude,
        "water_transmittance_0940": (
            water_band.transmittance_water.total
        ),
        "ozone_transmittance_0600": (
            ozone_band.transmittance_ozone.total
        ),
        "aerosol_optical_depth_0550": (
            visible.optical_depth_total.aerosol
        ),
    }


def verify_reported_inputs(records):
    """Check that 6S reports the user columns passed at every node."""

    for record in records:
        np.testing.assert_allclose(
            record["ground_altitude_reported_km"],
            record["altitude_m"] / 1000.0,
            rtol=0.0,
            atol=5.0e-4,
        )
        if not np.isfinite(record["water_reported_g_cm2"]):
            raise AssertionError("6S did not report the user water column")
        if not np.isfinite(record["ozone_reported_cm_atm"]):
            raise AssertionError("6S did not report the user ozone column")
        np.testing.assert_allclose(
            record["water_reported_g_cm2"],
            record["water_input_g_cm2"],
            rtol=0.0,
            atol=5.0e-4,
        )
        np.testing.assert_allclose(
            record["ozone_reported_cm_atm"],
            record["ozone_input_cm_atm"],
            rtol=0.0,
            atol=5.0e-4,
        )
        np.testing.assert_allclose(
            record["aod550_reported"],
            record["aod550_input"],
            rtol=0.0,
            atol=5.0e-4,
        )


def write_results(records):
    """Write the controlled experiment to a reusable CSV table."""

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def main():
    """Run the verification and report the supplied and effective columns."""

    records = [
        evaluate_case(mode, altitude_m)
        for mode in ("fixed", "target")
        for altitude_m in ALTITUDE_NODES_M
    ]
    verify_reported_inputs(records)
    write_results(records)

    columns = (
        "mode",
        "altitude_m",
        "water_input_g_cm2",
        "aod550_input",
        "water_transmittance_0940",
        "aerosol_optical_depth_0550",
    )
    print(" | ".join(columns))
    for record in records:
        print(
            " | ".join(
                str(record[name])
                if name in ("mode", "altitude_m")
                else f"{record[name]:.6f}"
                for name in columns
            )
        )

    print(
        "\nVerification passed: 6S reported the user water, ozone and AOD "
        "columns supplied at every custom target altitude."
    )
    print(
        "The target-column rows therefore provide the appropriate REDRESS "
        "configuration for an elevation lookup."
    )
    print(f"CSV: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
