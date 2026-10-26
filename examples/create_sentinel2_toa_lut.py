#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a diagnostic Sentinel-2 TOA snow endmember LUT with REDRESS.

The experiment changes one physical factor at a time around a common baseline.
This makes the spectra easier to interpret and avoids an unnecessarily large
Cartesian LUT during the first scientific checks.
"""

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from redress.lut import (
    Sentinel2EndmemberScenario,
    create_sentinel2_toa_lut,
)


# ---------------------------------------------------------------------------
# Sensor, acquisition and output configuration
# ---------------------------------------------------------------------------

PLATFORM = "S2B"
BANDS = (
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
)

# Example date corresponding to the Peteroa acquisition used in earlier tests.
MONTH = 3
DAY = 28

AEROSOL_MODEL = "Continental"
ATMOSPHERIC_PROFILE = None
REDRESS_CASE_NUMBER = 5
SCENE_SIZE = 9

OUTPUT_DIRECTORY = Path("outputs/sentinel2_toa_lut")


# ---------------------------------------------------------------------------
# Baseline and one-factor-at-a-time scenarios
# ---------------------------------------------------------------------------

BASELINE = Sentinel2EndmemberScenario(
    name="baseline",
    category="Baseline",
    specific_surface_area=20.0,
    solar_zenith_angle=45.0,
    local_incidence_angle=45.0,
    view_zenith_angle=0.0,
    local_view_zenith_angle=0.0,
    relative_azimuth_angle=0.0,
    altitude_m=2000.0,
    aod550=0.10,
    water_vapour_g_cm2=1.0,
    ozone_cm_atm=0.30,
    sky_view_factor=1.0,
    terrain_configuration_factor=0.0,
    shadowed=False,
)

SCENARIOS = (
    BASELINE,
    replace(BASELINE, name="ssa_5", category="Snow SSA",
            specific_surface_area=5.0),
    replace(BASELINE, name="ssa_60", category="Snow SSA",
            specific_surface_area=60.0),
    replace(BASELINE, name="sza_20", category="Solar zenith",
            solar_zenith_angle=20.0, local_incidence_angle=20.0),
    replace(BASELINE, name="sza_65", category="Solar zenith",
            solar_zenith_angle=65.0, local_incidence_angle=65.0),
    replace(BASELINE, name="incidence_20", category="Local incidence",
            local_incidence_angle=20.0),
    replace(BASELINE, name="incidence_70", category="Local incidence",
            local_incidence_angle=70.0),
    replace(BASELINE, name="aod_003", category="Atmosphere", aod550=0.03),
    replace(BASELINE, name="aod_025", category="Atmosphere", aod550=0.25),
    replace(BASELINE, name="water_02", category="Atmosphere",
            water_vapour_g_cm2=0.2),
    replace(BASELINE, name="water_20", category="Atmosphere",
            water_vapour_g_cm2=2.0),
    replace(BASELINE, name="altitude_0", category="Altitude and terrain",
            altitude_m=0.0),
    replace(BASELINE, name="altitude_4000", category="Altitude and terrain",
            altitude_m=4000.0),
    replace(BASELINE, name="cast_shadow", category="Altitude and terrain",
            shadowed=True, sky_view_factor=0.70,
            terrain_configuration_factor=0.15),
    replace(BASELINE, name="terrain_reillumination",
            category="Altitude and terrain", sky_view_factor=0.85,
            terrain_configuration_factor=0.15),
    replace(BASELINE, name="view_15_raa_0", category="Viewing geometry",
            view_zenith_angle=15.0, local_view_zenith_angle=15.0,
            relative_azimuth_angle=0.0),
    replace(BASELINE, name="view_15_raa_90", category="Viewing geometry",
            view_zenith_angle=15.0, local_view_zenith_angle=15.0,
            relative_azimuth_angle=90.0),
    replace(BASELINE, name="view_15_raa_180", category="Viewing geometry",
            view_zenith_angle=15.0, local_view_zenith_angle=15.0,
            relative_azimuth_angle=180.0),
)

CATEGORIES = (
    "Snow SSA",
    "Solar zenith",
    "Local incidence",
    "Atmosphere",
    "Altitude and terrain",
    "Viewing geometry",
)


def _label(lut, scenario_name, category):
    """Describe the panel parameters using values stored in the LUT."""

    parameters = {
        "Snow SSA": (
            ("specific_surface_area", "SSA", "m² kg⁻¹"),
        ),
        "Solar zenith": (
            ("solar_zenith_angle", "SZA", "°"),
            ("local_incidence_angle", "Incidence", "°"),
        ),
        "Local incidence": (
            ("local_incidence_angle", "Incidence", "°"),
        ),
        "Atmosphere": (
            ("aod550", "AOD550", ""),
            ("water_vapour_g_cm2", "Water", "g cm⁻²"),
            ("ozone_cm_atm", "Ozone", "cm-atm"),
        ),
        "Altitude and terrain": (
            ("altitude_m", "Altitude", "m"),
            ("sky_view_factor", "Sky view", ""),
            ("terrain_configuration_factor", "Terrain factor", ""),
            ("shadowed", "Shadow", ""),
        ),
        "Viewing geometry": (
            ("view_zenith_angle", "VZA", "°"),
            ("local_view_zenith_angle", "Local VZA", "°"),
            ("relative_azimuth_angle", "RAA", "°"),
        ),
    }

    descriptions = []
    for coordinate, label, unit in parameters[category]:
        value = lut[coordinate].sel(scenario=scenario_name).item()
        baseline_value = lut[coordinate].sel(scenario="baseline").item()
        if scenario_name != "baseline" and value == baseline_value:
            continue
        if coordinate == "shadowed":
            description = f"{label}={'yes' if value else 'no'}"
        else:
            spacing = " " if unit and unit != "°" else ""
            description = f"{label}={value:g}{spacing}{unit}"
        descriptions.append(description)

    # Wrap longer legends to keep the spectra visible.
    lines = [
        ", ".join(descriptions[index:index + 2])
        for index in range(0, len(descriptions), 2)
    ]
    prefix = "Baseline: " if scenario_name == "baseline" else ""
    return prefix + "\n".join(lines)


def plot_toa_sensitivity(lut, output_file):
    """Plot TOA-reflectance spectra grouped by physical process."""

    wavelength = lut["wavelength_nm"].values
    categories = lut["category"].values

    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(16, 9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for axis, category in zip(axes.flat, CATEGORIES):
        scenario_names = ["baseline"] + [
            name
            for name, scenario_category in zip(lut["scenario"].values,
                                                categories)
            if scenario_category == category
        ]
        colors = plt.cm.viridis(
            np.linspace(0.1, 0.9, len(scenario_names))
        )
        for scenario_name, color in zip(scenario_names, colors):
            spectrum = lut["toa_reflectance"].sel(scenario=scenario_name)
            linewidth = 2.4 if scenario_name == "baseline" else 1.5
            axis.plot(
                wavelength,
                spectrum,
                marker="o",
                linewidth=linewidth,
                color="black" if scenario_name == "baseline" else color,
                label=_label(lut, scenario_name, category),
            )

        axis.set_title(category)
        axis.grid(alpha=0.25)
        axis.set_ylim(0, 1.3)
        axis.legend(fontsize=8)

    for axis in axes[-1, :]:
        axis.set_xlabel("Wavelength (nm)")
    for axis in axes[:, 0]:
        axis.set_ylabel("TOA reflectance")

    figure.suptitle(
        f"REDRESS {lut.attrs['platform']} TOA snow endmember sensitivity",
        fontsize=15,
    )
    figure.savefig(output_file, dpi=250, bbox_inches="tight")
    plt.close(figure)


def main():
    """Generate, save and plot the diagnostic LUT."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    lut = create_sentinel2_toa_lut(
        platform=PLATFORM,
        band_names=BANDS,
        scenarios=SCENARIOS,
        month=MONTH,
        day=DAY,
        aerosol_model=AEROSOL_MODEL,
        atmospheric_profile=ATMOSPHERIC_PROFILE,
        case_number=REDRESS_CASE_NUMBER,
        scene_size=SCENE_SIZE,
    )

    netcdf_file = OUTPUT_DIRECTORY / f"{PLATFORM}_toa_endmember_lut.nc"
    csv_file = OUTPUT_DIRECTORY / f"{PLATFORM}_toa_endmember_lut.csv"
    figure_file = OUTPUT_DIRECTORY / f"{PLATFORM}_toa_sensitivity.png"

    encoding = {
        variable: {"dtype": "float32", "zlib": True, "complevel": 4}
        for variable in lut.data_vars
    }
    lut.to_netcdf(netcdf_file, encoding=encoding)
    lut.to_dataframe().reset_index().to_csv(csv_file, index=False)
    plot_toa_sensitivity(lut, figure_file)

    print(lut)
    print(f"NetCDF LUT: {netcdf_file.resolve()}")
    print(f"CSV LUT:    {csv_file.resolve()}")
    print(f"Figure:     {figure_file.resolve()}")


if __name__ == "__main__":
    main()
