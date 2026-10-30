#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot Sentinel-2 TOA reflectance for user-defined terrain cases.

The experiment follows the simplified formulations in Sect. 2.2 of
Lamare et al. (2020). REDRESS case 1 represents flat terrain. REDRESS case 2
represents a slope using the local illumination and viewing geometry and the
sky-view factor. Setting ``shadowed=True`` makes the direct-beam visibility
``b`` equal to zero.

Edit ``TERRAIN_CASES`` and the common physical parameters below, then run:

``python examples/illustrate_sentinel2_terrain_transition.py``
"""

from calendar import month_name
from dataclasses import dataclass
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt

from redress.lut import (
    Sentinel2EndmemberScenario,
    create_sentinel2_toa_lut,
)


# ---------------------------------------------------------------------------
# Parameters to edit
# ---------------------------------------------------------------------------

PLATFORM = "S2B"
BANDS = (
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
)

MONTH = 3
DAY = 28

# Scene-level angles used by 6S. These do not include terrain orientation.
SOLAR_ZENITH_ANGLE_DEG = 45.0
VIEW_ZENITH_ANGLE_DEG = 0.0

# Snow and atmosphere shared by every terrain case.
SPECIFIC_SURFACE_AREA_M2_KG = 20.0
BLACK_CARBON_NG_G = 0.0
DUST_UG_G = 0.0
DUST_DENSITY_KG_M3 = 2600.0
DUST_MODEL = "muller2011"
ALTITUDE_M = 2000.0
AOD550 = 0.10
WATER_VAPOUR_G_CM2 = 1.0
OZONE_CM_ATM = 0.30
AEROSOL_MODEL = "Continental"
ATMOSPHERIC_PROFILE = None

OUTPUT_FIGURE = Path(
    "outputs/sentinel2_terrain_transition/"
    "sentinel2_terrain_transition.png"
)
SHOW_FIGURE = False


@dataclass(frozen=True)
class TerrainCase:
    """Geometry and illumination state of one spectrum.

    ``relative_azimuth_angle_deg`` is the unsigned geometric relative azimuth
    in the local slope frame. ``sky_view_factor`` is Vd. REDRESS uses
    ``terrain_configuration_factor = 1 - Vd`` for these synthetic cases.
    """

    name: str
    label: str
    local_incidence_angle_deg: float
    local_view_zenith_angle_deg: float
    relative_azimuth_angle_deg: float
    sky_view_factor: float
    shadowed: bool
    redress_case: int
    color: str


TERRAIN_CASES = (
    TerrainCase(
        name="flat",
        label="Flat terrain",
        local_incidence_angle_deg=45.0,
        local_view_zenith_angle_deg=0.0,
        relative_azimuth_angle_deg=0.0,
        sky_view_factor=1.00,
        shadowed=False,
        redress_case=1,
        color="#30343b",
    ),
    TerrainCase(
        name="sun_facing",
        label="Sun-facing slope",
        local_incidence_angle_deg=20.0,
        local_view_zenith_angle_deg=25.0,
        relative_azimuth_angle_deg=180.0,
        sky_view_factor=0.91,
        shadowed=False,
        redress_case=2,
        color="#d95f02",
    ),
    TerrainCase(
        name="away_facing",
        label="Away-facing slope",
        local_incidence_angle_deg=70.0,
        local_view_zenith_angle_deg=25.0,
        relative_azimuth_angle_deg=0.0,
        sky_view_factor=0.91,
        shadowed=False,
        redress_case=2,
        color="#1b9e77",
    ),
    TerrainCase(
        name="cast_shadow",
        label="Cast shadow",
        local_incidence_angle_deg=70.0,
        local_view_zenith_angle_deg=25.0,
        relative_azimuth_angle_deg=0.0,
        sky_view_factor=0.60,
        shadowed=True,
        redress_case=2,
        color="#5e4fa2",
    ),
)


# ---------------------------------------------------------------------------
# Simulation and plot
# ---------------------------------------------------------------------------

def build_scenario(terrain_case):
    """Translate an editable terrain case into a REDRESS scenario."""

    return Sentinel2EndmemberScenario(
        name=terrain_case.name,
        category="Terrain transition",
        specific_surface_area=SPECIFIC_SURFACE_AREA_M2_KG,
        black_carbon_ng_g=BLACK_CARBON_NG_G,
        dust_ug_g=DUST_UG_G,
        dust_density_kg_m3=DUST_DENSITY_KG_M3,
        dust_model=DUST_MODEL,
        solar_zenith_angle=SOLAR_ZENITH_ANGLE_DEG,
        local_incidence_angle=terrain_case.local_incidence_angle_deg,
        view_zenith_angle=VIEW_ZENITH_ANGLE_DEG,
        local_view_zenith_angle=(
            terrain_case.local_view_zenith_angle_deg
        ),
        relative_azimuth_angle=(
            terrain_case.relative_azimuth_angle_deg
        ),
        altitude_m=ALTITUDE_M,
        aod550=AOD550,
        water_vapour_g_cm2=WATER_VAPOUR_G_CM2,
        ozone_cm_atm=OZONE_CM_ATM,
        sky_view_factor=terrain_case.sky_view_factor,
        terrain_configuration_factor=(
            1.0 - terrain_case.sky_view_factor
        ),
        shadowed=terrain_case.shadowed,
    )


def simulate_terrain_case(terrain_case):
    """Simulate one terrain case and return its Sentinel-2 LUT row."""

    print(f"Simulating {terrain_case.label}")
    return create_sentinel2_toa_lut(
        platform=PLATFORM,
        band_names=BANDS,
        scenarios=(build_scenario(terrain_case),),
        month=MONTH,
        day=DAY,
        aerosol_model=AEROSOL_MODEL,
        atmospheric_profile=ATMOSPHERIC_PROFILE,
        case_number=terrain_case.redress_case,
    )


def legend_label(terrain_case):
    """Describe the parameters that distinguish a plotted spectrum."""

    beam_visibility = 0 if terrain_case.shadowed else 1
    return (
        f"{terrain_case.label}: "
        f"i={terrain_case.local_incidence_angle_deg:g}°, "
        f"v={terrain_case.local_view_zenith_angle_deg:g}°, "
        f"RAA_geom={terrain_case.relative_azimuth_angle_deg:g}°, "
        f"Vd={terrain_case.sky_view_factor:.2f}, "
        f"b={beam_visibility}"
    )


def figure_caption():
    """Describe all fixed physical assumptions shown in the figure."""

    snow_description = (
        f"snow SSA={SPECIFIC_SURFACE_AREA_M2_KG:g} m² kg⁻¹, "
        f"black carbon={BLACK_CARBON_NG_G:g} ng g⁻¹ and "
        f"dust={DUST_UG_G:g} µg g⁻¹"
    )
    if DUST_UG_G > 0:
        snow_description += (
            f" ({DUST_MODEL}, density={DUST_DENSITY_KG_M3:g} kg m⁻³)"
        )

    if ATMOSPHERIC_PROFILE is None:
        gas_description = (
            f"water vapour={WATER_VAPOUR_G_CM2:g} g cm⁻² and "
            f"ozone={OZONE_CM_ATM:g} cm-atm"
        )
    else:
        gas_description = f"gaseous profile={ATMOSPHERIC_PROFILE}"

    caption = (
        f"{PLATFORM} MSI simulation for {DAY} {month_name[MONTH]} at "
        f"{ALTITUDE_M:g} m: {snow_description}. Atmosphere: "
        f"{AEROSOL_MODEL} aerosol, AOD₅₅₀={AOD550:g}, "
        f"{gas_description}. The flat spectrum uses REDRESS case 1; "
        "sloping spectra use the simplified case 2 of Lamare et al. "
        "(2020, Sect. 2.2). Vd and direct-beam visibility b are reported "
        "in the legend; prescribed surroundings satisfy Ct=1−Vd. Terrain "
        "reillumination and atmosphere–terrain coupling are excluded."
    )
    return textwrap.fill(caption, width=145)


def plot_reflectance(simulations):
    """Plot only the TOA-reflectance spectra for the configured cases."""

    figure, axis = plt.subplots(figsize=(10.0, 7.2))

    for terrain_case, simulation in zip(TERRAIN_CASES, simulations):
        axis.plot(
            simulation["wavelength_nm"].values,
            simulation["toa_reflectance"].squeeze().values,
            marker="o",
            markersize=4.5,
            linewidth=2.0,
            color=terrain_case.color,
            label=legend_label(terrain_case),
        )

    axis.set_title(
        "Sentinel-2 snow TOA reflectance from flat terrain to shadow\n"
        f"global SZA={SOLAR_ZENITH_ANGLE_DEG:g}°, "
        f"global VZA={VIEW_ZENITH_ANGLE_DEG:g}°"
    )
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel("TOA reflectance factor")
    axis.set_ylim(0.0, 1.30)
    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8.5)

    figure.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.86,
        bottom=0.25,
    )
    figure.text(
        0.10,
        0.035,
        figure_caption(),
        ha="left",
        va="bottom",
        fontsize=8.5,
        linespacing=1.35,
    )
    return figure


def main():
    """Run all configured cases and save the single spectral figure."""

    simulations = [
        simulate_terrain_case(terrain_case)
        for terrain_case in TERRAIN_CASES
    ]
    figure = plot_reflectance(simulations)

    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    print(f"Figure: {OUTPUT_FIGURE}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
