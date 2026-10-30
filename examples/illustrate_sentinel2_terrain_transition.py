#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare Sentinel-2 TOA snow reflectance across terrain and SSA.

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
from matplotlib.lines import Line2D
import numpy as np

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
SPECIFIC_SURFACE_AREA_LEVELS_M2_KG = (80.0, 40.0, 20.0, 10.0, 3.0)
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

VNIR_MAXIMUM_WAVELENGTH_NM = 1000.0
VNIR_REFLECTANCE_LIMITS = (0.0, 1.30)
SWIR_REFLECTANCE_LIMITS = (0.0, 0.15)

OUTPUT_FIGURE = Path(
    "outputs/sentinel2_terrain_transition/"
    "sentinel2_terrain_transition.png"
)
SHOW_FIGURE = False


@dataclass(frozen=True)
class TerrainCase:
    """Geometry and illumination state of one column in the figure.

    ``relative_azimuth_angle_deg`` is the unsigned geometric relative
    azimuth in the local slope frame. Zero means that the Sun and sensor are
    on the same side. ``sky_view_factor`` is Vd. Synthetic scenarios use
    ``terrain_configuration_factor = 1 - Vd``.
    """

    name: str
    label: str
    local_incidence_angle_deg: float
    local_view_zenith_angle_deg: float
    relative_azimuth_angle_deg: float
    sky_view_factor: float
    shadowed: bool
    redress_case: int


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
    ),
)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def build_scenario(terrain_case, specific_surface_area):
    """Build one REDRESS scenario for a terrain case and snow type."""

    return Sentinel2EndmemberScenario(
        name=(
            f"{terrain_case.name}_"
            f"ssa_{specific_surface_area:g}"
        ),
        category="Terrain and SSA transition",
        specific_surface_area=specific_surface_area,
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
    """Simulate every configured SSA level for one terrain geometry."""

    scenarios = tuple(
        build_scenario(terrain_case, specific_surface_area)
        for specific_surface_area in SPECIFIC_SURFACE_AREA_LEVELS_M2_KG
    )
    print(
        f"Simulating {terrain_case.label}: "
        f"{len(scenarios)} SSA levels"
    )
    return create_sentinel2_toa_lut(
        platform=PLATFORM,
        band_names=BANDS,
        scenarios=scenarios,
        month=MONTH,
        day=DAY,
        aerosol_model=AEROSOL_MODEL,
        atmospheric_profile=ATMOSPHERIC_PROFILE,
        case_number=terrain_case.redress_case,
    )


def simulate_all_terrain_cases():
    """Return one multi-SSA simulation for each terrain case."""

    return {
        terrain_case.name: simulate_terrain_case(terrain_case)
        for terrain_case in TERRAIN_CASES
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def terrain_header(terrain_case):
    """Return a self-contained geometry label for one panel column."""

    beam_visibility = 0 if terrain_case.shadowed else 1
    return (
        f"{terrain_case.label}\n"
        f"i={terrain_case.local_incidence_angle_deg:g}°, "
        f"v={terrain_case.local_view_zenith_angle_deg:g}°, "
        f"RAA_geom={terrain_case.relative_azimuth_angle_deg:g}°, "
        f"Vd={terrain_case.sky_view_factor:.2f}, "
        f"b={beam_visibility}"
    )


def figure_caption():
    """Describe the experiment and the interpretation of the ratio row."""

    ssa_values = ", ".join(
        f"{value:g}"
        for value in SPECIFIC_SURFACE_AREA_LEVELS_M2_KG[:-1]
    )
    ssa_values += (
        f" and {SPECIFIC_SURFACE_AREA_LEVELS_M2_KG[-1]:g}"
    )
    snow_description = (
        f"snow SSA={ssa_values} m² kg⁻¹, "
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
        f"{gas_description}. Global SZA={SOLAR_ZENITH_ANGLE_DEG:g}° and "
        f"VZA={VIEW_ZENITH_ANGLE_DEG:g}°. Flat terrain uses REDRESS case 1; "
        "slopes use the simplified case 2 of Lamare et al. (2020, "
        "Sect. 2.2), with prescribed Ct=1−Vd and without terrain "
        "reillumination or atmosphere–terrain coupling. The lower row is "
        "each spectrum divided band by band by flat terrain at the same SSA. "
        "Overlapping SSA colours indicate snow-independent terrain scaling; "
        "separation indicates that terrain-specific endmembers are required "
        "for each snow type."
    )
    return textwrap.fill(caption, width=225)


def _ssa_colors():
    """Assign perceptually uniform and consistent colours to SSA levels."""

    values = plt.colormaps["viridis"](
        np.linspace(
            0.08,
            0.92,
            len(SPECIFIC_SURFACE_AREA_LEVELS_M2_KG),
        )
    )
    return dict(zip(SPECIFIC_SURFACE_AREA_LEVELS_M2_KG, values))


def _common_ratio_limit(simulations):
    """Return a shared upper limit for all terrain-to-flat ratios."""

    flat = simulations["flat"]["toa_reflectance"].values
    maximum = 1.0
    for terrain_case in TERRAIN_CASES:
        reflectance = simulations[
            terrain_case.name
        ]["toa_reflectance"].values
        ratio = np.divide(
            reflectance,
            flat,
            out=np.full_like(reflectance, np.nan),
            where=flat != 0,
        )
        maximum = max(maximum, float(np.nanmax(ratio)))

    return max(1.20, np.ceil(maximum * 10.0) / 10.0 + 0.05)


def _format_spectral_axis(axis, spectral_region):
    """Apply shared styling and identify the spectral region."""

    axis.grid(alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.text(
        0.04,
        0.95,
        spectral_region,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color="0.30",
    )


def plot_reflectance(simulations):
    """Plot spectra and terrain-to-flat ratios for all SSA levels."""

    figure = plt.figure(figsize=(19.0, 10.5))
    outer_grid = figure.add_gridspec(
        2,
        len(TERRAIN_CASES),
        left=0.055,
        right=0.985,
        bottom=0.18,
        top=0.80,
        wspace=0.24,
        hspace=0.30,
    )

    wavelength = simulations["flat"]["wavelength_nm"].values
    vnir = wavelength < VNIR_MAXIMUM_WAVELENGTH_NM
    swir = ~vnir
    flat_reflectance = simulations["flat"]["toa_reflectance"].values
    ratio_upper_limit = _common_ratio_limit(simulations)
    colors = _ssa_colors()
    top_axis_pairs = []

    for column, terrain_case in enumerate(TERRAIN_CASES):
        top_grid = outer_grid[0, column].subgridspec(
            1, 2, width_ratios=(2.45, 1.0), wspace=0.13
        )
        ratio_grid = outer_grid[1, column].subgridspec(
            1, 2, width_ratios=(2.45, 1.0), wspace=0.13
        )
        top_vnir = figure.add_subplot(top_grid[0, 0])
        top_swir = figure.add_subplot(top_grid[0, 1])
        ratio_vnir = figure.add_subplot(ratio_grid[0, 0])
        ratio_swir = figure.add_subplot(
            ratio_grid[0, 1],
            sharey=ratio_vnir,
        )
        top_axis_pairs.append((top_vnir, top_swir, terrain_case))

        simulation = simulations[terrain_case.name]
        reflectance = simulation["toa_reflectance"].values
        ratio = np.divide(
            reflectance,
            flat_reflectance,
            out=np.full_like(reflectance, np.nan),
            where=flat_reflectance != 0,
        )

        for index, specific_surface_area in enumerate(
            SPECIFIC_SURFACE_AREA_LEVELS_M2_KG
        ):
            line_options = {
                "color": colors[specific_surface_area],
                "marker": "o",
                "markersize": 3.8,
                "linewidth": 1.8,
            }
            top_vnir.plot(
                wavelength[vnir],
                reflectance[index, vnir],
                **line_options,
            )
            top_swir.plot(
                wavelength[swir],
                reflectance[index, swir],
                **line_options,
            )
            ratio_vnir.plot(
                wavelength[vnir],
                ratio[index, vnir],
                **line_options,
            )
            ratio_swir.plot(
                wavelength[swir],
                ratio[index, swir],
                **line_options,
            )

        top_vnir.set_ylim(*VNIR_REFLECTANCE_LIMITS)
        top_swir.set_ylim(*SWIR_REFLECTANCE_LIMITS)
        ratio_vnir.set_ylim(0.0, ratio_upper_limit)
        ratio_swir.set_ylim(0.0, ratio_upper_limit)
        ratio_vnir.axhline(1.0, color="0.45", linewidth=0.8)
        ratio_swir.axhline(1.0, color="0.45", linewidth=0.8)

        vnir_padding = 20.0
        swir_padding = 70.0
        for axis in (top_vnir, ratio_vnir):
            axis.set_xlim(
                wavelength[vnir].min() - vnir_padding,
                wavelength[vnir].max() + vnir_padding,
            )
        for axis in (top_swir, ratio_swir):
            axis.set_xlim(
                wavelength[swir].min() - swir_padding,
                wavelength[swir].max() + swir_padding,
            )

        _format_spectral_axis(top_vnir, "VNIR")
        _format_spectral_axis(top_swir, "SWIR")
        _format_spectral_axis(ratio_vnir, "VNIR")
        _format_spectral_axis(ratio_swir, "SWIR")

        top_swir.yaxis.tick_right()
        top_swir.spines["left"].set_visible(False)
        top_swir.spines["right"].set_visible(True)
        top_swir.tick_params(axis="y", labelsize=8)
        ratio_swir.tick_params(axis="y", left=False, labelleft=False)
        ratio_swir.spines["left"].set_visible(False)

        top_vnir.tick_params(labelbottom=False)
        top_swir.tick_params(labelbottom=False)
        ratio_vnir.set_xlabel("Wavelength (nm)")
        ratio_swir.set_xlabel("Wavelength (nm)")

        if column > 0:
            top_vnir.tick_params(axis="y", labelleft=False)
            ratio_vnir.tick_params(axis="y", labelleft=False)
        if column == 0:
            top_vnir.set_ylabel("TOA reflectance factor")
            ratio_vnir.set_ylabel("Terrain / flat reflectance")

    figure.suptitle(
        "Sentinel-2 snow TOA reflectance across terrain and snow SSA",
        y=0.985,
        fontsize=16,
        fontweight="bold",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors[specific_surface_area],
            marker="o",
            markersize=5,
            linewidth=2,
            label=f"{specific_surface_area:g} m² kg⁻¹",
        )
        for specific_surface_area in SPECIFIC_SURFACE_AREA_LEVELS_M2_KG
    ]
    figure.legend(
        handles=legend_handles,
        title="Snow specific surface area",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=len(legend_handles),
        frameon=False,
    )

    for top_vnir, top_swir, terrain_case in top_axis_pairs:
        left = top_vnir.get_position().x0
        right = top_swir.get_position().x1
        top = max(
            top_vnir.get_position().y1,
            top_swir.get_position().y1,
        )
        figure.text(
            (left + right) / 2.0,
            top + 0.012,
            terrain_header(terrain_case),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            linespacing=1.35,
        )

    figure.text(
        0.055,
        0.025,
        figure_caption(),
        ha="left",
        va="bottom",
        fontsize=8.5,
        linespacing=1.35,
    )
    return figure


def main():
    """Run all terrain and SSA cases and save the comparison figure."""

    simulations = simulate_all_terrain_cases()
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
