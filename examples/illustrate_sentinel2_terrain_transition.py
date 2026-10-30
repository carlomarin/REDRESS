#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Illustrate the transition from flat to shadowed Sentinel-2 snow.

The experiment follows the simplified TOA-radiance formulations in Sect. 2.2
of Lamare et al. (2020). Flat terrain is simulated with REDRESS case 1.
Illuminated and shadowed slopes use REDRESS case 2, which retains local
illumination and viewing geometry and the sky-view factor but excludes
terrain reillumination and atmospheric-terrain coupling.

Edit the constants below and run this script from the repository root:

``python examples/illustrate_sentinel2_terrain_transition.py``
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon
import numpy as np
import xarray as xr

from redress.lut import (
    Sentinel2EndmemberScenario,
    create_sentinel2_toa_lut,
)


# ---------------------------------------------------------------------------
# Scientific configuration
# ---------------------------------------------------------------------------

PLATFORM = "S2B"
BANDS = (
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
)

MONTH = 3
DAY = 28

SOLAR_ZENITH_DEG = 45.0
SLOPE_DEG = 25.0
SKY_VIEW_OPEN_SLOPE = float(np.cos(np.deg2rad(SLOPE_DEG)))
SKY_VIEW_SHADOWED_SLOPE = 0.60

SPECIFIC_SURFACE_AREA = 20.0
ALTITUDE_M = 2000.0
AOD550 = 0.10
WATER_VAPOUR_G_CM2 = 1.0
OZONE_CM_ATM = 0.30

AEROSOL_MODEL = "Continental"
ATMOSPHERIC_PROFILE = None
SCENE_SIZE = 9
PIXEL_RESOLUTION_M = 20.0
TERRAIN_WINDOW_WIDTH_M = 1500.0
ATMOSPHERE_WINDOW_WIDTH_M = 2100.0

OUTPUT_DIRECTORY = Path("outputs/sentinel2_terrain_transition")
OUTPUT_DATASET = OUTPUT_DIRECTORY / "sentinel2_terrain_transition.nc"
OUTPUT_PNG = OUTPUT_DIRECTORY / "sentinel2_terrain_transition.png"
OUTPUT_PDF = OUTPUT_DIRECTORY / "sentinel2_terrain_transition.pdf"


@dataclass(frozen=True)
class TerrainExperiment:
    """One physically interpretable terrain configuration."""

    key: str
    title: str
    short_label: str
    redress_case: int
    scenario: Sentinel2EndmemberScenario
    terrain_kind: str


def _scenario(
    name,
    local_incidence_angle,
    local_view_zenith_angle,
    relative_azimuth_angle,
    sky_view_factor,
    shadowed,
):
    """Build one snow scenario while holding non-terrain inputs constant."""

    return Sentinel2EndmemberScenario(
        name=name,
        category="Terrain transition",
        specific_surface_area=SPECIFIC_SURFACE_AREA,
        solar_zenith_angle=SOLAR_ZENITH_DEG,
        local_incidence_angle=local_incidence_angle,
        view_zenith_angle=0.0,
        local_view_zenith_angle=local_view_zenith_angle,
        relative_azimuth_angle=relative_azimuth_angle,
        altitude_m=ALTITUDE_M,
        aod550=AOD550,
        water_vapour_g_cm2=WATER_VAPOUR_G_CM2,
        ozone_cm_atm=OZONE_CM_ATM,
        sky_view_factor=sky_view_factor,
        terrain_configuration_factor=1.0 - sky_view_factor,
        shadowed=shadowed,
    )


EXPERIMENTS = (
    TerrainExperiment(
        key="flat",
        title="Flat snow",
        short_label="Flat",
        redress_case=1,
        scenario=_scenario(
            name="flat",
            local_incidence_angle=SOLAR_ZENITH_DEG,
            local_view_zenith_angle=0.0,
            relative_azimuth_angle=0.0,
            sky_view_factor=1.0,
            shadowed=False,
        ),
        terrain_kind="flat",
    ),
    TerrainExperiment(
        key="sun_facing",
        title="Sun-facing slope",
        short_label="Sun-facing",
        redress_case=2,
        scenario=_scenario(
            name="sun_facing",
            local_incidence_angle=SOLAR_ZENITH_DEG - SLOPE_DEG,
            local_view_zenith_angle=SLOPE_DEG,
            relative_azimuth_angle=180.0,
            sky_view_factor=SKY_VIEW_OPEN_SLOPE,
            shadowed=False,
        ),
        terrain_kind="sun_facing",
    ),
    TerrainExperiment(
        key="away_facing",
        title="Away-facing slope",
        short_label="Away-facing",
        redress_case=2,
        scenario=_scenario(
            name="away_facing",
            local_incidence_angle=SOLAR_ZENITH_DEG + SLOPE_DEG,
            local_view_zenith_angle=SLOPE_DEG,
            relative_azimuth_angle=0.0,
            sky_view_factor=SKY_VIEW_OPEN_SLOPE,
            shadowed=False,
        ),
        terrain_kind="away_facing",
    ),
    TerrainExperiment(
        key="cast_shadow",
        title="Cast-shadowed slope",
        short_label="Cast shadow",
        redress_case=2,
        scenario=_scenario(
            name="cast_shadow",
            local_incidence_angle=SOLAR_ZENITH_DEG + SLOPE_DEG,
            local_view_zenith_angle=SLOPE_DEG,
            relative_azimuth_angle=0.0,
            sky_view_factor=SKY_VIEW_SHADOWED_SLOPE,
            shadowed=True,
        ),
        terrain_kind="shadow",
    ),
)

CASE_COLORS = {
    "flat": "#30343b",
    "sun_facing": "#d95f02",
    "away_facing": "#1b9e77",
    "cast_shadow": "#5e4fa2",
}


def simulate_experiments():
    """Run each terrain configuration with its corresponding REDRESS case."""

    simulations = []
    for experiment in EXPERIMENTS:
        print(f"Simulating {experiment.title}")
        result = create_sentinel2_toa_lut(
            platform=PLATFORM,
            band_names=BANDS,
            scenarios=(experiment.scenario,),
            month=MONTH,
            day=DAY,
            aerosol_model=AEROSOL_MODEL,
            atmospheric_profile=ATMOSPHERIC_PROFILE,
            case_number=experiment.redress_case,
            scene_size=SCENE_SIZE,
            pixel_resolution_m=PIXEL_RESOLUTION_M,
            terrain_window_width_m=TERRAIN_WINDOW_WIDTH_M,
            atmosphere_window_width_m=ATMOSPHERE_WINDOW_WIDTH_M,
        )
        result["redress_case"] = xr.DataArray(
            [experiment.redress_case], dims=("scenario",)
        )
        simulations.append(result)

    return xr.concat(
        simulations,
        dim="scenario",
        data_vars="all",
        coords="minimal",
        compat="override",
    )


def _draw_arrow(axis, start, end, color, linestyle="-", linewidth=2.0):
    """Draw an arrow in normalized panel coordinates."""

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.add_patch(arrow)


def _draw_terrain_panel(axis, experiment):
    """Draw the illumination geometry associated with one spectrum."""

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    if experiment.terrain_kind == "flat":
        surface = np.asarray([[0.04, 0.25], [0.96, 0.25]])
    elif experiment.terrain_kind == "sun_facing":
        surface = np.asarray([[0.04, 0.16], [0.96, 0.48]])
    else:
        surface = np.asarray([[0.04, 0.48], [0.96, 0.16]])

    polygon = np.vstack((surface, [[0.96, 0.02], [0.04, 0.02]]))
    axis.add_patch(Polygon(
        polygon,
        closed=True,
        facecolor="#d9dde3",
        edgecolor="#454b54",
        linewidth=1.5,
        transform=axis.transAxes,
    ))

    target = tuple(np.mean(surface, axis=0))
    if experiment.terrain_kind == "shadow":
        ridge = Polygon(
            [[0.00, 0.28], [0.22, 0.76], [0.38, 0.36]],
            closed=True,
            facecolor="#8b919a",
            edgecolor="#454b54",
            linewidth=1.3,
            transform=axis.transAxes,
        )
        axis.add_patch(ridge)
        _draw_arrow(
            axis,
            (0.02, 0.92),
            (0.22, 0.74),
            "#e6a700",
            linestyle="--",
        )
        axis.text(
            0.05,
            0.84,
            "direct beam blocked",
            color="#8a6500",
            fontsize=8,
            transform=axis.transAxes,
        )
    else:
        _draw_arrow(axis, (0.03, 0.92), target, "#e6a700")

    _draw_arrow(axis, target, (0.72, 0.94), "#4c78a8", linewidth=1.8)
    _draw_arrow(axis, (0.88, 0.84), target, "#72b7d2", linestyle=":")

    scenario = experiment.scenario
    axis.set_title(experiment.title, fontsize=11, fontweight="bold", pad=3)
    axis.text(
        0.5,
        -0.01,
        (
            f"local incidence = {scenario.local_incidence_angle:g}°\n"
            f"$V_d$ = {scenario.sky_view_factor:.2f}, "
            f"$b$ = {0 if scenario.shadowed else 1}"
        ),
        ha="center",
        va="top",
        fontsize=8.5,
        transform=axis.transAxes,
    )


def _component_reflectance(lut, component_name):
    """Express a radiance component using the total TOA-reflectance scale."""

    total_radiance = lut["toa_radiance"]
    component_radiance = lut[f"toa_{component_name}_radiance"]
    return xr.where(
        total_radiance != 0,
        lut["toa_reflectance"] * component_radiance / total_radiance,
        0.0,
    )


def plot_terrain_transition(lut):
    """Create the terrain schematic and quantitative Sentinel-2 spectra."""

    figure = plt.figure(figsize=(16, 9.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, height_ratios=(0.75, 1.25))

    for column, experiment in enumerate(EXPERIMENTS):
        _draw_terrain_panel(figure.add_subplot(grid[0, column]), experiment)

    spectra_axis = figure.add_subplot(grid[1, :2])
    difference_axis = figure.add_subplot(grid[1, 2])
    components_axis = figure.add_subplot(grid[1, 3])

    wavelength = lut["wavelength_nm"].values
    flat_reflectance = lut["toa_reflectance"].sel(scenario="flat")

    for experiment in EXPERIMENTS:
        reflectance = lut["toa_reflectance"].sel(
            scenario=experiment.key
        )
        color = CASE_COLORS[experiment.key]
        spectra_axis.plot(
            wavelength,
            reflectance,
            marker="o",
            markersize=4,
            linewidth=2.0,
            color=color,
            label=experiment.short_label,
        )
        difference_axis.plot(
            wavelength,
            reflectance - flat_reflectance,
            marker="o",
            markersize=3.5,
            linewidth=1.7,
            color=color,
            label=experiment.short_label,
        )

    spectra_axis.set_title("Sentinel-2 TOA reflectance")
    spectra_axis.set_xlabel("Wavelength (nm)")
    spectra_axis.set_ylabel("TOA reflectance factor")
    spectra_axis.set_ylim(0.0, 1.30)
    spectra_axis.legend(frameon=False, ncols=2)

    difference_axis.axhline(0.0, color="0.45", linewidth=0.8)
    difference_axis.set_title("Difference from flat snow")
    difference_axis.set_xlabel("Wavelength (nm)")
    difference_axis.set_ylabel("Δ TOA reflectance")

    component_names = (
        "direct_surface",
        "diffuse_surface",
        "atmospheric_path",
        "neighbour_atmosphere",
    )
    component_labels = (
        "Direct surface",
        "Diffuse surface",
        "Atmospheric path",
        "Neighbour-atmosphere",
    )
    component_colors = ("#e6a700", "#72b7d2", "#9b9da1", "#b279a2")

    selected_bands = ("B03", "B11")
    bar_width = 0.34
    x_position = np.arange(len(EXPERIMENTS), dtype=np.float64)
    for band_index, band in enumerate(selected_bands):
        offset = (band_index - 0.5) * bar_width
        bottom = np.zeros(len(EXPERIMENTS), dtype=np.float64)
        for component_index, (name, color) in enumerate(zip(
            component_names, component_colors
        )):
            values = _component_reflectance(lut, name).sel(
                scenario=[item.key for item in EXPERIMENTS],
                band=band,
            ).values
            components_axis.bar(
                x_position + offset,
                values,
                width=bar_width,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                label=(
                    component_labels[component_index]
                    if band_index == 0 else None
                ),
            )
            bottom += values

        for position, height in zip(x_position + offset, bottom):
            components_axis.text(
                position,
                height + 0.015,
                band,
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=7,
            )

    components_axis.set_title("TOA contributions at 560 and 1610 nm")
    components_axis.set_ylabel("Reflectance-equivalent contribution")
    components_axis.set_xticks(x_position)
    components_axis.set_xticklabels(
        [item.short_label for item in EXPERIMENTS],
        rotation=25,
        ha="right",
    )
    components_axis.legend(frameon=False, fontsize=7, loc="upper right")

    for axis in (spectra_axis, difference_axis, components_axis):
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "Evolution of Sentinel-2 snow reflectance from flat terrain to shadow",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.01,
        (
            "Flat: Eqs. (23-25). Slopes: local geometry in Eqs. (3, 5) "
            "and diffuse irradiance scaled by $V_d$ in Eq. (26). "
            "In cast shadow, $b=0$ removes the direct surface term."
        ),
        ha="center",
        fontsize=9,
    )
    return figure


def main():
    """Run REDRESS and save the reproducible dataset and illustration."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    lut = simulate_experiments()
    lut.attrs["figure_purpose"] = (
        "Terrain transition following Lamare et al. (2020), Sect. 2.2"
    )
    lut.to_netcdf(OUTPUT_DATASET)

    figure = plot_terrain_transition(lut)
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(figure)

    print(f"Dataset: {OUTPUT_DATASET}")
    print(f"PNG figure: {OUTPUT_PNG}")
    print(f"PDF figure: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
