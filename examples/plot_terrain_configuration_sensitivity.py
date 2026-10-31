#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot REDRESS sensitivity to the terrain configuration convention."""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from plot_idealized_radiance_decomposition import (
    BANDS,
    INPUT_CSV,
    OPPOSITE_WALL_SNOW_FRACTION,
    SHADED_BACKGROUND,
    SUNLIT_BACKGROUND,
    case_label,
    illumination_state,
    ordered_cases,
    read_records,
)


METHODS = ("hemispheric_closure", "dozier_frew")
METHOD_LABELS = {
    "hemispheric_closure": "Hemispheric closure: Ct = 1 − Vd",
    "dozier_frew": "Dozier–Frew: Ct = (1 + cos S)/2 − Vd",
}
METHOD_COLORS = {
    "hemispheric_closure": "#0072B2",
    "dozier_frew": "#D55E00",
}

OUTPUT_PNG = Path(
    "outputs/idealized_terrain_experiment/"
    "terrain_configuration_sensitivity.png"
)
OUTPUT_PDF = Path(
    "outputs/idealized_terrain_experiment/"
    "terrain_configuration_sensitivity.pdf"
)


def select_sensitivity_records(records):
    """Select all-snow experiments for both Ct conventions."""

    selected = [
        record
        for record in records
        if record["band"] in BANDS
        and np.isclose(
            record["opposite_wall_snow_fraction"],
            OPPOSITE_WALL_SNOW_FRACTION,
        )
        and record["terrain_configuration_method"] in METHODS
    ]
    if not selected:
        raise ValueError("No terrain-configuration sensitivity records found")
    return selected


def sensitivity_lookup(records):
    """Index observations by terrain, sample, band and Ct convention."""

    lookup = {}
    for record in records:
        key = (
            record["terrain"],
            record["sample"],
            record["band"],
            record["terrain_configuration_method"],
        )
        if key in lookup:
            raise ValueError(f"Duplicate observation: {key}")
        lookup[key] = record
    return lookup


def add_illumination_background(axis, x_position, states):
    """Shade each sample according to its computed beam visibility."""

    for position, state in zip(x_position, states):
        axis.axvspan(
            position - 0.48,
            position + 0.48,
            color=(
                SUNLIT_BACKGROUND
                if state == "Sunlit"
                else SHADED_BACKGROUND
            ),
            alpha=0.45,
            linewidth=0.0,
            zorder=0,
        )


def figure_caption():
    """Describe the two conventions and independent V-valley check."""

    caption = (
        "All-snow idealized-terrain experiments with full REDRESS case 5. "
        "Atmospheric quantities are interpolated from Py6S elevation nodes "
        "with a maximum spacing of 250 m. "
        "The primary hemispheric convention treats sky and terrain as the "
        "complete outward hemisphere, Ct=1−Vd. The Dozier–Frew sensitivity "
        "omits the downslope fraction (1−cos S)/2. Black crosses in panel "
        "(a) compare the exact area-mean wall-to-wall view factor of an "
        "infinite symmetric two-dimensional V-valley, F12=1−cos β (cross), "
        "with the raster wall mean (open circle). Lower panels report "
        "100(ρ_closure/ρ_DF−1); positive values "
        "therefore quantify the TOA-reflectance increase obtained when the "
        "complete terrain hemisphere is included."
    )
    return textwrap.fill(caption, width=205)


def plot_sensitivity(records):
    """Plot Ct itself and its effect on three Sentinel-2 bands."""

    closure_records = [
        record
        for record in records
        if record["terrain_configuration_method"] == METHODS[0]
    ]
    cases = ordered_cases(closure_records)
    lookup = sensitivity_lookup(records)
    x_position = np.arange(len(cases), dtype=np.float64)
    reference_records = [
        lookup[(terrain, sample, BANDS[0], METHODS[0])]
        for terrain, sample in cases
    ]
    states = [illumination_state(record) for record in reference_records]
    labels = [
        case_label(case, state)
        for case, state in zip(cases, states)
    ]

    figure, axes = plt.subplots(
        1 + len(BANDS),
        1,
        figsize=(17.5, 14.0),
        sharex=True,
    )
    for axis in axes:
        add_illumination_background(axis, x_position, states)
        axis.grid(axis="y", alpha=0.20, zorder=1)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=0.01)

    bar_width = 0.34
    for method_index, method in enumerate(METHODS):
        values = np.asarray([
            lookup[(terrain, sample, BANDS[0], method)]["ct"]
            for terrain, sample in cases
        ])
        offset = (method_index - 0.5) * bar_width
        axes[0].bar(
            x_position + offset,
            values,
            width=bar_width,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )

    analytical_x = []
    analytical_y = []
    raster_mean_x = []
    raster_mean_y = []
    for position, ((terrain, sample), record) in enumerate(zip(
        cases, reference_records
    )):
        exact = record.get("analytical_wall_view_factor", np.nan)
        if terrain.startswith("v_valley_") and "wall" in sample:
            if np.isfinite(exact):
                closure_position = position - 0.5 * bar_width
                analytical_x.append(closure_position + 0.045)
                analytical_y.append(exact)
                raster_mean = record.get("raster_wall_mean_ct", np.nan)
                if np.isfinite(raster_mean):
                    raster_mean_x.append(closure_position - 0.045)
                    raster_mean_y.append(raster_mean)
    axes[0].plot(
        raster_mean_x,
        raster_mean_y,
        linestyle="none",
        marker="o",
        markersize=6,
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=1.2,
        zorder=4,
    )
    axes[0].plot(
        analytical_x,
        analytical_y,
        linestyle="none",
        marker="x",
        markersize=7,
        markeredgewidth=1.4,
        color="black",
        zorder=4,
    )
    axes[0].set_ylabel("Terrain configuration\nfactor Ct")
    axes[0].set_title("(a) Configuration factors", loc="left", pad=7)

    for panel_index, (axis, band) in enumerate(zip(axes[1:], BANDS), start=1):
        closure = np.asarray([
            lookup[(terrain, sample, band, METHODS[0])]["toa_reflectance"]
            for terrain, sample in cases
        ])
        dozier_frew = np.asarray([
            lookup[(terrain, sample, band, METHODS[1])]["toa_reflectance"]
            for terrain, sample in cases
        ])
        relative_difference = np.divide(
            closure,
            dozier_frew,
            out=np.full_like(closure, np.nan),
            where=dozier_frew != 0.0,
        )
        relative_difference = 100.0 * (relative_difference - 1.0)
        axis.axhline(0.0, color="0.35", linewidth=0.8, zorder=1)
        axis.bar(
            x_position,
            relative_difference,
            width=0.68,
            color="#6A51A3",
            edgecolor="white",
            linewidth=0.4,
            zorder=2,
        )
        letter = chr(ord("a") + panel_index)
        axis.set_title(
            f"({letter}) {band}",
            loc="left",
            pad=7,
        )
        axis.set_ylabel("Closure − Dozier–Frew\nTOA difference (%)")

    axes[-1].set_xticks(x_position)
    axes[-1].set_xticklabels(labels, rotation=48, ha="right", fontsize=8)

    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    legend_handles.extend((
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            color="black",
            linestyle="none",
            label="Raster V-wall mean",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            linestyle="none",
            label="Exact V-valley F12",
        ),
        Patch(
            facecolor=SUNLIT_BACKGROUND,
            alpha=0.45,
            label="Sunlit sample",
        ),
        Patch(
            facecolor=SHADED_BACKGROUND,
            alpha=0.45,
            label="Shaded sample",
        ),
    ))
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        ncol=5,
        frameon=False,
    )
    figure.suptitle(
        "Sensitivity to the terrain configuration-factor convention",
        y=0.988,
        fontsize=16,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.89,
        bottom=0.225,
        hspace=0.25,
    )
    figure.text(
        0.075,
        0.025,
        figure_caption(),
        ha="left",
        va="bottom",
        fontsize=8.5,
        linespacing=1.35,
    )
    return figure


def main():
    """Read both Ct experiments and save the sensitivity figure."""

    records = select_sensitivity_records(read_records(INPUT_CSV))
    figure = plot_sensitivity(records)

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(figure)

    print(f"PNG figure: {OUTPUT_PNG}")
    print(f"PDF figure: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
