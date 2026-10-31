#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the analytical two-wall check of REDRESS terrain irradiance."""

import csv
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np


INPUT_CSV = Path(
    "outputs/idealized_terrain_experiment/two_wall_benchmark.csv"
)
OUTPUT_PNG = Path(
    "outputs/idealized_terrain_experiment/two_wall_benchmark.png"
)
OUTPUT_PDF = Path(
    "outputs/idealized_terrain_experiment/two_wall_benchmark.pdf"
)

BANDS = ("B03", "B08", "B11")
OPPOSITE_WALL_SNOW_FRACTION = 1.0
TERRAIN_CONFIGURATION_METHOD = "hemispheric_closure"

ILLUMINATION_FIELDS = (
    ("pre_exchange_irradiance_a", "Away-facing wall", "#CC79A7"),
    ("pre_exchange_irradiance_b", "Sun-facing wall", "#E69F00"),
    (
        "redress_flat_neighbour_irradiance",
        "REDRESS flat-neighbour assumption",
        "#999999",
    ),
)
TERRAIN_FIELDS = (
    ("redress_terrain_irradiance", "REDRESS", "#D55E00"),
    (
        "two_wall_raster_terrain_irradiance",
        "Two-wall, raster F",
        "#0072B2",
    ),
    (
        "two_wall_infinite_terrain_irradiance",
        "Two-wall, infinite-wall F=sin²β",
        "#009E73",
    ),
)


def read_records(input_file):
    """Read and type-convert the two-wall benchmark table."""

    records = []
    with input_file.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            converted = {}
            for name, value in row.items():
                if name in {
                    "terrain",
                    "band",
                    "terrain_configuration_method",
                }:
                    converted[name] = value
                else:
                    converted[name] = float(value)
            records.append(converted)
    return records


def select_records(records):
    """Select the all-snow primary configuration-factor experiment."""

    selected = [
        record
        for record in records
        if record["band"] in BANDS
        and np.isclose(
            record["opposite_wall_snow_fraction"],
            OPPOSITE_WALL_SNOW_FRACTION,
        )
        and record["terrain_configuration_method"]
        == TERRAIN_CONFIGURATION_METHOD
    ]
    expected = len(BANDS) * 3
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} selected records; found {len(selected)}"
        )
    return selected


def record_lookup(records):
    """Index records by Sentinel-2 band and valley-wall slope."""

    return {
        (record["band"], record["wall_slope_degree"]): record
        for record in records
    }


def grouped_bars(axis, x_position, records, fields):
    """Draw one grouped comparison and return its maximum value."""

    width = 0.24
    maximum = 0.0
    for index, (field, label, color) in enumerate(fields):
        values = np.asarray([record[field] for record in records])
        maximum = max(maximum, float(np.nanmax(values)))
        offset = (index - (len(fields) - 1) / 2.0) * width
        axis.bar(
            x_position + offset,
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
    return maximum


def figure_caption():
    """Return a self-contained description of the analytical benchmark."""

    caption = (
        "Full REDRESS case 5 over symmetric all-snow V-valleys with global "
        "SZA=45°. Wall A faces away from the Sun and wall B faces toward it. "
        "The upper row compares each wall’s own direct plus sky-diffuse "
        "irradiance before terrain exchange with the flat-neighbour "
        "irradiance used by the REDRESS approximation. The lower row compares "
        "REDRESS’s converged terrain-reflected irradiance on wall A with the "
        "2×2 Lambertian solution. ‘Raster F’ uses the local DEM-derived "
        "Ct=1−Vd and therefore isolates the neighbour-illumination assumption. "
        "The infinite-wall reference uses F=sin²β. Atmospheric quantities "
        "are interpolated from Py6S elevation nodes spaced by at most 250 m."
    )
    return textwrap.fill(caption, width=205)


def plot_benchmark(records):
    """Create the illumination and terrain-exchange comparison figure."""

    lookup = record_lookup(records)
    slopes = sorted({record["wall_slope_degree"] for record in records})
    x_position = np.arange(len(slopes), dtype=np.float64)
    figure, axes = plt.subplots(
        2,
        len(BANDS),
        figsize=(15.5, 8.8),
        sharex="col",
        constrained_layout=False,
    )

    for column, band in enumerate(BANDS):
        band_records = [lookup[(band, slope)] for slope in slopes]
        illumination_maximum = grouped_bars(
            axes[0, column],
            x_position,
            band_records,
            ILLUMINATION_FIELDS,
        )
        terrain_maximum = grouped_bars(
            axes[1, column],
            x_position,
            band_records,
            TERRAIN_FIELDS,
        )
        axes[0, column].set_ylim(0.0, 1.15 * illumination_maximum)
        axes[1, column].set_ylim(0.0, 1.15 * terrain_maximum)
        axes[0, column].set_title(band, fontweight="bold")
        axes[1, column].set_xticks(x_position)
        axes[1, column].set_xticklabels(
            [f"β={slope:g}°" for slope in slopes]
        )
        axes[1, column].set_xlabel("Symmetric valley-wall slope")
        for row in range(2):
            axes[row, column].grid(axis="y", alpha=0.22)
            axes[row, column].spines[["top", "right"]].set_visible(False)

    axes[0, 0].set_ylabel("Pre-exchange irradiance\n(Py6S band units)")
    axes[1, 0].set_ylabel("Terrain-reflected irradiance\n(Py6S band units)")
    axes[0, 0].legend(
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    axes[1, 0].legend(
        loc="upper left",
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        "Two-wall benchmark of the REDRESS neighbour-illumination assumption",
        y=0.975,
        fontsize=15,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.90,
        bottom=0.20,
        hspace=0.28,
        wspace=0.20,
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
    """Read the benchmark table and save publication-ready figures."""

    records = select_records(read_records(INPUT_CSV))
    figure = plot_benchmark(records)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(figure)
    print(f"PNG figure: {OUTPUT_PNG}")
    print(f"PDF figure: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
