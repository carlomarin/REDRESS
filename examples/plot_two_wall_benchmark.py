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
CONFIGURATION_METHODS = ("hemispheric_closure", "dozier_frew")
FINAL_CONFIGURATION_METHOD = "hemispheric_closure"
WALLS = (
    ("A", "Wall A (away from the Sun)"),
    ("B", "Wall B (toward the Sun)"),
)

ILLUMINATION_FIELDS = (
    ("pre_exchange_irradiance_a", "Away-facing wall", "#CC79A7"),
    ("pre_exchange_irradiance_b", "Sun-facing wall", "#E69F00"),
    (
        "redress_flat_neighbour_irradiance",
        "REDRESS flat-neighbour assumption",
        "#999999",
    ),
)
REDRESS_VARIANTS = (
    (
        "flat_irradiance",
        1.0,
        "REDRESS: flat irradiance, beam weighting",
        "#D55E00",
    ),
    (
        "flat_irradiance",
        0.0,
        "REDRESS: flat irradiance, no beam weighting",
        "#56B4E9",
    ),
    (
        "neighbour_exitance",
        0.0,
        "REDRESS: window-mean exitance",
        "#0072B2",
    ),
    (
        "visible_exitance",
        0.0,
        "REDRESS: visible view-factor exitance",
        "#000000",
    ),
)
REFERENCE_FIELDS = (
    (
        "two_wall_raster_terrain_irradiance",
        "Two-wall: same raster F",
        "#009E73",
    ),
    (
        "two_wall_infinite_terrain_irradiance",
        "Two-wall: infinite-wall F=sin²β",
        "#CC79A7",
    ),
)
BAR_WIDTH = 0.13


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
                    "wall",
                    "terrain_configuration_method",
                    "terrain_reflection_method",
                }:
                    converted[name] = value
                else:
                    converted[name] = float(value)
            records.append(converted)
    return records


def select_records(records):
    """Select the all-snow experiments for both walls."""

    selected = [
        record
        for record in records
        if record["band"] in BANDS
        and np.isclose(
            record["opposite_wall_snow_fraction"],
            OPPOSITE_WALL_SNOW_FRACTION,
        )
    ]
    expected = (
        len(BANDS)
        * 3
        * len(WALLS)
        * len(CONFIGURATION_METHODS)
        * len(REDRESS_VARIANTS)
    )
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} selected records; found {len(selected)}"
        )
    return selected


def record_lookup(records):
    """Index records by band, slope, wall, Ct and reflection variant."""

    return {
        (
            record["band"],
            record["wall_slope_degree"],
            record["wall"],
            record["terrain_configuration_method"],
            record["terrain_reflection_method"],
            record["neighbour_beam_weighting"],
        ): record
        for record in records
    }


def format_percent(value):
    """Format a relative difference compactly for a bar label."""

    if abs(value) >= 100.0:
        return f"{value:+.0f}%"
    return f"{value:+.1f}%"


def draw_bars(axis, x_position, series):
    """Draw grouped bars and return the maximum bar height.

    Each series is (values, label, color, percents); percents is None or
    the per-bar difference from the reference, printed above each bar.
    """

    maximum = 0.0
    for index, (values, label, color, percents) in enumerate(series):
        values = np.asarray(values, dtype=np.float64)
        maximum = max(maximum, float(np.nanmax(values)))
        offset = (index - (len(series) - 1) / 2.0) * BAR_WIDTH
        bars = axis.bar(
            x_position + offset,
            values,
            width=BAR_WIDTH,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )
        if percents is not None:
            axis.bar_label(
                bars,
                labels=[format_percent(value) for value in percents],
                padding=2,
                rotation=90,
                fontsize=6.5,
                color=color,
            )
    return maximum


def wall_series(lookup, band, slopes, wall):
    """Return the REDRESS variants followed by the two-wall references."""

    variant_records = {
        (reflection, weighting): [
            lookup[(
                band,
                slope,
                wall,
                FINAL_CONFIGURATION_METHOD,
                reflection,
                weighting,
            )]
            for slope in slopes
        ]
        for reflection, weighting, _, _ in REDRESS_VARIANTS
    }
    series = [
        (
            [
                record["redress_terrain_irradiance"]
                for record in variant_records[(reflection, weighting)]
            ],
            label,
            color,
            [
                record["redress_vs_raster_two_wall_percent"]
                for record in variant_records[(reflection, weighting)]
            ],
        )
        for reflection, weighting, label, color in REDRESS_VARIANTS
    ]
    # The references depend only on the wall's own Ct and illumination,
    # which are identical across reflection variants.
    reference_records = next(iter(variant_records.values()))
    series.extend(
        (
            [record[field] for record in reference_records],
            label,
            color,
            None,
        )
        for field, label, color in REFERENCE_FIELDS
    )
    return series


def print_summary(lookup, slopes):
    """Print the REDRESS error against the same-F two-wall solution."""

    print(
        "REDRESS vs two-wall (same raster F), "
        f"Ct={FINAL_CONFIGURATION_METHOD}, percent"
    )
    for wall, _ in WALLS:
        for reflection, weighting, label, _ in REDRESS_VARIANTS:
            values = [
                lookup[(
                    band,
                    slope,
                    wall,
                    FINAL_CONFIGURATION_METHOD,
                    reflection,
                    weighting,
                )]["redress_vs_raster_two_wall_percent"]
                for band in BANDS
                for slope in slopes
            ]
            formatted = " ".join(f"{value:+7.1f}" for value in values)
            print(f"  wall {wall} | {label:<45} | {formatted}")
    print(
        "  columns: "
        + ", ".join(
            f"{band} β={slope:g}°" for band in BANDS for slope in slopes
        )
    )


def figure_caption():
    """Return a self-contained description of the analytical benchmark."""

    caption = (
        "Full REDRESS case 5 over symmetric all-snow V-valleys with global "
        "SZA=45° and Ct=1−Vd. Wall A faces away from the Sun and wall B "
        "faces toward it. The upper row compares each wall’s own direct plus "
        "sky-diffuse irradiance before terrain exchange with the "
        "flat-neighbour irradiance of the published REDRESS approximation. "
        "The lower rows compare four REDRESS terrain-reflection variants at "
        "each wall with the 2×2 Lambertian solution: the published "
        "flat-irradiance term with and without the neighbour beam-visibility "
        "weighting, the window-mean exitance term, which averages each "
        "neighbour’s reflectance times its own pre-exchange irradiance over "
        "the uniform terrain window, and the visible exitance term, which "
        "weights neighbours by their line-of-sight view-factor element and "
        "iterates on their complete irradiance, with rays reaching 3 km. The "
        "raster-F reference uses the same local Ct; the percentage above "
        "each REDRESS bar is its difference from this reference, and the "
        "remaining difference of the visible variant measures the non-uniform "
        "irradiance along each wall. The infinite-wall reference uses "
        "F=sin²β. Atmospheric quantities are interpolated from Py6S "
        "elevation nodes spaced by at most 250 m."
    )
    return textwrap.fill(caption, width=205)


def plot_benchmark(records):
    """Create the illumination and terrain-exchange comparison figure."""

    lookup = record_lookup(records)
    slopes = sorted({record["wall_slope_degree"] for record in records})
    print_summary(lookup, slopes)
    x_position = np.arange(len(slopes), dtype=np.float64)
    figure, axes = plt.subplots(
        1 + len(WALLS),
        len(BANDS),
        figsize=(15.5, 11.6),
        sharex="col",
        constrained_layout=False,
    )

    for column, band in enumerate(BANDS):
        illumination_records = [
            lookup[(
                band,
                slope,
                "A",
                FINAL_CONFIGURATION_METHOD,
                "flat_irradiance",
                0.0,
            )]
            for slope in slopes
        ]
        illumination_maximum = draw_bars(
            axes[0, column],
            x_position,
            [
                (
                    [record[field] for record in illumination_records],
                    label,
                    color,
                    None,
                )
                for field, label, color in ILLUMINATION_FIELDS
            ],
        )
        axes[0, column].set_ylim(0.0, 1.15 * illumination_maximum)
        for row, (wall, _) in enumerate(WALLS, start=1):
            wall_maximum = draw_bars(
                axes[row, column],
                x_position,
                wall_series(lookup, band, slopes, wall),
            )
            # Headroom for the rotated percent labels above the bars.
            axes[row, column].set_ylim(0.0, 1.32 * wall_maximum)
        axes[0, column].set_title(band, fontweight="bold")
        axes[-1, column].set_xticks(x_position)
        axes[-1, column].set_xticklabels(
            [f"β={slope:g}°" for slope in slopes]
        )
        axes[-1, column].set_xlabel("Symmetric valley-wall slope")
        for row in range(axes.shape[0]):
            axes[row, column].grid(axis="y", alpha=0.22)
            axes[row, column].spines[["top", "right"]].set_visible(False)

    axes[0, 0].set_ylabel("Pre-exchange irradiance\n(Py6S band units)")
    for row, (_, label) in enumerate(WALLS, start=1):
        axes[row, 0].set_ylabel(
            f"{label}\nterrain irradiance (Py6S band units)"
        )
    for row in range(axes.shape[0]):
        axes[row, 0].legend(
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
        bottom=0.17,
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
