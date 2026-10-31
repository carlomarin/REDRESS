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
FINAL_CONFIGURATION_METHOD = "hemispheric_closure"
FINAL_NEIGHBOUR_BEAM_WEIGHTING = 0.0

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
        "dozier_frew",
        1.0,
        "Dozier–Frew Ct; beam weighting",
        "#999999",
    ),
    (
        "hemispheric_closure",
        1.0,
        "Ct=1−Vd; beam weighting",
        "#D55E00",
    ),
    (
        "dozier_frew",
        0.0,
        "Dozier–Frew Ct; no beam weighting",
        "#56B4E9",
    ),
    (
        "hemispheric_closure",
        0.0,
        "Ct=1−Vd; no beam weighting",
        "#0072B2",
    ),
)
FINAL_FIELDS = (
    ("redress_terrain_irradiance", "REDRESS: both changes", "#0072B2"),
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
    ]
    expected = len(BANDS) * 3 * len(REDRESS_VARIANTS)
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} selected records; found {len(selected)}"
        )
    return selected


def record_lookup(records):
    """Index records by band, slope, Ct convention and beam weighting."""

    return {
        (
            record["band"],
            record["wall_slope_degree"],
            record["terrain_configuration_method"],
            record["neighbour_beam_weighting"],
        ): record
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


def redress_variant_bars(axis, x_position, lookup, band, slopes):
    """Plot the full two-by-two REDRESS attribution experiment."""

    width = 0.18
    maximum = 0.0
    for index, (method, weighting, label, color) in enumerate(
        REDRESS_VARIANTS
    ):
        values = np.asarray([
            lookup[(band, slope, method, weighting)][
                "redress_terrain_irradiance"
            ]
            for slope in slopes
        ])
        maximum = max(maximum, float(np.nanmax(values)))
        offset = (index - (len(REDRESS_VARIANTS) - 1) / 2.0) * width
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
        "irradiance used by the REDRESS approximation. The middle row crosses "
        "the two Ct conventions with the neighbour beam-visibility weighting "
        "enabled and disabled. The lower row compares the final REDRESS "
        "variant (Ct=1−Vd without beam weighting) with the 2×2 Lambertian "
        "solution. The raster-F reference uses the same local Ct, so its "
        "remaining difference measures the flat-neighbour irradiance and "
        "neighbourhood-aggregation approximations together. The infinite-wall "
        "reference uses F=sin²β. Atmospheric quantities are interpolated from "
        "Py6S elevation nodes spaced by at most 250 m."
    )
    return textwrap.fill(caption, width=205)


def plot_benchmark(records):
    """Create the illumination and terrain-exchange comparison figure."""

    lookup = record_lookup(records)
    slopes = sorted({record["wall_slope_degree"] for record in records})
    x_position = np.arange(len(slopes), dtype=np.float64)
    figure, axes = plt.subplots(
        3,
        len(BANDS),
        figsize=(15.5, 11.6),
        sharex="col",
        constrained_layout=False,
    )

    for column, band in enumerate(BANDS):
        final_records = [
            lookup[(
                band,
                slope,
                FINAL_CONFIGURATION_METHOD,
                FINAL_NEIGHBOUR_BEAM_WEIGHTING,
            )]
            for slope in slopes
        ]
        illumination_maximum = grouped_bars(
            axes[0, column],
            x_position,
            final_records,
            ILLUMINATION_FIELDS,
        )
        variant_maximum = redress_variant_bars(
            axes[1, column],
            x_position,
            lookup,
            band,
            slopes,
        )
        final_maximum = grouped_bars(
            axes[2, column],
            x_position,
            final_records,
            FINAL_FIELDS,
        )
        axes[0, column].set_ylim(0.0, 1.15 * illumination_maximum)
        axes[1, column].set_ylim(0.0, 1.15 * variant_maximum)
        axes[2, column].set_ylim(0.0, 1.15 * final_maximum)
        axes[0, column].set_title(band, fontweight="bold")
        axes[2, column].set_xticks(x_position)
        axes[2, column].set_xticklabels(
            [f"β={slope:g}°" for slope in slopes]
        )
        axes[2, column].set_xlabel("Symmetric valley-wall slope")
        for row in range(3):
            axes[row, column].grid(axis="y", alpha=0.22)
            axes[row, column].spines[["top", "right"]].set_visible(False)

    axes[0, 0].set_ylabel("Pre-exchange irradiance\n(Py6S band units)")
    axes[1, 0].set_ylabel("REDRESS terrain irradiance\n(Py6S band units)")
    axes[2, 0].set_ylabel("Final comparison\n(Py6S band units)")
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
    axes[2, 0].legend(
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
