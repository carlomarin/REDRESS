#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the REDRESS TOA decomposition for idealized terrain samples.

The input is produced by ``run_idealized_terrain_experiment.py``. Radiance
components are converted to reflectance-equivalent contributions so that each
stack sums to the reported Sentinel-2 TOA reflectance.

Run from the repository root:

``python examples/plot_idealized_radiance_decomposition.py``
"""

import csv
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


INPUT_CSV = Path(
    "outputs/idealized_terrain_experiment/idealized_sample_results.csv"
)
OUTPUT_PNG = Path(
    "outputs/idealized_terrain_experiment/"
    "idealized_component_decomposition.png"
)
OUTPUT_PDF = Path(
    "outputs/idealized_terrain_experiment/"
    "idealized_component_decomposition.pdf"
)

BANDS = ("B03", "B08", "B11")
OPPOSITE_WALL_SNOW_FRACTION = 1.0
TERRAIN_CONFIGURATION_METHOD = "hemispheric_closure"
NEIGHBOUR_BEAM_WEIGHTING = 1.0
SUNLIT_VISIBILITY_THRESHOLD = 0.5

TERRAIN_ORDER = (
    "flat_plateau",
    "slope_sun_25",
    "slope_away_25",
    "v_valley_25",
    "v_valley_45",
    "v_valley_55",
    "circular_bowl_40",
    "narrow_couloir_sun_aligned_65",
    "narrow_couloir_cross_sun_65",
)

TERRAIN_SHORT_LABELS = {
    "flat_plateau": "Flat plateau",
    "slope_sun_25": "25° slope\nfacing Sun",
    "slope_away_25": "25° slope\nfacing away",
    "v_valley_25": "V-valley\nβ=25°",
    "v_valley_45": "V-valley\nβ=45°",
    "v_valley_55": "V-valley\nβ=55°",
    "circular_bowl_40": "Circular bowl\nβ=40°",
    "narrow_couloir_sun_aligned_65": (
        "Couloir β=65°\naxis along Sun"
    ),
    "narrow_couloir_cross_sun_65": (
        "Couloir β=65°\naxis across Sun"
    ),
}

SAMPLE_SHORT_LABELS = {
    "centre": "centre",
    "mid_slope": "mid-slope",
    "away_wall_midpoint": "away wall",
    "valley_floor": "floor",
    "bowl_floor": "floor",
    "cross_wall_midpoint": "mid-wall",
    "couloir_floor": "floor",
}

COMPONENTS = (
    (
        "toa_direct_surface_radiance",
        "Direct surface",
        "#E69F00",
    ),
    (
        "toa_diffuse_surface_radiance",
        "Diffuse surface (including terrain)",
        "#56B4E9",
    ),
    (
        "toa_neighbour_atmosphere_radiance",
        "Neighbour-to-sensor",
        "#009E73",
    ),
    (
        "toa_atmospheric_path_radiance",
        "Atmospheric path",
        "#7A7A7A",
    ),
)

SUNLIT_BACKGROUND = "#FFF3BF"
SHADED_BACKGROUND = "#D9D2E9"


def read_records(input_file):
    """Read numeric CSV fields while preserving categorical identifiers."""

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input not found: {input_file}. Run the idealized-terrain "
            "experiment first."
        )

    categorical_fields = {
        "terrain",
        "terrain_label",
        "sample",
        "band",
        "terrain_configuration_method",
    }
    records = []
    with input_file.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            records.append({
                key: value if key in categorical_fields else float(value)
                for key, value in row.items()
            })
    if not records:
        raise ValueError(f"No observations found in {input_file}")
    return records


def select_all_snow_records(records):
    """Select the common all-snow reference for the central decomposition."""

    selected = [
        record
        for record in records
        if record["band"] in BANDS
        and np.isclose(
            record["opposite_wall_snow_fraction"],
            OPPOSITE_WALL_SNOW_FRACTION,
        )
        and record.get(
            "terrain_configuration_method",
            TERRAIN_CONFIGURATION_METHOD,
        ) == TERRAIN_CONFIGURATION_METHOD
        and record.get(
            "neighbour_beam_weighting",
            NEIGHBOUR_BEAM_WEIGHTING,
        ) == NEIGHBOUR_BEAM_WEIGHTING
    ]
    if not selected:
        raise ValueError(
            "No records match the requested bands and opposite-wall cover"
        )
    return selected


def ordered_cases(records):
    """Return unique terrain/sample combinations in scientific order."""

    available = {
        (record["terrain"], record["sample"])
        for record in records
    }
    sample_order = {
        "centre": 0,
        "mid_slope": 0,
        "away_wall_midpoint": 0,
        "cross_wall_midpoint": 0,
        "valley_floor": 1,
        "bowl_floor": 1,
        "couloir_floor": 1,
    }
    terrain_rank = {
        terrain: rank for rank, terrain in enumerate(TERRAIN_ORDER)
    }
    unexpected = {
        terrain for terrain, _ in available if terrain not in terrain_rank
    }
    if unexpected:
        raise ValueError(
            "TERRAIN_ORDER is missing: " + ", ".join(sorted(unexpected))
        )
    return sorted(
        available,
        key=lambda case: (
            terrain_rank[case[0]],
            sample_order.get(case[1], 99),
            case[1],
        ),
    )


def record_lookup(records):
    """Index records by terrain, sample and Sentinel-2 band."""

    lookup = {}
    for record in records:
        key = (record["terrain"], record["sample"], record["band"])
        if key in lookup:
            raise ValueError(f"Duplicate observation: {key}")
        lookup[key] = record
    return lookup


def reflectance_components(record):
    """Convert the four radiance terms to TOA-reflectance contributions."""

    component_radiance = np.asarray([
        record[field] for field, _, _ in COMPONENTS
    ], dtype=np.float64)
    total_radiance = np.sum(component_radiance)
    if not np.isfinite(total_radiance) or total_radiance <= 0.0:
        raise ValueError(
            "Component radiances do not form a positive finite total for "
            f"{record['terrain']}/{record['sample']}/{record['band']}"
        )
    contributions = (
        record["toa_reflectance"]
        * component_radiance
        / total_radiance
    )
    if not np.all(np.isfinite(contributions)):
        raise ValueError("Non-finite reflectance contribution encountered")
    return contributions


def illumination_state(record):
    """Classify the sample using REDRESS beam visibility."""

    visibility = record["beam_visibility"]
    if not np.isfinite(visibility):
        raise ValueError(
            f"Invalid beam visibility for {record['terrain']}/"
            f"{record['sample']}"
        )
    return (
        "Sunlit"
        if visibility >= SUNLIT_VISIBILITY_THRESHOLD
        else "Shaded"
    )


def case_label(case, state):
    """Create a compact three-line label for one sampled location."""

    terrain, sample = case
    terrain_label = TERRAIN_SHORT_LABELS.get(terrain, terrain)
    sample_label = SAMPLE_SHORT_LABELS.get(sample, sample)
    return f"{terrain_label}\n{sample_label} · {state.lower()}"


def stacked_bars(axis, x_position, values, colors):
    """Draw positive and negative contributions with separate baselines."""

    positive_bottom = np.zeros(values.shape[1], dtype=np.float64)
    negative_bottom = np.zeros(values.shape[1], dtype=np.float64)
    for component_index, color in enumerate(colors):
        component = values[component_index]
        bottom = np.where(
            component >= 0.0,
            positive_bottom,
            negative_bottom,
        )
        axis.bar(
            x_position,
            component,
            width=0.72,
            bottom=bottom,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=2,
        )
        positive_bottom += np.where(component >= 0.0, component, 0.0)
        negative_bottom += np.where(component < 0.0, component, 0.0)


def figure_caption():
    """Return a self-contained explanation of the plotted decomposition."""

    caption = (
        "Full REDRESS case 5 over analytical DEMs; snow SSA=20 m² kg⁻¹, "
        "global SZA=45°, and the designated opposite wall is fully snow "
        "covered. Terrain configuration uses hemispheric closure, Ct=1−Vd. "
        "The published REDRESS neighbour beam-visibility weighting is active. "
        "Atmospheric quantities are linearly interpolated from Py6S runs "
        "with a maximum elevation spacing of 250 m. "
        "Each radiance component is expressed on the TOA-reflectance "
        "scale as ρ_TOA L_component / ΣL_component, so each stack sums to the "
        "modelled TOA reflectance (open circle). ‘Diffuse surface’ includes "
        "sky-diffuse, terrain-reflected, and atmosphere–terrain-coupled "
        "downwelling irradiance reflected by the target. ‘Neighbour-to-sensor’ "
        "is the separate LtNA adjacency term. Yellow and violet backgrounds "
        "identify samples with computed beam visibility b≥0.5 and b<0.5, "
        "respectively."
    )
    return textwrap.fill(caption, width=205)


def plot_decomposition(records):
    """Create the three-band stacked decomposition figure."""

    cases = ordered_cases(records)
    lookup = record_lookup(records)
    x_position = np.arange(len(cases), dtype=np.float64)
    colors = [color for _, _, color in COMPONENTS]

    reference_band = BANDS[0]
    reference_records = [
        lookup[(terrain, sample, reference_band)]
        for terrain, sample in cases
    ]
    states = [illumination_state(record) for record in reference_records]
    labels = [
        case_label(case, state)
        for case, state in zip(cases, states)
    ]

    figure, axes = plt.subplots(
        len(BANDS),
        1,
        figsize=(17.5, 12.0),
        sharex=True,
    )
    for band_index, (axis, band) in enumerate(zip(axes, BANDS)):
        band_records = [
            lookup[(terrain, sample, band)]
            for terrain, sample in cases
        ]
        component_values = np.stack([
            reflectance_components(record)
            for record in band_records
        ], axis=1)
        total_reflectance = np.asarray([
            record["toa_reflectance"] for record in band_records
        ])

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

        stacked_bars(axis, x_position, component_values, colors)
        axis.plot(
            x_position,
            total_reflectance,
            linestyle="none",
            marker="o",
            markersize=5.5,
            markerfacecolor="none",
            markeredgecolor="black",
            markeredgewidth=1.0,
            zorder=4,
        )
        axis.axhline(0.0, color="0.35", linewidth=0.7)
        axis.set_ylabel("TOA reflectance\ncontribution")
        axis.set_title(band, loc="left", fontweight="bold")
        axis.grid(axis="y", alpha=0.20, zorder=1)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=0.01)

        panel_letter = chr(ord("a") + band_index)
        axis.text(
            0.995,
            0.95,
            f"({panel_letter})",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontweight="bold",
        )

    axes[-1].set_xticks(x_position)
    axes[-1].set_xticklabels(labels, rotation=48, ha="right", fontsize=8)

    component_handles = [
        Patch(facecolor=color, label=label)
        for _, label, color in COMPONENTS
    ]
    context_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="black",
            linestyle="none",
            label="Total TOA reflectance",
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
    ]
    figure.legend(
        handles=component_handles + context_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Sentinel-2 TOA decomposition over idealized mountain terrain",
        y=0.988,
        fontsize=16,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.88,
        bottom=0.245,
        hspace=0.24,
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
    """Read the experiment CSV and save the central decomposition figure."""

    records = select_all_snow_records(read_records(INPUT_CSV))
    figure = plot_decomposition(records)

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(figure)

    print(f"PNG figure: {OUTPUT_PNG}")
    print(f"PDF figure: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
