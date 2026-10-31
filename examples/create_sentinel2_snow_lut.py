#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create and plot a small Sentinel-2 snow endmember LUT.

Edit the parameters in the first section and run the script from the REDRESS
repository root. The outputs are a NetCDF LUT, a tidy CSV table and a figure
that isolates the effects of snow SSA and observation geometry.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from redress.lut import create_sentinel2_snow_brf_lut


# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------

PLATFORM = "S2B"

# B01, B09 and B10 are omitted because they are primarily intended for
# aerosol, water-vapour and cirrus characterization.
BANDS = (
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
)

SPECIFIC_SURFACE_AREA = (5.0, 20.0, 60.0)       # m2 kg-1
SOLAR_ZENITH_ANGLE = (20.0, 45.0, 65.0)         # degree
VIEW_ZENITH_ANGLE = (0.0, 15.0)                 # degree
RELATIVE_AZIMUTH_ANGLE = (0.0, 90.0, 180.0)     # degree

REFERENCE_SSA = 20.0
REFERENCE_SZA = 45.0
REFERENCE_VZA = 0.0
REFERENCE_RAA = 0.0

OUTPUT_DIRECTORY = Path("outputs/sentinel2_snow_lut")


def plot_diagnostic_categories(lut, output_file):
    """Plot SSA, illumination and viewing-geometry effects separately."""

    wavelength = lut["wavelength_nm"].values
    bands = lut["band"].values

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(16, 5.2),
        sharey=True,
        constrained_layout=True,
    )

    colors = plt.cm.viridis(
        np.linspace(0.1, 0.9, len(SPECIFIC_SURFACE_AREA))
    )
    for ssa, color in zip(SPECIFIC_SURFACE_AREA, colors):
        spectrum = lut["brf"].sel(
            specific_surface_area=ssa,
            solar_zenith_angle=REFERENCE_SZA,
            view_zenith_angle=REFERENCE_VZA,
            relative_azimuth_angle=REFERENCE_RAA,
        )
        axes[0].plot(
            wavelength, spectrum, marker="o", color=color,
            label=f"SSA = {ssa:g} m² kg⁻¹",
        )
    axes[0].set_title(
        f"Snow SSA\nSZA={REFERENCE_SZA:g}°, VZA={REFERENCE_VZA:g}°"
    )

    colors = plt.cm.plasma(
        np.linspace(0.1, 0.9, len(SOLAR_ZENITH_ANGLE))
    )
    for sza, color in zip(SOLAR_ZENITH_ANGLE, colors):
        spectrum = lut["brf"].sel(
            specific_surface_area=REFERENCE_SSA,
            solar_zenith_angle=sza,
            view_zenith_angle=REFERENCE_VZA,
            relative_azimuth_angle=REFERENCE_RAA,
        )
        axes[1].plot(
            wavelength, spectrum, marker="o", color=color,
            label=f"SZA = {sza:g}°",
        )
    axes[1].set_title(
        f"Solar geometry\nSSA={REFERENCE_SSA:g} m² kg⁻¹, "
        f"VZA={REFERENCE_VZA:g}°"
    )

    geometry_pairs = []
    for vza in VIEW_ZENITH_ANGLE:
        # Relative azimuth has no effect for a nadir view, so plot it once.
        azimuth_values = ((REFERENCE_RAA,) if vza == 0
                          else RELATIVE_AZIMUTH_ANGLE)
        geometry_pairs.extend((vza, raa) for raa in azimuth_values)

    colors = plt.cm.tab10(np.linspace(0, 1, len(geometry_pairs)))
    for (vza, raa), color in zip(geometry_pairs, colors):
        spectrum = lut["brf"].sel(
            specific_surface_area=REFERENCE_SSA,
            solar_zenith_angle=REFERENCE_SZA,
            view_zenith_angle=vza,
            relative_azimuth_angle=raa,
        )
        axes[2].plot(
            wavelength,
            spectrum,
            marker="o",
            color=color,
            label=f"VZA={vza:g}°, RAA={raa:g}°",
        )
    axes[2].set_title(
        f"Viewing geometry\nSSA={REFERENCE_SSA:g} m² kg⁻¹, "
        f"SZA={REFERENCE_SZA:g}°"
    )

    tick_labels = [
        f"{band}\n{central_wavelength:.0f}"
        for band, central_wavelength in zip(bands, wavelength)
    ]
    for axis in axes:
        axis.set_xticks(wavelength)
        axis.set_xticklabels(tick_labels, rotation=45, ha="right")
        axis.set_xlabel("Sentinel-2 band and effective wavelength (nm)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        axis.set_ylim(bottom=0)

    axes[0].set_ylabel("Bidirectional reflectance factor")
    figure.suptitle(
        f"REDRESS {PLATFORM} snow endmember sensitivity",
        fontsize=14,
    )
    figure.savefig(output_file, dpi=250, bbox_inches="tight")
    plt.close(figure)


def main():
    """Generate the LUT and diagnostic outputs."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    lut = create_sentinel2_snow_brf_lut(
        platform=PLATFORM,
        band_names=BANDS,
        specific_surface_area=SPECIFIC_SURFACE_AREA,
        solar_zenith_angle=SOLAR_ZENITH_ANGLE,
        view_zenith_angle=VIEW_ZENITH_ANGLE,
        relative_azimuth_angle=RELATIVE_AZIMUTH_ANGLE,
    )

    netcdf_file = OUTPUT_DIRECTORY / f"{PLATFORM}_snow_brf_lut.nc"
    csv_file = OUTPUT_DIRECTORY / f"{PLATFORM}_snow_brf_lut.csv"
    figure_file = OUTPUT_DIRECTORY / f"{PLATFORM}_snow_brf_sensitivity.png"

    lut.to_netcdf(
        netcdf_file,
        encoding={"brf": {"dtype": "float32", "zlib": True, "complevel": 4}},
    )
    lut["brf"].to_dataframe(name="brf").reset_index().to_csv(
        csv_file, index=False
    )
    plot_diagnostic_categories(lut, figure_file)

    print(lut)
    print(f"NetCDF LUT: {netcdf_file.resolve()}")
    print(f"CSV LUT:    {csv_file.resolve()}")
    print(f"Figure:     {figure_file.resolve()}")


if __name__ == "__main__":
    main()
