#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare a simulated Sentinel-2 window with L1C, map by map and pixel by pixel.

Reads the GeoTIFFs written by ``simulate_sentinel2_window.py`` in one output
folder and draws, for each band, the REDRESS and L1C TOA reflectance on a
shared grey scale, their difference over pure snow and a density scatter
plot of the pure-snow pixels with bias, RMSE and correlation.

Run, for example:

    python examples/plot_sentinel2_window.py outputs/sentinel2_window
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from osgeo import gdal

gdal.UseExceptions()

# B02 and B04 show the impurity signal, B8A the NIR and B11 the grain size.
DEFAULT_BANDS = ("B02", "B04", "B8A", "B11")
PURE_SNOW_MIN_FRACTION = 0.999
OUTPUT_NAME = "window_check.png"


def read_bands(path: Path) -> tuple[dict[str, np.ndarray], tuple]:
    """Read a band-named GeoTIFF into a dictionary and its geotransform."""

    raster = gdal.Open(str(path))
    if raster is None:
        raise FileNotFoundError(path)
    bands = {}
    for index in range(1, raster.RasterCount + 1):
        band = raster.GetRasterBand(index)
        bands[band.GetDescription()] = band.ReadAsArray().astype(np.float64)
    return bands, raster.GetGeoTransform()


def plot_window_check(
    output_directory: Path,
    bands=DEFAULT_BANDS,
    pure_snow_min_fraction: float = PURE_SNOW_MIN_FRACTION,
    title: str | None = None,
) -> Path:
    """Draw the map and scatter comparison and return the PNG path."""

    output_directory = Path(output_directory)
    simulated, transform = read_bands(
        output_directory / "redress_toa_reflectance.tif"
    )
    observed, _ = read_bands(output_directory / "l1c_toa_reflectance.tif")
    fraction_tif = output_directory / "snow_fraction.tif"
    shape = next(iter(simulated.values())).shape
    if fraction_tif.exists():
        fraction = read_bands(fraction_tif)[0]["snow_fraction"]
        mask = fraction >= pure_snow_min_fraction
        mask_label = "pure snow"
    else:
        mask = np.ones(shape, dtype=bool)
        mask_label = "all pixels"

    # Map axes in km from the upper-left corner of the window.
    extent = (
        0.0,
        shape[1] * transform[1] / 1000.0,
        shape[0] * abs(transform[5]) / 1000.0,
        0.0,
    )
    grey = plt.get_cmap("gray").copy()
    grey.set_bad("#d9d9d9")
    diverging = plt.get_cmap("RdBu_r").copy()
    diverging.set_bad("#d9d9d9")

    figure, axes = plt.subplots(
        len(bands), 4, figsize=(15.5, 3.6 * len(bands)),
        constrained_layout=True, squeeze=False,
    )
    for row, band in zip(axes, bands):
        sim = simulated[band]
        obs = observed[band]
        valid = mask & np.isfinite(sim) & np.isfinite(obs)

        both = np.concatenate([sim[np.isfinite(sim)], obs[np.isfinite(obs)]])
        low, high = np.percentile(both, [1.0, 99.5])
        for axis, image, name in (
            (row[0], sim, "REDRESS"), (row[1], obs, "L1C")
        ):
            shown = axis.imshow(
                image, cmap=grey, vmin=low, vmax=high, extent=extent,
                interpolation="nearest",
            )
            axis.set_title(f"{band} {name}")
        figure.colorbar(shown, ax=row[:2], shrink=0.85,
                        label="TOA reflectance")

        difference = np.where(valid, sim - obs, np.nan)
        limit = float(np.nanpercentile(np.abs(difference), 98.0))
        shown = row[2].imshow(
            difference, cmap=diverging, vmin=-limit, vmax=limit,
            extent=extent, interpolation="nearest",
        )
        row[2].set_title(f"{band} REDRESS − L1C, {mask_label}")
        figure.colorbar(shown, ax=row[2], shrink=0.85)
        for axis in row[:3]:
            axis.set_xlabel("km")
            axis.set_ylabel("km")

        x, y = obs[valid], sim[valid]
        top = float(max(np.percentile(x, 99.9), np.percentile(y, 99.9)))
        top *= 1.05
        density = row[3].hexbin(
            x, y, gridsize=70, extent=(0.0, top, 0.0, top), mincnt=1,
            norm=LogNorm(), cmap="viridis", linewidths=0.0,
        )
        row[3].plot([0.0, top], [0.0, top], color="black", linewidth=1.0,
                    linestyle="--")
        row[3].set_xlim(0.0, top)
        row[3].set_ylim(0.0, top)
        row[3].set_aspect("equal")
        row[3].set_xlabel("L1C")
        row[3].set_ylabel("REDRESS")
        row[3].set_title(f"{band}, {mask_label}")
        figure.colorbar(density, ax=row[3], shrink=0.85, label="pixels")

        error = y - x
        statistics = (
            f"n = {x.size}\n"
            f"bias = {error.mean():+.3f}\n"
            f"RMSE = {np.sqrt(np.mean(error ** 2)):.3f}\n"
            f"r = {np.corrcoef(x, y)[0, 1]:.3f}"
        )
        row[3].text(
            0.04, 0.96, statistics, transform=row[3].transAxes,
            va="top", ha="left", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#bbbbbb",
                  "alpha": 0.85},
        )

    figure.suptitle(title or output_directory.name)
    output_png = output_directory / OUTPUT_NAME
    figure.savefig(output_png, dpi=110)
    plt.close(figure)
    return output_png


def parse_arguments() -> argparse.Namespace:
    """Read the output folders and bands from the command line."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "output_directories", type=Path, nargs="+",
        help="folders written by simulate_sentinel2_window.py",
    )
    parser.add_argument(
        "--bands", nargs="+", default=list(DEFAULT_BANDS),
        help="bands to draw, one row each (default %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    """Draw the comparison for each output folder."""

    arguments = parse_arguments()
    for directory in arguments.output_directories:
        print(plot_window_check(directory, arguments.bands))


if __name__ == "__main__":
    main()
