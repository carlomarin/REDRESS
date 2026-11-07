#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulate a central window of Sentinel-2 pixels and compare with L1C.

The source DEM, typically much finer than 20 m, is averaged onto the
Sentinel-2 20 m tile grid so each simulated pixel matches one observed pixel.
REDRESS runs over the whole DEM, and only pixels at least
``CONTEXT_MARGIN_M`` from its edges are kept. The script writes multiband
GeoTIFFs on the same grid, simulated and observed L1C TOA reflectance,
prints a per-band comparison and draws it with ``plot_sentinel2_window.py``.
With a fractional snow cover reference, the statistics use only pure-snow
pixels, since REDRESS models snow everywhere.

Run, for example:

    python examples/simulate_sentinel2_window.py \\
        --source-dem /path/to/dem.tif \\
        --safe /path/to/S2B_MSIL1C_...SAFE \\
        --snow-fraction /path/to/snow_cover_fraction_20m.tif \\
        --output-directory outputs/sentinel2_window
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import xarray as xr

from plot_sentinel2_window import plot_window_check
from redress.optics.snow import DUST_MODELS, SNOW_ALBEDO_MODELS
from redress.simulation.sentinel2_scene import (
    SnowSurface,
    crop_scene,
    edge_independent_window,
    read_dem,
    read_l1c_toa_reflectance,
    read_mtd_tl_grid,
    read_sentinel2_auxiliary_data,
    read_snow_fraction_on_grid,
    reflectance_comparison_table,
    resample_dem_to_sentinel2_grid,
    simulate_sentinel2_scene,
    write_multiband_geotiff,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PLATFORM = "S2B"
BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12")
GRID_RESOLUTION_M = 20.0

# TARTES supplies the snow albedo because the asymptotic ART albedo is up
# to 1.5 times too dark in B11 and B12 near normal incidence. Over Weissee
# on 2022-01-13, a TARTES SSA sweep from 20 to 60 m2 kg-1 put the zero bias
# near 44 in B11 and 33 in B12; 40 gives -0.006 and +0.011 on pure snow.
# With ART, the same fit needed SSA 60 and the bands disagreed more.
SNOW = SnowSurface(
    specific_surface_area_m2_kg=40.0,
    black_carbon_ng_g=0.0,
    dust_ug_g=0.0,
    albedo_model="tartes",
)

# Products before processing baseline 04.00 do not contain CAMS aerosol
# fields. Set an explicit AOD550 only for those products; otherwise keep None.
LEGACY_AOD550 = None
LEGACY_AEROSOL_MODEL = "Continental"

MAXIMUM_DEM_FILL_DISTANCE_M = 200.0
HORIZON_HALF_DIRECTIONS = 32  # 64 azimuth directions.
HORIZON_METHOD = "ray_march"  # Exact on the bilinearly interpolated DEM.
TERRAIN_CONFIGURATION_METHOD = "hemispheric_closure"
TERRAIN_REFLECTION_METHOD = "visible_exitance"
TERRAIN_VISIBILITY_RADIUS_M = 3000.0
TERRAIN_WINDOW_WIDTH_M = 1500.0
ATMOSPHERE_WINDOW_WIDTH_M = 2100.0

# Distance kept from the DEM edges. Half the atmosphere window (1050 m)
# keeps every moving window inside the DEM. Over the Weissee DEM, trimming
# the DEM by 800 m changed the 95th-percentile reflectance by at most 0.015
# beyond 600 m from the new edge, against up to 0.09 closer in, where cut
# horizons and missing sunlit neighbours dominate. None restores the
# strict half-window margin.
CONTEXT_MARGIN_M = 600.0

# Minimum fractional snow cover of a pixel counted as pure snow.
PURE_SNOW_MIN_FRACTION = 0.999


def parse_arguments() -> argparse.Namespace:
    """Read the input and output locations from the command line."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--source-dem", type=Path, required=True,
        help="projected DEM in metres, usually finer than 20 m",
    )
    parser.add_argument(
        "--safe", type=Path, required=True,
        help="Sentinel-2 L1C .SAFE folder",
    )
    parser.add_argument(
        "--snow-fraction", type=Path, default=None,
        help="fractional snow cover (0-1 or percent) on the 20 m tile grid",
    )
    parser.add_argument(
        "--ssa", type=float, default=SNOW.specific_surface_area_m2_kg,
        help="snow specific surface area in m2 kg-1 (default %(default)g)",
    )
    parser.add_argument(
        "--snow-albedo", choices=SNOW_ALBEDO_MODELS,
        default=SNOW.albedo_model,
        help="snow albedo model; the BRF angular shape is ART in both "
        "cases (default %(default)s)",
    )
    parser.add_argument(
        "--dust", type=float, default=SNOW.dust_ug_g,
        help="mineral dust in the snow in microgram g-1 (default %(default)g)",
    )
    parser.add_argument(
        "--black-carbon", type=float, default=SNOW.black_carbon_ng_g,
        help="black carbon in the snow in ng g-1 (default %(default)g)",
    )
    parser.add_argument(
        "--dust-model", choices=DUST_MODELS, default=SNOW.dust_model,
        help="dust refractive index; muller2011 absorbs more than "
        "skiles2014 (default %(default)s)",
    )
    parser.add_argument(
        "--aod550", type=float, default=None,
        help="replace the CAMS aerosol optical depth at 550 nm, for example "
        "with a sun-photometer value at the scene elevation; the CAMS "
        "aerosol mixture is kept",
    )
    parser.add_argument(
        "--water-vapour", type=float, default=None,
        help="replace the ECMWF precipitable water vapour in g cm-2 "
        "(numerically cm), for example a sun-photometer value at the "
        "scene elevation",
    )
    parser.add_argument(
        "--output-directory", type=Path, required=True,
        help="folder for the aligned DEM, GeoTIFFs and comparison table",
    )
    return parser.parse_args()


def main() -> None:
    """Simulate the edge-independent window and compare it with L1C."""

    arguments = parse_arguments()
    output_directory = arguments.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    safe_folder = arguments.safe.expanduser().resolve()
    mtd_tl_files = sorted(safe_folder.glob("GRANULE/*/MTD_TL.xml"))
    if len(mtd_tl_files) != 1:
        raise FileNotFoundError(f"expected one MTD_TL.xml in {safe_folder}")
    mtd_tl_file = mtd_tl_files[0]

    grid = read_mtd_tl_grid(mtd_tl_file, GRID_RESOLUTION_M)
    aligned_dem = resample_dem_to_sentinel2_grid(
        arguments.source_dem,
        output_directory / "dem_sentinel2_grid.tif",
        grid,
    )
    dem = read_dem(
        aligned_dem,
        fill_missing=True,
        maximum_fill_distance_m=MAXIMUM_DEM_FILL_DISTANCE_M,
    )
    rows, columns = edge_independent_window(
        dem,
        max(TERRAIN_WINDOW_WIDTH_M, ATMOSPHERE_WINDOW_WIDTH_M),
        margin_m=CONTEXT_MARGIN_M,
    )
    print(
        f"DEM on the Sentinel-2 grid: {dem.bands['altitude'].shape[0]} by "
        f"{dem.bands['altitude'].shape[1]} pixels; "
        f"{dem.meta['filled_missing_cells']} cells filled"
    )
    print(
        f"Comparison window: {rows.stop - rows.start} by "
        f"{columns.stop - columns.start} pixels "
        f"(rows {rows.start}-{rows.stop - 1}, "
        f"columns {columns.start}-{columns.stop - 1})"
    )

    auxiliary = read_sentinel2_auxiliary_data(
        safe_folder,
        mtd_tl_file,
        dem,
        band_names=BANDS,
        legacy_aod550=LEGACY_AOD550,
        legacy_aerosol_model=LEGACY_AEROSOL_MODEL,
    )
    atmosphere = auxiliary.atmosphere
    if arguments.aod550 is not None:
        atmosphere = replace(atmosphere, aod550=arguments.aod550)
    if arguments.water_vapour is not None:
        atmosphere = replace(
            atmosphere, water_vapour_g_cm2=arguments.water_vapour
        )
    atmosphere.validate()
    print(f"Acquisition: {auxiliary.acquisition_time.isoformat()}")
    print(
        f"Water vapour: {atmosphere.water_vapour_g_cm2:.3f} g cm-2"
        + (" (ECMWF)" if arguments.water_vapour is None else
           f" (set; ECMWF {auxiliary.atmosphere.water_vapour_g_cm2:.3f})")
    )
    print(
        f"AOD550: {atmosphere.aod550:.4f}"
        + (" (CAMS)" if arguments.aod550 is None else
           f" (set; CAMS {auxiliary.atmosphere.aod550:.4f})")
    )
    print(
        f"Snow SSA: {arguments.ssa:g} m2 kg-1 "
        f"(optical diameter {6.0e3 / (917.0 * arguments.ssa):.2f} mm), "
        f"{arguments.snow_albedo.upper()} albedo; "
        f"dust {arguments.dust:g} µg g-1 ({arguments.dust_model}), "
        f"black carbon {arguments.black_carbon:g} ng g-1"
    )
    print(
        "Solar geometry: "
        f"SZA={auxiliary.geometry.solar_zenith_deg:.3f}°, "
        f"SAA={auxiliary.geometry.solar_azimuth_deg:.3f}°"
    )
    print("Band mean viewing geometry:")
    for band in BANDS:
        view_zenith, view_azimuth = (
            auxiliary.geometry.viewing_for_band(band)
        )
        print(
            f"  {band}: VZA={view_zenith:.3f}°, "
            f"VAA={view_azimuth:.3f}°"
        )

    scene = simulate_sentinel2_scene(
        dem,
        auxiliary.geometry,
        atmosphere,
        replace(
            SNOW,
            specific_surface_area_m2_kg=arguments.ssa,
            albedo_model=arguments.snow_albedo,
            black_carbon_ng_g=arguments.black_carbon,
            dust_ug_g=arguments.dust,
            dust_model=arguments.dust_model,
        ),
        platform=PLATFORM,
        band_names=BANDS,
        case_number=5,
        horizon_half_directions=HORIZON_HALF_DIRECTIONS,
        horizon_method=HORIZON_METHOD,
        terrain_configuration_method=TERRAIN_CONFIGURATION_METHOD,
        terrain_reflection_method=TERRAIN_REFLECTION_METHOD,
        terrain_window_width_m=TERRAIN_WINDOW_WIDTH_M,
        atmosphere_window_width_m=ATMOSPHERE_WINDOW_WIDTH_M,
        # AUX columns were corrected to the central DEM elevation. Keeping
        # this disabled prevents applying one corrected column to other
        # elevation nodes or correcting it twice.
        atmosphere_altitude_step_m=None,
        terrain_visibility_radius_m=TERRAIN_VISIBILITY_RADIUS_M,
        verbose=False,
    )
    simulated = crop_scene(scene, rows, columns)
    observed = read_l1c_toa_reflectance(
        safe_folder,
        grid,
        simulated["x"].values,
        simulated["y"].values,
        band_names=BANDS,
    )

    simulated_tif = write_multiband_geotiff(
        simulated, dem, output_directory / "redress_toa_reflectance.tif"
    )
    observed_tif = write_multiband_geotiff(
        observed, dem, output_directory / "l1c_toa_reflectance.tif"
    )
    mask = None
    if arguments.snow_fraction is not None:
        fraction = read_snow_fraction_on_grid(
            arguments.snow_fraction,
            simulated["x"].values,
            simulated["y"].values,
        )
        mask = fraction >= PURE_SNOW_MIN_FRACTION
        fraction_tif = write_multiband_geotiff(
            xr.Dataset(
                {"snow_fraction": (
                    ("band", "y", "x"), fraction[np.newaxis]
                )},
                coords={
                    "band": ["snow_fraction"],
                    "x": simulated["x"].values,
                    "y": simulated["y"].values,
                },
            ),
            dem,
            output_directory / "snow_fraction.tif",
            variable="snow_fraction",
        )
        print(
            f"\nPure snow (fraction >= {PURE_SNOW_MIN_FRACTION}): "
            f"{int(mask.sum())} of {mask.size} pixels; "
            f"{int(np.isnan(fraction).sum())} without a reference"
        )
        print(f"Snow fraction GeoTIFF: {fraction_tif}")
    table = reflectance_comparison_table(simulated, observed, mask=mask)
    table_csv = output_directory / "reflectance_comparison.csv"
    table.to_csv(table_csv, index=False)

    print("\nSimulated minus observed TOA reflectance:")
    print(table.to_string(index=False, float_format="{:.4f}".format))
    print(f"\nSimulated GeoTIFF: {simulated_tif}")
    print(f"Observed GeoTIFF:  {observed_tif}")
    print(f"Comparison table:  {table_csv}")
    figure_png = plot_window_check(
        output_directory,
        pure_snow_min_fraction=PURE_SNOW_MIN_FRACTION,
        title=(
            f"SSA {arguments.ssa:g} m2 kg-1, "
            f"{arguments.snow_albedo.upper()} albedo, "
            f"dust {arguments.dust:g} µg g-1 ({arguments.dust_model}), "
            f"BC {arguments.black_carbon:g} ng g-1, "
            f"AOD550 {atmosphere.aod550:.3f}, "
            f"water vapour {atmosphere.water_vapour_g_cm2:.2f} g cm-2"
        ),
    )
    print(f"Comparison figure: {figure_png}")


if __name__ == "__main__":
    main()
