#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulate a Sentinel-2 TOA snow image over a DEM.

Edit the constants in the configuration section and run:

    python examples/simulate_sentinel2_scene.py

The DEM must be projected in metres, north-up, and use square pixels. It
should include a terrain buffer outside the final area of interest so that
horizons and neighbouring-slope illumination are not truncated at its edge.
"""

from pathlib import Path

from redress.simulation.sentinel2_scene import (
    Sentinel2Atmosphere,
    Sentinel2Geometry,
    SnowSurface,
    read_dem,
    simulate_sentinel2_scene,
    write_multiband_geotiff,
    write_scene_netcdf,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEM_FILE = Path("/path/to/buffered_dem_20m.tif")
OUTPUT_DIRECTORY = Path("outputs/sentinel2_scene")

PLATFORM = "S2B"
BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12")

# Sentinel-2 metadata angles. Azimuth is clockwise from geographic north.
GEOMETRY = Sentinel2Geometry(
    solar_zenith_deg=45.0,
    solar_azimuth_deg=180.0,
    view_zenith_deg=5.0,
    view_azimuth_deg=0.0,
)

# Py6S units: water vapour in g cm-2, ozone in cm-atm, AOD at 550 nm.
ATMOSPHERE = Sentinel2Atmosphere(
    aerosol="Continental",
    aod550=0.10,
    water_vapour_g_cm2=1.0,
    ozone_cm_atm=0.30,
    month=3,
    day=28,
)

SNOW = SnowSurface(
    specific_surface_area_m2_kg=20.0,
    black_carbon_ng_g=0.0,
    dust_ug_g=0.0,
)

# N=32 produces 64 horizon directions. Neighbour exitance is currently the
# practical choice for an image-sized raster. Visible exitance can be used on
# small tiles while its view-factor geometry is being optimized and validated.
HORIZON_HALF_DIRECTIONS = 32
TERRAIN_CONFIGURATION_METHOD = "hemispheric_closure"
TERRAIN_REFLECTION_METHOD = "neighbour_exitance"
TERRAIN_WINDOW_WIDTH_M = 1500.0
ATMOSPHERE_WINDOW_WIDTH_M = 2100.0

# None uses one scene-mean atmospheric calculation. A value such as 250.0
# interpolates 6S outputs over elevation. Leave this disabled when the supplied
# water and aerosol columns have already been corrected to one site elevation.
ATMOSPHERE_ALTITUDE_STEP_M = None


def main() -> None:
    """Run the simulation and write QGIS-compatible outputs."""

    dem = read_dem(DEM_FILE)
    scene = simulate_sentinel2_scene(
        dem,
        GEOMETRY,
        ATMOSPHERE,
        SNOW,
        platform=PLATFORM,
        band_names=BANDS,
        case_number=5,
        horizon_half_directions=HORIZON_HALF_DIRECTIONS,
        terrain_configuration_method=TERRAIN_CONFIGURATION_METHOD,
        terrain_reflection_method=TERRAIN_REFLECTION_METHOD,
        terrain_window_width_m=TERRAIN_WINDOW_WIDTH_M,
        atmosphere_window_width_m=ATMOSPHERE_WINDOW_WIDTH_M,
        atmosphere_altitude_step_m=ATMOSPHERE_ALTITUDE_STEP_M,
        verbose=True,
    )

    netcdf_file = write_scene_netcdf(
        scene, dem, OUTPUT_DIRECTORY / "sentinel2_redress_scene.nc"
    )
    reflectance_file = write_multiband_geotiff(
        scene,
        dem,
        OUTPUT_DIRECTORY / "sentinel2_toa_reflectance.tif",
        variable="toa_reflectance",
    )
    radiance_file = write_multiband_geotiff(
        scene,
        dem,
        OUTPUT_DIRECTORY / "sentinel2_toa_radiance.tif",
        variable="toa_radiance",
    )

    print(f"NetCDF: {netcdf_file}")
    print(f"TOA reflectance: {reflectance_file}")
    print(f"TOA radiance: {radiance_file}")


if __name__ == "__main__":
    main()
