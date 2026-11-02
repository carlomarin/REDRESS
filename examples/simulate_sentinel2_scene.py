#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulate the central Sentinel-2 pixel over a 20 m DEM.

The L1C SAFE auxiliary files provide water vapour, ozone and, from processing
baseline 04.00 onward, CAMS aerosol optical depth and species. ``MTD_TL.xml``
provides the acquisition time and mean solar and viewing geometry.

Edit the configuration below and run:

    python examples/simulate_sentinel2_scene.py

The DEM supplies terrain context around the selected central pixel. For the
default 2.1 km atmospheric window it must contain at least 105 by 105 cells.
Additional terrain outside this extent improves the horizon calculation.
"""

from pathlib import Path

from redress.simulation.sentinel2_scene import (
    SnowSurface,
    central_pixel_spectrum,
    read_dem,
    read_sentinel2_auxiliary_data,
    simulate_sentinel2_scene,
    validate_central_pixel_context,
    write_central_pixel_csv,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEM_FILE = Path("/path/to/dem_20m_with_terrain_buffer.tif")
L1C_SAFE_FOLDER = Path("/path/to/S2B_MSIL1C_YYYYMMDD...SAFE")
MTD_TL_FILE = Path("/path/to/S2B_MSIL1C_...SAFE/GRANULE/.../MTD_TL.xml")
OUTPUT_DIRECTORY = Path("outputs/sentinel2_central_pixel")

# Small edge gaps introduced by reprojection can be completed from the nearest
# valid elevation. The run stops if any missing cell is farther than this from
# measured terrain.
MAXIMUM_DEM_FILL_DISTANCE_M = 200.0

PLATFORM = "S2B"
BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12")

SNOW = SnowSurface(
    specific_surface_area_m2_kg=20.0,
    black_carbon_ng_g=0.0,
    dust_ug_g=0.0,
)

# Products before processing baseline 04.00 do not contain CAMS aerosol
# fields. Set an explicit AOD550 only for those products; otherwise keep None.
LEGACY_AOD550 = None
LEGACY_AEROSOL_MODEL = "Continental"

HORIZON_HALF_DIRECTIONS = 32  # 64 azimuth directions.
TERRAIN_CONFIGURATION_METHOD = "hemispheric_closure"
TERRAIN_REFLECTION_METHOD = "neighbour_exitance"
TERRAIN_WINDOW_WIDTH_M = 1500.0
ATMOSPHERE_WINDOW_WIDTH_M = 2100.0


def main() -> None:
    """Read the physical inputs and simulate the DEM central pixel."""

    dem = read_dem(
        DEM_FILE,
        fill_missing=True,
        maximum_fill_distance_m=MAXIMUM_DEM_FILL_DISTANCE_M,
    )
    if dem.meta["filled_missing_cells"]:
        print(
            f"Filled {dem.meta['filled_missing_cells']} DEM cells from the "
            "nearest valid elevation; maximum distance "
            f"{dem.meta['maximum_fill_distance_m']:.1f} m"
        )
    validate_central_pixel_context(
        dem,
        maximum_window_width_m=max(
            TERRAIN_WINDOW_WIDTH_M,
            ATMOSPHERE_WINDOW_WIDTH_M,
        ),
    )
    auxiliary = read_sentinel2_auxiliary_data(
        L1C_SAFE_FOLDER,
        MTD_TL_FILE,
        dem,
        band_names=BANDS,
        legacy_aod550=LEGACY_AOD550,
        legacy_aerosol_model=LEGACY_AEROSOL_MODEL,
    )

    print(
        "Central pixel: "
        f"{auxiliary.site.latitude_deg:.6f}°, "
        f"{auxiliary.site.longitude_deg:.6f}°, "
        f"{auxiliary.site.elevation_m:.1f} m"
    )
    print(f"Acquisition: {auxiliary.acquisition_time.isoformat()}")
    print(
        "Geometry: "
        f"SZA={auxiliary.geometry.solar_zenith_deg:.3f}°, "
        f"SAA={auxiliary.geometry.solar_azimuth_deg:.3f}°, "
        f"VZA={auxiliary.geometry.view_zenith_deg:.3f}°, "
        f"VAA={auxiliary.geometry.view_azimuth_deg:.3f}°"
    )
    print(
        "Atmosphere: "
        f"AOD550={auxiliary.atmosphere.aod550:.4f}, "
        f"water={auxiliary.atmosphere.water_vapour_g_cm2:.4f} g cm-2, "
        f"ozone={auxiliary.atmosphere.ozone_cm_atm:.4f} cm-atm"
    )

    scene = simulate_sentinel2_scene(
        dem,
        auxiliary.geometry,
        auxiliary.atmosphere,
        SNOW,
        platform=PLATFORM,
        band_names=BANDS,
        case_number=5,
        horizon_half_directions=HORIZON_HALF_DIRECTIONS,
        terrain_configuration_method=TERRAIN_CONFIGURATION_METHOD,
        terrain_reflection_method=TERRAIN_REFLECTION_METHOD,
        terrain_window_width_m=TERRAIN_WINDOW_WIDTH_M,
        atmosphere_window_width_m=ATMOSPHERE_WINDOW_WIDTH_M,
        # AUX columns were corrected to the central DEM elevation. Keeping
        # this disabled prevents applying one corrected column to other
        # elevation nodes or correcting it twice.
        atmosphere_altitude_step_m=None,
        verbose=True,
    )

    spectrum = central_pixel_spectrum(scene, dem)
    output_csv = write_central_pixel_csv(
        spectrum,
        OUTPUT_DIRECTORY / "sentinel2_central_pixel_spectrum.csv",
    )
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_netcdf = OUTPUT_DIRECTORY / "sentinel2_central_pixel_spectrum.nc"
    spectrum.to_netcdf(output_netcdf, engine="netcdf4")

    print("\nSimulated central-pixel spectrum:")
    print(
        spectrum[["toa_reflectance", "toa_radiance"]]
        .to_dataframe()
        .reset_index()[
            ["band", "wavelength_nm", "toa_reflectance", "toa_radiance"]
        ]
        .to_string(index=False)
    )
    print(f"\nCSV: {output_csv}")
    print(f"NetCDF: {output_netcdf.resolve()}")


if __name__ == "__main__":
    main()
