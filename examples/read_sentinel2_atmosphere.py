#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read site atmospheric inputs from a Sentinel-2 L1C SAFE product.

Edit the constants below, then run this script from the repository root with
``python examples/read_sentinel2_atmosphere.py``.
"""

from pathlib import Path

from redress.readers.sentinel2_atmosphere import (
    Sentinel2Site,
    read_sentinel2_l1c_atmosphere,
)


SAFE_FOLDER = Path(
    "/path/to/S2B_MSIL1C_YYYYMMDDTHHMMSS_N0511_R000_T00XXX_*.SAFE"
)
SITE = Sentinel2Site(
    name="research_site",
    latitude_deg=46.0,
    longitude_deg=11.0,
    elevation_m=2000.0,
)

# These values are used only for processing baselines older than 04.00.
LEGACY_AOD550 = None
LEGACY_AEROSOL_MODEL = "Continental"

# Scale-height approximation applied from the CAMS surface to SITE.elevation_m.
APPLY_ELEVATION_CORRECTION = True
WATER_VAPOUR_SCALE_HEIGHT_M = 2000.0
AEROSOL_SCALE_HEIGHT_M = 1500.0


def main():
    """Read the auxiliary fields and print Py6S-ready values."""

    atmosphere = read_sentinel2_l1c_atmosphere(
        SAFE_FOLDER,
        SITE,
        apply_elevation_correction=APPLY_ELEVATION_CORRECTION,
        water_scale_height_m=WATER_VAPOUR_SCALE_HEIGHT_M,
        aerosol_scale_height_m=AEROSOL_SCALE_HEIGHT_M,
        legacy_aod550=LEGACY_AOD550,
        legacy_aerosol_model=LEGACY_AEROSOL_MODEL,
    )

    print(f"Processing baseline: {atmosphere.processing_baseline}")
    print(
        "CAMS model-surface elevation: "
        f"{atmosphere.model_surface_elevation_m} m"
    )
    print(
        "Water vapour: "
        f"{atmosphere.water_vapour_g_cm2:.6f} g cm-2 "
        f"(factor {atmosphere.water_elevation_factor:.6f})"
    )
    print(f"Ozone: {atmosphere.ozone_cm_atm:.6f} cm-atm")
    print(
        f"AOD550: {atmosphere.aod550:.6f} "
        f"(factor {atmosphere.aerosol_elevation_factor:.6f})"
    )
    print(f"Aerosol source: {atmosphere.aerosol_source}")
    print(f"Aerosol components: {atmosphere.aerosol_components}")
    print("REDRESS rtmodel.run keywords:")
    print(atmosphere.redress_rt_options())
    for note in atmosphere.notes:
        print(f"Note: {note}")


if __name__ == "__main__":
    main()
