#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a diagnostic Sentinel-2 TOA snow endmember LUT with REDRESS.

The experiment changes one physical factor at a time around a common baseline.
This makes the spectra easier to interpret and avoids an unnecessarily large
Cartesian LUT during the first scientific checks.
"""

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from redress.lut import (
    Sentinel2EndmemberScenario,
    create_sentinel2_toa_lut,
)


# ---------------------------------------------------------------------------
# Sensor, acquisition and output configuration
# ---------------------------------------------------------------------------

PLATFORM = "S2B"
BANDS = (
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
)

# Example date corresponding to the Peteroa acquisition used in earlier tests.
MONTH = 3
DAY = 28

AEROSOL_MODEL = "Continental"
ATMOSPHERIC_PROFILE = None
REDRESS_CASE_NUMBER = 5
SCENE_SIZE = 9
PIXEL_RESOLUTION_M = 20.0
TERRAIN_WINDOW_WIDTH_M = 1500.0
ATMOSPHERE_WINDOW_WIDTH_M = 2100.0

OUTPUT_DIRECTORY = Path("outputs/sentinel2_toa_lut")


# ---------------------------------------------------------------------------
# Baseline and one-factor-at-a-time scenarios
# ---------------------------------------------------------------------------

BASELINE = Sentinel2EndmemberScenario(
    name="baseline",
    category="Baseline",
    specific_surface_area=20.0,
    black_carbon_ng_g=0.0,
    dust_ug_g=0.0,
    dust_density_kg_m3=2600.0,
    dust_model="muller2011",
    solar_zenith_angle=45.0,
    local_incidence_angle=45.0,
    view_zenith_angle=0.0,
    local_view_zenith_angle=0.0,
    relative_azimuth_angle=0.0,
    altitude_m=2000.0,
    aod550=0.10,
    water_vapour_g_cm2=1.0,
    ozone_cm_atm=0.30,
    sky_view_factor=1.0,
    terrain_configuration_factor=0.0,
    shadowed=False,
)

SCENARIOS = (
    BASELINE,
    replace(BASELINE, name="ssa_5", category="Snow SSA",
            specific_surface_area=5.0),
    replace(BASELINE, name="ssa_60", category="Snow SSA",
            specific_surface_area=60.0),
    replace(BASELINE, name="bc_50", category="LAP",
            black_carbon_ng_g=50.0),
    replace(BASELINE, name="bc_200", category="LAP",
            black_carbon_ng_g=200.0),
    replace(BASELINE, name="dust_20", category="LAP",
            dust_ug_g=20.0),
    replace(BASELINE, name="dust_200", category="LAP",
            dust_ug_g=200.0),
    replace(BASELINE, name="sza_20", category="Solar zenith",
            solar_zenith_angle=20.0, local_incidence_angle=20.0),
    replace(BASELINE, name="sza_65", category="Solar zenith",
            solar_zenith_angle=65.0, local_incidence_angle=65.0),
    replace(BASELINE, name="incidence_20", category="Local incidence",
            local_incidence_angle=20.0),
    replace(BASELINE, name="incidence_70", category="Local incidence",
            local_incidence_angle=70.0),
    replace(BASELINE, name="aod_003", category="Atmosphere", aod550=0.03),
    replace(BASELINE, name="aod_025", category="Atmosphere", aod550=0.25),
    replace(BASELINE, name="water_02", category="Atmosphere",
            water_vapour_g_cm2=0.2),
    replace(BASELINE, name="water_20", category="Atmosphere",
            water_vapour_g_cm2=2.0),
    replace(BASELINE, name="altitude_0", category="Altitude and terrain",
            altitude_m=0.0),
    replace(BASELINE, name="altitude_4000", category="Altitude and terrain",
            altitude_m=4000.0),
    replace(BASELINE, name="cast_shadow", category="Altitude and terrain",
            shadowed=True, sky_view_factor=0.70,
            terrain_configuration_factor=0.30),
    replace(BASELINE, name="terrain_reillumination",
            category="Altitude and terrain", sky_view_factor=0.85,
            terrain_configuration_factor=0.15),
    replace(BASELINE, name="view_15_raa_0", category="Viewing geometry",
            view_zenith_angle=15.0, local_view_zenith_angle=15.0,
            relative_azimuth_angle=0.0),
    replace(BASELINE, name="view_15_raa_90", category="Viewing geometry",
            view_zenith_angle=15.0, local_view_zenith_angle=15.0,
            relative_azimuth_angle=90.0),
    replace(BASELINE, name="view_15_raa_180", category="Viewing geometry",
            view_zenith_angle=15.0, local_view_zenith_angle=15.0,
            relative_azimuth_angle=180.0),
)

CATEGORIES = (
    "Snow SSA",
    "LAP",
    "Solar zenith",
    "Local incidence",
    "Atmosphere",
    "Altitude and terrain",
    "Viewing geometry",
)


def _label(lut, scenario_name, category):
    """Describe the panel parameters using values stored in the LUT."""

    parameters = {
        "Snow SSA": (
            ("specific_surface_area", "SSA", "m² kg⁻¹"),
        ),
        "LAP": (
            ("black_carbon_ng_g", "BC", "ng g⁻¹"),
            ("dust_ug_g", "Dust", "µg g⁻¹"),
        ),
        "Solar zenith": (
            ("solar_zenith_angle", "SZA", "°"),
            ("local_incidence_angle", "Incidence", "°"),
        ),
        "Local incidence": (
            ("local_incidence_angle", "Incidence", "°"),
        ),
        "Atmosphere": (
            ("aod550", "AOD550", ""),
            ("water_vapour_g_cm2", "Water", "g cm⁻²"),
            ("ozone_cm_atm", "Ozone", "cm-atm"),
        ),
        "Altitude and terrain": (
            ("altitude_m", "Altitude", "m"),
            ("sky_view_factor", "Sky view", ""),
            ("terrain_configuration_factor", "Terrain factor", ""),
            ("shadowed", "Shadow", ""),
        ),
        "Viewing geometry": (
            ("view_zenith_angle", "VZA", "°"),
            ("local_view_zenith_angle", "Local VZA", "°"),
            ("relative_azimuth_angle", "RAA", "°"),
        ),
    }

    descriptions = []
    for coordinate, label, unit in parameters[category]:
        value = lut[coordinate].sel(scenario=scenario_name).item()
        baseline_value = lut[coordinate].sel(scenario="baseline").item()
        if scenario_name != "baseline" and value == baseline_value:
            continue
        if coordinate == "shadowed":
            description = f"{label}={'yes' if value else 'no'}"
        else:
            spacing = " " if unit and unit != "°" else ""
            description = f"{label}={value:g}{spacing}{unit}"
        descriptions.append(description)

    # Wrap longer legends to keep the spectra visible.
    lines = [
        ", ".join(descriptions[index:index + 2])
        for index in range(0, len(descriptions), 2)
    ]
    prefix = "Baseline: " if scenario_name == "baseline" else ""
    return prefix + "\n".join(lines)


def plot_toa_sensitivity(lut, output_file):
    """Plot TOA-reflectance spectra grouped by physical process."""

    wavelength = lut["wavelength_nm"].values
    categories = lut["category"].values

    figure, axes = plt.subplots(
        nrows=2,
        ncols=4,
        figsize=(20, 9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for axis, category in zip(axes.flat, CATEGORIES):
        scenario_names = ["baseline"] + [
            name
            for name, scenario_category in zip(lut["scenario"].values,
                                                categories)
            if scenario_category == category
        ]
        colors = plt.cm.viridis(
            np.linspace(0.1, 0.9, len(scenario_names))
        )
        for scenario_name, color in zip(scenario_names, colors):
            spectrum = lut["toa_reflectance"].sel(scenario=scenario_name)
            linewidth = 2.4 if scenario_name == "baseline" else 1.5
            axis.plot(
                wavelength,
                spectrum,
                marker="o",
                linewidth=linewidth,
                color="black" if scenario_name == "baseline" else color,
                label=_label(lut, scenario_name, category),
            )

        axis.set_title(category)
        axis.grid(alpha=0.25)
        axis.set_ylim(0, 1.3)
        axis.legend(fontsize=8)

    for axis in axes.flat[len(CATEGORIES):]:
        axis.set_visible(False)

    for axis in axes.flat:
        if axis.get_visible():
            axis.set_xlabel("Wavelength (nm)")
    for axis in axes[:, 0]:
        axis.set_ylabel("TOA reflectance")

    figure.suptitle(
        f"REDRESS {lut.attrs['platform']} TOA snow endmember sensitivity",
        fontsize=15,
    )
    figure.savefig(output_file, dpi=250, bbox_inches="tight")
    plt.close(figure)


def incidence_ratio_diagnostics(
    lut,
    low_scenario="incidence_20",
    high_scenario="incidence_70",
):
    """Calculate spectral diagnostics for two local-incidence scenarios.

    The direct surface-radiance term in REDRESS is proportional to
    ``cos(local incidence) * BRF`` when the global solar geometry and
    atmosphere are held fixed. Comparing this proxy with the full TOA ratio
    identifies compensation by the angular BRF and by additive diffuse or
    atmospheric radiance.
    """

    low_angle = float(
        lut["local_incidence_angle"].sel(scenario=low_scenario).item()
    )
    high_angle = float(
        lut["local_incidence_angle"].sel(scenario=high_scenario).item()
    )
    low_toa = lut["toa_reflectance"].sel(scenario=low_scenario).values
    high_toa = lut["toa_reflectance"].sel(scenario=high_scenario).values
    low_brf = lut["surface_brf"].sel(scenario=low_scenario).values
    high_brf = lut["surface_brf"].sel(scenario=high_scenario).values
    low_direct_albedo = lut["direct_albedo"].sel(
        scenario=low_scenario
    ).values
    high_direct_albedo = lut["direct_albedo"].sel(
        scenario=high_scenario
    ).values
    low_hdr = lut["hemispherical_directional_reflectance"].sel(
        scenario=low_scenario
    ).values
    high_hdr = lut["hemispherical_directional_reflectance"].sel(
        scenario=high_scenario
    ).values
    low_radiance = lut["toa_radiance"].sel(
        scenario=low_scenario
    ).values
    high_radiance = lut["toa_radiance"].sel(
        scenario=high_scenario
    ).values
    low_direct_radiance = lut["toa_direct_surface_radiance"].sel(
        scenario=low_scenario
    ).values
    high_direct_radiance = lut["toa_direct_surface_radiance"].sel(
        scenario=high_scenario
    ).values

    def safe_ratio(numerator, denominator):
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=np.float64),
            where=np.abs(denominator) > np.finfo(np.float32).eps,
        )

    cosine_ratio = (
        np.cos(np.deg2rad(low_angle))
        / np.cos(np.deg2rad(high_angle))
    )
    brf_ratio = safe_ratio(low_brf, high_brf)

    diagnostics = xr.Dataset(
        data_vars={
            "toa_reflectance_low": ("band", low_toa),
            "toa_reflectance_high": ("band", high_toa),
            "toa_reflectance_ratio": (
                "band", safe_ratio(low_toa, high_toa)
            ),
            "surface_brf_low": ("band", low_brf),
            "surface_brf_high": ("band", high_brf),
            "surface_brf_ratio": ("band", brf_ratio),
            "direct_albedo_ratio": (
                "band", safe_ratio(low_direct_albedo, high_direct_albedo)
            ),
            "hemispherical_directional_reflectance_ratio": (
                "band", safe_ratio(low_hdr, high_hdr)
            ),
            "direct_surface_proxy_ratio": (
                "band", cosine_ratio * brf_ratio
            ),
            "direct_surface_radiance_ratio": (
                "band",
                safe_ratio(low_direct_radiance, high_direct_radiance),
            ),
            "direct_surface_fraction_low": (
                "band", safe_ratio(low_direct_radiance, low_radiance)
            ),
            "direct_surface_fraction_high": (
                "band", safe_ratio(high_direct_radiance, high_radiance)
            ),
        },
        coords={
            "band": lut["band"].values,
            "wavelength_nm": (
                "band", lut["wavelength_nm"].values
            ),
        },
        attrs={
            "low_scenario": low_scenario,
            "high_scenario": high_scenario,
            "low_local_incidence_degree": low_angle,
            "high_local_incidence_degree": high_angle,
            "cosine_ratio": cosine_ratio,
            "direct_surface_proxy": "cos(local incidence) * surface BRF",
        },
    )
    for variable in diagnostics.data_vars:
        diagnostics[variable].attrs["units"] = "1"
    return diagnostics


def plot_incidence_ratio(diagnostics, output_file):
    """Plot TOA and direct-surface incidence ratios by wavelength."""

    wavelength = diagnostics["wavelength_nm"].values
    toa_ratio = diagnostics["toa_reflectance_ratio"].values
    brf_ratio = diagnostics["surface_brf_ratio"].values
    direct_albedo_ratio = diagnostics["direct_albedo_ratio"].values
    hdr_ratio = diagnostics[
        "hemispherical_directional_reflectance_ratio"
    ].values
    direct_ratio = diagnostics["direct_surface_proxy_ratio"].values
    direct_radiance_ratio = diagnostics[
        "direct_surface_radiance_ratio"
    ].values
    direct_fraction_low = diagnostics[
        "direct_surface_fraction_low"
    ].values
    direct_fraction_high = diagnostics[
        "direct_surface_fraction_high"
    ].values
    cosine_ratio = diagnostics.attrs["cosine_ratio"]
    low_angle = diagnostics.attrs["low_local_incidence_degree"]
    high_angle = diagnostics.attrs["high_local_incidence_degree"]

    figure, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(10, 14),
        sharex=True,
        constrained_layout=True,
    )

    axes[0].plot(wavelength, toa_ratio, marker="o", linewidth=2)
    axes[0].axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    axes[0].set_ylabel("TOA-reflectance ratio")
    axes[0].set_title(
        f"TOA({low_angle:g}°) / TOA({high_angle:g}°)"
    )

    axes[1].plot(
        wavelength,
        brf_ratio,
        marker="o",
        label="BRF ratio",
    )
    axes[1].plot(
        wavelength,
        direct_albedo_ratio,
        marker="s",
        label="Direct-albedo ratio",
    )
    axes[1].plot(
        wavelength,
        hdr_ratio,
        marker="^",
        label="HDR ratio",
    )
    axes[1].axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Ratio")
    axes[1].set_title("Angular surface-reflectance quantities")
    axes[1].legend()

    axes[2].plot(
        wavelength,
        direct_ratio,
        marker="s",
        label="cos(i) × BRF ratio",
    )
    axes[2].plot(
        wavelength,
        direct_radiance_ratio,
        marker="^",
        linestyle="--",
        label="Direct-radiance ratio",
    )
    axes[2].axhline(
        cosine_ratio,
        color="0.4",
        linestyle=":",
        label=f"cosine ratio = {cosine_ratio:.2f}",
    )
    axes[2].axhline(1.0, color="0.4", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Ratio")
    axes[2].set_title("Direct surface-radiance decomposition")
    axes[2].legend()

    axes[3].plot(
        wavelength,
        direct_fraction_low,
        marker="o",
        label=f"Incidence {low_angle:g}°",
    )
    axes[3].plot(
        wavelength,
        direct_fraction_high,
        marker="s",
        label=f"Incidence {high_angle:g}°",
    )
    axes[3].set_xlabel("Wavelength (nm)")
    axes[3].set_ylabel("Direct / total TOA radiance")
    axes[3].set_ylim(0, 1.05)
    axes[3].set_title("Direct surface contribution to the TOA signal")
    axes[3].legend()

    for axis in axes:
        axis.grid(alpha=0.25)
        for band in ("B11", "B12"):
            if band not in diagnostics["band"].values:
                continue
            band_wavelength = diagnostics["wavelength_nm"].sel(
                band=band
            ).item()
            band_value = diagnostics["toa_reflectance_ratio"].sel(
                band=band
            ).item()
            if axis is axes[0]:
                axis.annotate(
                    band,
                    (band_wavelength, band_value),
                    xytext=(4, 6),
                    textcoords="offset points",
                )

    figure.savefig(output_file, dpi=250, bbox_inches="tight")
    plt.close(figure)


def main():
    """Generate, save and plot the diagnostic LUT."""

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    lut = create_sentinel2_toa_lut(
        platform=PLATFORM,
        band_names=BANDS,
        scenarios=SCENARIOS,
        month=MONTH,
        day=DAY,
        aerosol_model=AEROSOL_MODEL,
        atmospheric_profile=ATMOSPHERIC_PROFILE,
        case_number=REDRESS_CASE_NUMBER,
        scene_size=SCENE_SIZE,
        pixel_resolution_m=PIXEL_RESOLUTION_M,
        terrain_window_width_m=TERRAIN_WINDOW_WIDTH_M,
        atmosphere_window_width_m=ATMOSPHERE_WINDOW_WIDTH_M,
    )

    netcdf_file = OUTPUT_DIRECTORY / f"{PLATFORM}_toa_endmember_lut.nc"
    csv_file = OUTPUT_DIRECTORY / f"{PLATFORM}_toa_endmember_lut.csv"
    figure_file = OUTPUT_DIRECTORY / f"{PLATFORM}_toa_sensitivity.png"
    incidence_csv_file = (
        OUTPUT_DIRECTORY / f"{PLATFORM}_incidence_ratio.csv"
    )
    incidence_figure_file = (
        OUTPUT_DIRECTORY / f"{PLATFORM}_incidence_ratio.png"
    )

    encoding = {
        variable: {"dtype": "float32", "zlib": True, "complevel": 4}
        for variable in lut.data_vars
    }
    lut.to_netcdf(netcdf_file, encoding=encoding)
    lut.to_dataframe().reset_index().to_csv(csv_file, index=False)
    plot_toa_sensitivity(lut, figure_file)
    incidence_diagnostics = incidence_ratio_diagnostics(lut)
    incidence_diagnostics.to_dataframe().reset_index().to_csv(
        incidence_csv_file,
        index=False,
    )
    plot_incidence_ratio(incidence_diagnostics, incidence_figure_file)

    print(lut)
    print(f"NetCDF LUT: {netcdf_file.resolve()}")
    print(f"CSV LUT:    {csv_file.resolve()}")
    print(f"Figure:     {figure_file.resolve()}")
    print(f"Ratio CSV:  {incidence_csv_file.resolve()}")
    print(f"Ratio plot: {incidence_figure_file.resolve()}")


if __name__ == "__main__":
    main()
