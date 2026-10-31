#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run REDRESS over analytical DEMs with controlled surface cover.

Every terrain product is derived from the elevation raster with the standard
REDRESS production methods. The full rugged-terrain case is then evaluated
with moving-window widths specified in metres. On enclosed terrains, the
snow fraction of the wall opposite the sampled wall is varied from one to
zero while the sampled pixels remain snow covered.

Run from the repository root:

``python examples/run_idealized_terrain_experiment.py``
"""

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from redress.inputs.products import Dem, Model
from redress.topography import dem_products
from redress.topography.forward_model import Cases, iterative_radiance
from redress.topography.idealized import (
    circular_bowl,
    flat_plateau,
    narrow_couloir,
    symmetric_v_valley,
    symmetric_v_valley_wall_view_factor,
    uniform_slope,
)


# ---------------------------------------------------------------------------
# Experimental configuration
# ---------------------------------------------------------------------------

PLATFORM = "S2B"
BANDS = ("B03", "B08", "B11")

PIXEL_RESOLUTION_M = 20.0
DOMAIN_HALF_WIDTH_M = 3000.0
GRID_SIZE = int(2 * DOMAIN_HALF_WIDTH_M / PIXEL_RESOLUTION_M) + 1
HORIZON_HALF_DIRECTIONS = 32  # Produces 64 azimuth directions.

TERRAIN_WINDOW_WIDTH_M = 1500.0
ATMOSPHERE_WINDOW_WIDTH_M = 2100.0
REFERENCE_MEAN_ALTITUDE_M = 2500.0

SOLAR_ZENITH_DEG = 45.0
SOLAR_AZIMUTH_DEG = 180.0
VIEW_ZENITH_DEG = 0.0
VIEW_AZIMUTH_DEG = 0.0
SELF_SHADOW_COSINE_THRESHOLD = 0.035

SNOW_SPECIFIC_SURFACE_AREA_M2_KG = 20.0
BLACK_CARBON_NG_G = 0.0
DUST_UG_G = 0.0

MONTH = 3
DAY = 28
AEROSOL_MODEL = "Continental"
AOD550 = 0.10
WATER_VAPOUR_G_CM2 = 1.0
OZONE_CM_ATM = 0.30
ATMOSPHERIC_PROFILE = None

REDRESS_CASE_NUMBER = 5
OPPOSITE_WALL_SNOW_FRACTIONS = (1.0, 0.5, 0.0)
TERRAIN_CONFIGURATION_METHODS = (
    "hemispheric_closure",
    "dozier_frew",
)
PRIMARY_TERRAIN_CONFIGURATION_METHOD = "hemispheric_closure"

# Illustrative Lambertian snow-free alpine ground. Values are surface
# reflectance factors, not Sentinel-2 TOA reflectances.
GROUND_REFLECTANCE = {
    "B03": 0.18,
    "B08": 0.30,
    "B11": 0.25,
}

OUTPUT_DIRECTORY = Path("outputs/idealized_terrain_experiment")
TOPOGRAPHY_OUTPUT = OUTPUT_DIRECTORY / "idealized_topography.nc"
REFLECTANCE_OUTPUT = OUTPUT_DIRECTORY / "idealized_toa_reflectance.nc"
SAMPLE_OUTPUT = OUTPUT_DIRECTORY / "idealized_sample_results.nc"
SAMPLE_CSV = OUTPUT_DIRECTORY / "idealized_sample_results.csv"
DEM_FIGURE = OUTPUT_DIRECTORY / "idealized_dem_overview.png"

TOPOGRAPHIC_VARIABLES = (
    "altitude",
    "slope",
    "aspect",
    "eff_sza",
    "eff_vza",
    "vt",
    "ct",
    "ct_hemispheric_closure",
    "ct_dozier_frew",
    "beam_visibility",
    "self_shadows",
    "cast_shadows",
    "view_ground",
)


def build_terrain_suite():
    """Construct the analytical DEMs used by the experiment."""

    terrains = [
        flat_plateau(
            GRID_SIZE,
            PIXEL_RESOLUTION_M,
            altitude_m=REFERENCE_MEAN_ALTITUDE_M,
        ),
        uniform_slope(
            GRID_SIZE,
            PIXEL_RESOLUTION_M,
            slope_degree=25.0,
            aspect_degree=SOLAR_AZIMUTH_DEG,
            name="slope_sun_25",
            label="25° slope facing the Sun",
        ),
        uniform_slope(
            GRID_SIZE,
            PIXEL_RESOLUTION_M,
            slope_degree=25.0,
            aspect_degree=(SOLAR_AZIMUTH_DEG + 180.0) % 360.0,
            name="slope_away_25",
            label="25° slope facing away from the Sun",
        ),
    ]
    terrains.extend(
        symmetric_v_valley(
            GRID_SIZE,
            PIXEL_RESOLUTION_M,
            wall_slope_degree=wall_slope,
        )
        for wall_slope in (25.0, 45.0, 55.0)
    )
    terrains.extend((
        circular_bowl(
            GRID_SIZE,
            PIXEL_RESOLUTION_M,
            wall_slope_degree=40.0,
        ),
        narrow_couloir(
            GRID_SIZE,
            PIXEL_RESOLUTION_M,
            wall_slope_degree=65.0,
            axis_azimuth_degree=SOLAR_AZIMUTH_DEG,
            name="narrow_couloir_sun_aligned_65",
            label="65° couloir aligned with the Sun",
        ),
        narrow_couloir(
            GRID_SIZE,
            PIXEL_RESOLUTION_M,
            wall_slope_degree=65.0,
            axis_azimuth_degree=(SOLAR_AZIMUTH_DEG + 90.0) % 180.0,
            name="narrow_couloir_cross_sun_65",
            label="65° couloir perpendicular to the Sun",
        ),
    ))

    # Holding the domain-mean elevation fixed prevents atmospheric altitude
    # from becoming an unintended difference between terrain experiments.
    normalized = []
    for terrain in terrains:
        elevation = (
            terrain.elevation_m
            - np.mean(terrain.elevation_m)
            + REFERENCE_MEAN_ALTITUDE_M
        )
        normalized.append(replace(terrain, elevation_m=elevation))
    return tuple(normalized)


def constant_angles(terrain):
    """Return spatially constant satellite geometry on the DEM grid."""

    shape = terrain.elevation_m.shape
    coordinates = {
        "x": terrain.northing_m,
        "y": terrain.easting_m,
    }
    angle_values = {
        "SZA": SOLAR_ZENITH_DEG,
        "SAA": SOLAR_AZIMUTH_DEG,
        "VZA": VIEW_ZENITH_DEG,
        "VAA": VIEW_AZIMUTH_DEG,
    }
    return xr.Dataset({
        name: xr.DataArray(
            np.full(shape, np.deg2rad(value), dtype=np.float64),
            dims=("x", "y"),
            coords=coordinates,
            attrs={"units": "radian"},
        )
        for name, value in angle_values.items()
    })


def derive_topography(terrain):
    """Run the standard REDRESS DEM processing chain."""

    dem = Dem()
    dem.pixel_size = (PIXEL_RESOLUTION_M, PIXEL_RESOLUTION_M)
    dem.bands["altitude"] = xr.DataArray(
        terrain.elevation_m,
        dims=("x", "y"),
        coords={
            "x": terrain.northing_m,
            "y": terrain.easting_m,
        },
        attrs={"units": "m"},
    )
    dem.angles = constant_angles(terrain)

    print(f"Computing horizons: {terrain.label}")
    dem.compute_horizon(N=HORIZON_HALF_DIRECTIONS)
    dem.compute_topo_bands(
        self_shadow_cosine_threshold=SELF_SHADOW_COSINE_THRESHOLD,
        clean_cast_shadows=False,
        terrain_configuration_method=(
            PRIMARY_TERRAIN_CONFIGURATION_METHOD
        ),
    )
    for method in TERRAIN_CONFIGURATION_METHODS:
        _, terrain_factor = dem_products.skyview(
            dem.bands["horizon_ele"].data,
            dem.bands["slope"].data,
            dem.bands["aspect"].data,
            terrain_configuration_method=method,
        )
        variable = f"ct_{method}"
        dem.bands[variable] = xr.DataArray(
            terrain_factor,
            dims=("x", "y"),
            coords={
                "x": terrain.northing_m,
                "y": terrain.easting_m,
            },
            attrs={"units": "1", "method": method},
        )
    return dem


def sample_index(terrain, sample):
    """Return the raster indices nearest a metric sample coordinate."""

    row = int(np.argmin(np.abs(terrain.northing_m - sample.northing_m)))
    column = int(np.argmin(np.abs(terrain.easting_m - sample.easting_m)))
    return row, column


def validate_sample_buffers(terrains):
    """Ensure moving windows never touch a domain boundary at samples."""

    required_buffer = max(
        TERRAIN_WINDOW_WIDTH_M,
        ATMOSPHERE_WINDOW_WIDTH_M,
    ) / 2.0
    domain_limit = DOMAIN_HALF_WIDTH_M - required_buffer
    for terrain in terrains:
        for sample in terrain.samples:
            if (
                abs(sample.northing_m) > domain_limit
                or abs(sample.easting_m) > domain_limit
            ):
                raise ValueError(
                    f"Sample {terrain.name}/{sample.name} is too close to "
                    "the domain edge for the configured neighbourhoods"
                )


def validate_terrain_sequence(terrain_products):
    """Check the expected flat, slope and self-shadow limiting cases."""

    def value(terrain_name, variable, sample_name):
        terrain, dem = terrain_products[terrain_name]
        sample = next(
            item for item in terrain.samples if item.name == sample_name
        )
        row, column = sample_index(terrain, sample)
        return float(dem.bands[variable].values[row, column])

    if not np.isclose(
        value("flat_plateau", "vt", "centre"), 1.0, atol=0.02
    ):
        raise AssertionError("Flat plateau does not have an open sky view")
    if not np.isclose(
        np.rad2deg(value("slope_sun_25", "eff_sza", "mid_slope")),
        20.0,
        atol=0.5,
    ):
        raise AssertionError("Sun-facing slope has an unexpected incidence")
    if not np.isclose(
        np.rad2deg(value("slope_away_25", "eff_sza", "mid_slope")),
        70.0,
        atol=0.5,
    ):
        raise AssertionError("Away-facing slope has an unexpected incidence")
    if value(
        "v_valley_25", "self_shadows", "away_wall_midpoint"
    ) != 1.0:
        raise AssertionError("The 25° away-facing valley wall should be lit")
    if value(
        "v_valley_45", "self_shadows", "away_wall_midpoint"
    ) != 0.0:
        raise AssertionError(
            "The 45° away-facing valley wall should be self-shadowed at "
            "tangent solar incidence"
        )
    if value(
        "v_valley_55", "self_shadows", "away_wall_midpoint"
    ) != 0.0:
        raise AssertionError(
            "The 55° away-facing valley wall should be self-shadowed"
        )
    if value(
        "narrow_couloir_sun_aligned_65",
        "beam_visibility",
        "couloir_floor",
    ) != 1.0:
        raise AssertionError("The sun-aligned couloir floor should be lit")
    if value(
        "narrow_couloir_cross_sun_65",
        "beam_visibility",
        "couloir_floor",
    ) != 0.0:
        raise AssertionError(
            "The cross-sun couloir floor should be cast-shadowed"
        )


def v_valley_away_wall_mask(terrain):
    """Return the wall opposite the designated sun-facing source wall."""

    if not terrain.name.startswith("v_valley_"):
        return np.zeros(terrain.elevation_m.shape, dtype=bool)
    north = np.broadcast_to(
        terrain.northing_m[:, np.newaxis],
        terrain.elevation_m.shape,
    )
    source_north = north[terrain.opposite_wall_mask]
    wall_extent = float(np.max(source_north))
    return (north < 0.0) & (north >= -wall_extent)


def report_v_valley_view_factor_check(terrain_products):
    """Compare raster Ct with the exact two-wall crossed-string solution."""

    print("\nV-valley wall-to-wall view-factor check")
    print("beta  raster mean Ct  exact 1-cos(beta)  difference")
    for wall_slope in (25.0, 45.0, 55.0):
        terrain_name = f"v_valley_{wall_slope:g}"
        terrain, dem = terrain_products[terrain_name]
        away_wall = v_valley_away_wall_mask(terrain)
        valid = away_wall & np.isfinite(
            dem.bands["ct_hemispheric_closure"].values
        )
        raster_mean = float(np.mean(
            dem.bands["ct_hemispheric_closure"].values[valid]
        ))
        exact = symmetric_v_valley_wall_view_factor(wall_slope)
        print(
            f"{wall_slope:4.0f}°  {raster_mean:14.4f}  "
            f"{exact:17.4f}  {raster_mean - exact:+.4f}"
        )


def configured_model(dem):
    """Prepare snow optics and atmospheric inputs on one derived DEM."""

    model = Model()
    model.set_sentinel2(PLATFORM, BANDS)
    model.import_rt_model("python_6S")
    model.ssa = SNOW_SPECIFIC_SURFACE_AREA_M2_KG
    model.set_lap(
        black_carbon_ng_g=BLACK_CARBON_NG_G,
        dust_ug_g=DUST_UG_G,
    )
    model.angles = dem.angles
    model.topo_bands = dem.bands
    model.set_rt_options(
        aero=AEROSOL_MODEL,
        aod=AOD550,
        water=WATER_VAPOUR_G_CM2,
        ozone=OZONE_CM_ATM,
        atmos=ATMOSPHERIC_PROFILE,
        atcor=False,
        month=MONTH,
        day=DAY,
    )
    model.compute_albedo()
    model.compute_brf()
    return model


def surface_snow_fraction(terrain, opposite_wall_snow_fraction):
    """Keep targets snowy while varying the designated opposite wall."""

    fraction = np.ones(terrain.elevation_m.shape, dtype=np.float64)
    fraction[terrain.opposite_wall_mask] = opposite_wall_snow_fraction
    return fraction


def mixed_surface_optics(snow_value, snow_fraction, ground_reflectance):
    """Linearly mix snowy and Lambertian snow-free subpixel fractions."""

    return (
        snow_fraction * np.asarray(snow_value, dtype=np.float64)
        + (1.0 - snow_fraction) * ground_reflectance
    )


def simulate_cover(
    model,
    terrain,
    opposite_wall_snow_fraction,
    terrain_configuration_method,
):
    """Run full REDRESS for one opposite-wall material configuration."""

    case = Cases()
    case.create_cases(REDRESS_CASE_NUMBER)
    snow_fraction = surface_snow_fraction(
        terrain, opposite_wall_snow_fraction
    )
    topography = model.topo_bands.copy(deep=False)
    topography["ct"] = model.topo_bands[
        f"ct_{terrain_configuration_method}"
    ]
    band_definitions = model.meta["band_definitions"]
    reflectance = []
    components = {
        name: []
        for name in (
            "direct_surface",
            "diffuse_surface",
            "neighbour_atmosphere",
            "atmospheric_path",
        )
    }

    for index, (band, wavelength, definition) in enumerate(zip(
        model.meta["bandlist"],
        model.meta["wavelengths"],
        band_definitions,
    )):
        ground = GROUND_REFLECTANCE[band]
        hdr = mixed_surface_optics(
            model.albedo["hdr"][index], snow_fraction, ground
        )
        bhr = mixed_surface_optics(
            model.albedo["diffuse"][index], snow_fraction, ground
        )
        brf = mixed_surface_optics(
            model.brf[index][1], snow_fraction, ground
        )

        result = iterative_radiance(
            topo_bands=topography,
            angles=model.angles,
            wavelength=wavelength,
            hdr=hdr,
            bhr=bhr,
            rt_model=model.rt_model(),
            rt_options=model.rt_options,
            brf=brf,
            case=case,
            dif_anis=False,
            rt_wavelength=definition.py6s_filter,
            return_reflectance=True,
            return_components=True,
            verbose=False,
            pixel_resolution_m=PIXEL_RESOLUTION_M,
            terrain_window_width_m=TERRAIN_WINDOW_WIDTH_M,
            atmosphere_window_width_m=ATMOSPHERE_WINDOW_WIDTH_M,
        )
        _, band_reflectance, band_components = result
        reflectance.append(band_reflectance)
        for name in components:
            components[name].append(band_components[name])

    return (
        np.stack(reflectance, axis=0),
        {
            name: np.stack(values, axis=0)
            for name, values in components.items()
        },
    )


def topography_for_output(terrain, dem):
    """Select derived raster terms and attach the terrain dimension."""

    dataset = dem.bands[list(TOPOGRAPHIC_VARIABLES)].copy()
    dataset["opposite_wall_mask"] = xr.DataArray(
        terrain.opposite_wall_mask.astype(np.int8),
        dims=("x", "y"),
        coords={"x": terrain.northing_m, "y": terrain.easting_m},
    )
    dataset = dataset.expand_dims(terrain=[terrain.name])
    dataset = dataset.assign_coords(
        terrain_label=("terrain", [terrain.label])
    )
    return dataset


def raster_output_dataset(
    terrain,
    model,
    opposite_wall_snow_fraction,
    terrain_configuration_method,
    reflectance,
):
    """Package full TOA-reflectance rasters for one experiment."""

    experiment = (
        f"{terrain.name}_wall_snow_"
        f"{opposite_wall_snow_fraction:.2f}_"
        f"{terrain_configuration_method}"
    )
    dataset = xr.Dataset(
        {
            "toa_reflectance": (
                ("band", "x", "y"),
                reflectance.astype(np.float32),
            )
        },
        coords={
            "band": model.meta["bandlist"],
            "wavelength_nm": (
                "band", model.meta["wavelengths"]
            ),
            "x": terrain.northing_m,
            "y": terrain.easting_m,
        },
    )
    dataset = dataset.expand_dims(experiment=[experiment])
    return dataset.assign_coords(
        terrain_name=("experiment", [terrain.name]),
        opposite_wall_snow_fraction=(
            "experiment", [opposite_wall_snow_fraction]
        ),
        terrain_configuration_method=(
            "experiment", [terrain_configuration_method]
        ),
    )


def sample_records(
    terrain,
    dem,
    model,
    opposite_wall_snow_fraction,
    terrain_configuration_method,
    reflectance,
    components,
):
    """Extract radiometric and terrain diagnostics at named samples."""

    records = []
    for sample in terrain.samples:
        row, column = sample_index(terrain, sample)
        for band_index, band in enumerate(model.meta["bandlist"]):
            record = {
                "terrain": terrain.name,
                "terrain_label": terrain.label,
                "sample": sample.name,
                "northing_m": terrain.northing_m[row],
                "easting_m": terrain.easting_m[column],
                "opposite_wall_snow_fraction": (
                    opposite_wall_snow_fraction
                ),
                "terrain_configuration_method": (
                    terrain_configuration_method
                ),
                "self_shadow_cosine_threshold": (
                    SELF_SHADOW_COSINE_THRESHOLD
                ),
                "band": band,
                "wavelength_nm": model.meta["wavelengths"][band_index],
                "toa_reflectance": reflectance[
                    band_index, row, column
                ],
            }
            for name in TOPOGRAPHIC_VARIABLES:
                source_name = (
                    f"ct_{terrain_configuration_method}"
                    if name == "ct"
                    else name
                )
                record[name] = dem.bands[source_name].values[row, column]
            record["analytical_wall_view_factor"] = (
                symmetric_v_valley_wall_view_factor(
                    float(terrain.name.removeprefix("v_valley_"))
                )
                if terrain.name.startswith("v_valley_")
                else np.nan
            )
            away_wall = v_valley_away_wall_mask(terrain)
            record["raster_wall_mean_ct"] = (
                float(np.nanmean(
                    dem.bands[
                        f"ct_{terrain_configuration_method}"
                    ].values[away_wall]
                ))
                if np.any(away_wall)
                else np.nan
            )
            for name, values in components.items():
                record[f"toa_{name}_radiance"] = values[
                    band_index, row, column
                ]
            records.append(record)
    return records


def records_to_dataset(records):
    """Convert flat sample records to a self-describing xarray dataset."""

    fields = records[0].keys()
    data_variables = {
        field: ("observation", [record[field] for record in records])
        for field in fields
    }
    return xr.Dataset(data_variables)


def save_csv(records, output_file):
    """Write sample diagnostics in a portable tabular format."""

    with output_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def plot_dem_overview(terrains):
    """Plot every analytical elevation raster and its sample locations."""

    columns = 3
    rows = int(np.ceil(len(terrains) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, 4.2 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, terrain in zip(axes.ravel(), terrains):
        image = axis.imshow(
            terrain.elevation_m,
            extent=(
                terrain.easting_m.min(),
                terrain.easting_m.max(),
                terrain.northing_m.min(),
                terrain.northing_m.max(),
            ),
            origin="upper",
            cmap="terrain",
        )
        axis.contour(
            terrain.easting_m,
            terrain.northing_m,
            terrain.opposite_wall_mask.astype(float),
            levels=[0.5],
            colors=["cyan"],
            linewidths=1.0,
        )
        for sample in terrain.samples:
            axis.plot(
                sample.easting_m,
                sample.northing_m,
                marker="o",
                color="black",
                markersize=4,
            )
        axis.set_title(terrain.label, fontsize=9)
        axis.set_xlabel("Easting (m)")
        axis.set_ylabel("Northing (m)")
        figure.colorbar(image, ax=axis, shrink=0.72, label="Elevation (m)")
    for axis in axes.ravel()[len(terrains):]:
        axis.set_visible(False)
    return figure


def rename_spatial_dimensions(dataset):
    """Replace REDRESS legacy x/y dimension names in exported datasets."""

    renamed = dataset.rename({"x": "northing", "y": "easting"})
    renamed["northing"].attrs.update(units="m", axis="Y")
    renamed["easting"].attrs.update(units="m", axis="X")
    return renamed


def main():
    """Derive terrain terms, run REDRESS and save raster/sample outputs."""

    if GRID_SIZE % 2 == 0:
        raise ValueError("GRID_SIZE must be odd")
    if (
        PRIMARY_TERRAIN_CONFIGURATION_METHOD
        not in TERRAIN_CONFIGURATION_METHODS
    ):
        raise ValueError(
            "PRIMARY_TERRAIN_CONFIGURATION_METHOD must be included in "
            "TERRAIN_CONFIGURATION_METHODS"
        )
    missing_ground_bands = set(BANDS).difference(GROUND_REFLECTANCE)
    if missing_ground_bands:
        raise ValueError(
            "Missing ground reflectance for: "
            + ", ".join(sorted(missing_ground_bands))
        )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    terrains = build_terrain_suite()
    validate_sample_buffers(terrains)

    terrain_products = {}
    topography_outputs = []
    for terrain in terrains:
        dem = derive_topography(terrain)
        terrain_products[terrain.name] = (terrain, dem)
        topography_outputs.append(topography_for_output(terrain, dem))

    validate_terrain_sequence(terrain_products)
    report_v_valley_view_factor_check(terrain_products)

    raster_outputs = []
    records = []
    for terrain in terrains:
        _, dem = terrain_products[terrain.name]
        model = configured_model(dem)
        cover_fractions = (
            OPPOSITE_WALL_SNOW_FRACTIONS
            if np.any(terrain.opposite_wall_mask)
            else (1.0,)
        )
        for snow_fraction in cover_fractions:
            for method in TERRAIN_CONFIGURATION_METHODS:
                print(
                    f"Running {terrain.label}; opposite-wall snow "
                    f"fraction={snow_fraction:.2f}; Ct={method}"
                )
                reflectance, components = simulate_cover(
                    model,
                    terrain,
                    snow_fraction,
                    method,
                )
                raster_outputs.append(raster_output_dataset(
                    terrain,
                    model,
                    snow_fraction,
                    method,
                    reflectance,
                ))
                records.extend(sample_records(
                    terrain,
                    dem,
                    model,
                    snow_fraction,
                    method,
                    reflectance,
                    components,
                ))

    topography = xr.concat(
        topography_outputs,
        dim="terrain",
        join="exact",
        compat="no_conflicts",
    )
    topography.attrs.update(
        title="REDRESS analytical-terrain products",
        pixel_resolution_m=PIXEL_RESOLUTION_M,
        horizon_directions=2 * HORIZON_HALF_DIRECTIONS,
        terrain_window_width_m=TERRAIN_WINDOW_WIDTH_M,
        atmosphere_window_width_m=ATMOSPHERE_WINDOW_WIDTH_M,
        self_shadow_cosine_threshold=SELF_SHADOW_COSINE_THRESHOLD,
        primary_terrain_configuration_method=(
            PRIMARY_TERRAIN_CONFIGURATION_METHOD
        ),
        terrain_configuration_sensitivity_methods=(
            ", ".join(TERRAIN_CONFIGURATION_METHODS)
        ),
    )
    rename_spatial_dimensions(topography).to_netcdf(TOPOGRAPHY_OUTPUT)

    reflectance_maps = xr.concat(
        raster_outputs,
        dim="experiment",
        join="exact",
        compat="no_conflicts",
    )
    reflectance_maps.attrs.update(
        title="REDRESS TOA reflectance over analytical terrains",
        redress_case_number=REDRESS_CASE_NUMBER,
        ground_model="Lambertian linear snow-ground areal mixture",
        primary_terrain_configuration_method=(
            PRIMARY_TERRAIN_CONFIGURATION_METHOD
        ),
    )
    rename_spatial_dimensions(reflectance_maps).to_netcdf(
        REFLECTANCE_OUTPUT
    )

    sample_dataset = records_to_dataset(records)
    sample_dataset.attrs["title"] = (
        "REDRESS analytical-terrain sample diagnostics"
    )
    sample_dataset.to_netcdf(SAMPLE_OUTPUT)
    save_csv(records, SAMPLE_CSV)

    figure = plot_dem_overview(terrains)
    figure.savefig(DEM_FIGURE, dpi=250, bbox_inches="tight")
    plt.close(figure)

    print(f"Topographic rasters: {TOPOGRAPHY_OUTPUT}")
    print(f"TOA-reflectance rasters: {REFLECTANCE_OUTPUT}")
    print(f"Sample diagnostics: {SAMPLE_OUTPUT}")
    print(f"Sample CSV: {SAMPLE_CSV}")
    print(f"DEM overview: {DEM_FIGURE}")


if __name__ == "__main__":
    main()
