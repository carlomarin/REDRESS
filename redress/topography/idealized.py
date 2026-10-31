#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analytical DEMs for controlled REDRESS terrain experiments."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TerrainSample:
    """Named location expressed in the local metric coordinate system."""

    name: str
    northing_m: float
    easting_m: float


@dataclass(frozen=True)
class IdealizedTerrain:
    """One analytical elevation raster and its experimental metadata."""

    name: str
    label: str
    elevation_m: np.ndarray
    northing_m: np.ndarray
    easting_m: np.ndarray
    opposite_wall_mask: np.ndarray
    samples: tuple[TerrainSample, ...]


def symmetric_v_valley_wall_view_factor(wall_slope_degree):
    """Return the exact wall-to-wall view factor of an infinite 2-D V.

    The crossed-string construction for two equal walls of length L gives
    F12 = (2 L - 2 L cos(beta)) / (2 L) = 1 - cos(beta). This is the
    wall-area-mean Lambertian view factor and provides an independent check
    on the raster-derived hemispheric terrain factor.
    """

    if not 0.0 <= wall_slope_degree < 90.0:
        raise ValueError("wall_slope_degree must be in [0, 90) degrees")
    return 1.0 - np.cos(np.deg2rad(wall_slope_degree))


def symmetric_v_valley_local_wall_view_factor(wall_slope_degree):
    """Return the local wall-to-wall factor for a semi-infinite 2-D V.

    For wall A of length ``a`` and opposite wall B of length ``b``, the
    crossed-string result is

    ``a F_A,B = (a + b - d) / 2``,

    where ``d² = a² + b² + 2ab cos(2 beta)``. Differentiating with respect
    to ``a`` gives the factor for an infinitesimal strip at the end of A.
    Taking ``b`` to infinity gives

    ``F_dA,B = (1 - cos(2 beta)) / 2 = sin²(beta)``.

    This local factor must not be confused with the equal-wall area mean
    returned by :func:`symmetric_v_valley_wall_view_factor`.
    """

    if not 0.0 <= wall_slope_degree < 90.0:
        raise ValueError("wall_slope_degree must be in [0, 90) degrees")
    return np.sin(np.deg2rad(wall_slope_degree)) ** 2


def solve_two_wall_irradiance(
    irradiance_a_0,
    irradiance_b_0,
    view_factor,
    reflectance_a,
    reflectance_b,
):
    """Solve diffuse inter-reflection between two Lambertian walls.

    ``irradiance_a_0`` and ``irradiance_b_0`` contain each wall's own direct
    and sky-diffuse illumination before terrain exchange. The returned
    terrain terms include every order of reflection between the walls.
    """

    values = {
        "irradiance_a_0": irradiance_a_0,
        "irradiance_b_0": irradiance_b_0,
        "view_factor": view_factor,
        "reflectance_a": reflectance_a,
        "reflectance_b": reflectance_b,
    }
    values = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in values.items()
    }
    if np.any((values["view_factor"] < 0.0) | (
        values["view_factor"] > 1.0
    )):
        raise ValueError("view_factor must lie in [0, 1]")
    for name in ("reflectance_a", "reflectance_b"):
        if np.any((values[name] < 0.0) | (values[name] > 1.0)):
            raise ValueError(f"{name} must lie in [0, 1]")

    denominator = 1.0 - (
        values["view_factor"] ** 2
        * values["reflectance_a"]
        * values["reflectance_b"]
    )
    if np.any(denominator <= 0.0):
        raise ValueError("two-wall reflection series does not converge")

    irradiance_a = (
        values["irradiance_a_0"]
        + values["view_factor"]
        * values["reflectance_b"]
        * values["irradiance_b_0"]
    ) / denominator
    irradiance_b = (
        values["irradiance_b_0"]
        + values["view_factor"]
        * values["reflectance_a"]
        * values["irradiance_a_0"]
    ) / denominator
    return {
        "irradiance_a": irradiance_a,
        "irradiance_b": irradiance_b,
        "terrain_irradiance_a": (
            values["view_factor"]
            * values["reflectance_b"]
            * irradiance_b
        ),
        "terrain_irradiance_b": (
            values["view_factor"]
            * values["reflectance_a"]
            * irradiance_a
        ),
    }


def metric_grid(grid_size, pixel_resolution_m):
    """Return northing/easting axes and grids centred on the middle cell."""

    if not isinstance(grid_size, int) or grid_size < 5 or grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd integer of at least 5")
    if not np.isfinite(pixel_resolution_m) or pixel_resolution_m <= 0:
        raise ValueError("pixel_resolution_m must be positive and finite")

    offsets = (
        np.arange(grid_size, dtype=np.float64) - grid_size // 2
    ) * pixel_resolution_m
    easting_m = offsets
    northing_m = -offsets
    northing, easting = np.meshgrid(
        northing_m,
        easting_m,
        indexing="ij",
    )
    return northing_m, easting_m, northing, easting


def flat_plateau(
    grid_size,
    pixel_resolution_m,
    altitude_m=2000.0,
):
    """Build a horizontal reference surface."""

    north_axis, east_axis, north, _ = metric_grid(
        grid_size, pixel_resolution_m
    )
    elevation = np.full(north.shape, altitude_m, dtype=np.float64)
    return IdealizedTerrain(
        name="flat_plateau",
        label="Flat plateau",
        elevation_m=elevation,
        northing_m=north_axis,
        easting_m=east_axis,
        opposite_wall_mask=np.zeros(north.shape, dtype=bool),
        samples=(TerrainSample("centre", 0.0, 0.0),),
    )


def uniform_slope(
    grid_size,
    pixel_resolution_m,
    slope_degree,
    aspect_degree,
    altitude_at_centre_m=2500.0,
    name=None,
    label=None,
):
    """Build a plane whose downslope compass direction is ``aspect_degree``."""

    north_axis, east_axis, north, east = metric_grid(
        grid_size, pixel_resolution_m
    )
    aspect = np.deg2rad(aspect_degree)
    downslope_distance = (
        east * np.sin(aspect) + north * np.cos(aspect)
    )
    elevation = (
        altitude_at_centre_m
        - np.tan(np.deg2rad(slope_degree)) * downslope_distance
    )
    suffix = f"{slope_degree:g}_{aspect_degree:g}"
    return IdealizedTerrain(
        name=name or f"uniform_slope_{suffix}",
        label=label or (
            f"Uniform {slope_degree:g}° slope, aspect "
            f"{aspect_degree:g}°"
        ),
        elevation_m=elevation,
        northing_m=north_axis,
        easting_m=east_axis,
        opposite_wall_mask=np.zeros(north.shape, dtype=bool),
        samples=(TerrainSample("mid_slope", 0.0, 0.0),),
    )


def symmetric_v_valley(
    grid_size,
    pixel_resolution_m,
    wall_slope_degree,
    wall_extent_m=1000.0,
    floor_altitude_m=1500.0,
):
    """Build an east-west V-valley with north and south planar walls.

    For a southerly solar azimuth, the northern wall faces the Sun and is the
    variable opposite wall. The sample wall lies to the south and faces away.
    """

    north_axis, east_axis, north, _ = metric_grid(
        grid_size, pixel_resolution_m
    )
    wall_distance = np.minimum(np.abs(north), wall_extent_m)
    elevation = (
        floor_altitude_m
        + np.tan(np.deg2rad(wall_slope_degree)) * wall_distance
    )
    opposite_wall = (north > 0.0) & (north <= wall_extent_m)
    midpoint = wall_extent_m / 2.0
    return IdealizedTerrain(
        name=f"v_valley_{wall_slope_degree:g}",
        label=f"Symmetric V-valley, β={wall_slope_degree:g}°",
        elevation_m=elevation,
        northing_m=north_axis,
        easting_m=east_axis,
        opposite_wall_mask=opposite_wall,
        samples=(
            TerrainSample("away_wall_midpoint", -midpoint, 0.0),
            TerrainSample("valley_floor", 0.0, 0.0),
        ),
    )


def circular_bowl(
    grid_size,
    pixel_resolution_m,
    wall_slope_degree=40.0,
    floor_radius_m=200.0,
    wall_width_m=800.0,
    floor_altitude_m=1500.0,
):
    """Build a circular flat-floored bowl surrounded in all directions."""

    north_axis, east_axis, north, east = metric_grid(
        grid_size, pixel_resolution_m
    )
    radius = np.hypot(north, east)
    wall_distance = np.clip(
        radius - floor_radius_m,
        0.0,
        wall_width_m,
    )
    elevation = (
        floor_altitude_m
        + np.tan(np.deg2rad(wall_slope_degree)) * wall_distance
    )
    opposite_wall = (
        (north > 0.0)
        & (radius > floor_radius_m)
        & (radius <= floor_radius_m + wall_width_m)
    )
    midpoint = floor_radius_m + wall_width_m / 2.0
    return IdealizedTerrain(
        name=f"circular_bowl_{wall_slope_degree:g}",
        label=f"Circular bowl, wall slope {wall_slope_degree:g}°",
        elevation_m=elevation,
        northing_m=north_axis,
        easting_m=east_axis,
        opposite_wall_mask=opposite_wall,
        samples=(
            TerrainSample("away_wall_midpoint", -midpoint, 0.0),
            TerrainSample("bowl_floor", 0.0, 0.0),
        ),
    )


def narrow_couloir(
    grid_size,
    pixel_resolution_m,
    wall_slope_degree=65.0,
    axis_azimuth_degree=0.0,
    floor_half_width_m=100.0,
    wall_width_m=600.0,
    floor_altitude_m=1500.0,
    name=None,
    label=None,
):
    """Build a narrow couloir along a specified compass azimuth."""

    north_axis, east_axis, north, east = metric_grid(
        grid_size, pixel_resolution_m
    )
    axis_azimuth = np.deg2rad(axis_azimuth_degree)
    cross_east = np.cos(axis_azimuth)
    cross_north = -np.sin(axis_azimuth)
    if abs(cross_east) < 1e-12:
        cross_east = 0.0
    if abs(cross_north) < 1e-12:
        cross_north = 0.0
    cross_axis_distance = (
        east * cross_east
        + north * cross_north
    )
    wall_distance = np.clip(
        np.abs(cross_axis_distance) - floor_half_width_m,
        0.0,
        wall_width_m,
    )
    elevation = (
        floor_altitude_m
        + np.tan(np.deg2rad(wall_slope_degree)) * wall_distance
    )
    opposite_wall = (
        (cross_axis_distance > floor_half_width_m)
        & (
            cross_axis_distance
            <= floor_half_width_m + wall_width_m
        )
    )
    midpoint = floor_half_width_m + wall_width_m / 2.0
    sample_easting = -midpoint * cross_east
    sample_northing = -midpoint * cross_north
    suffix = f"{wall_slope_degree:g}_{axis_azimuth_degree:g}"
    return IdealizedTerrain(
        name=name or f"narrow_couloir_{suffix}",
        label=label or (
            f"Narrow couloir, wall slope {wall_slope_degree:g}°, "
            f"axis {axis_azimuth_degree:g}°"
        ),
        elevation_m=elevation,
        northing_m=north_axis,
        easting_m=east_axis,
        opposite_wall_mask=opposite_wall,
        samples=(
            TerrainSample(
                "cross_wall_midpoint",
                sample_northing,
                sample_easting,
            ),
            TerrainSample("couloir_floor", 0.0, 0.0),
        ),
    )
