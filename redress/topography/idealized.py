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
    floor_half_width_m=100.0,
    wall_width_m=600.0,
    floor_altitude_m=1500.0,
):
    """Build a north-south couloir enclosed by steep east and west walls."""

    north_axis, east_axis, north, east = metric_grid(
        grid_size, pixel_resolution_m
    )
    wall_distance = np.clip(
        np.abs(east) - floor_half_width_m,
        0.0,
        wall_width_m,
    )
    elevation = (
        floor_altitude_m
        + np.tan(np.deg2rad(wall_slope_degree)) * wall_distance
    )
    opposite_wall = (
        (east > floor_half_width_m)
        & (east <= floor_half_width_m + wall_width_m)
    )
    midpoint = floor_half_width_m + wall_width_m / 2.0
    return IdealizedTerrain(
        name=f"narrow_couloir_{wall_slope_degree:g}",
        label=f"Narrow couloir, wall slope {wall_slope_degree:g}°",
        elevation_m=elevation,
        northing_m=north_axis,
        easting_m=east_axis,
        opposite_wall_mask=opposite_wall,
        samples=(
            TerrainSample("west_wall_midpoint", 0.0, -midpoint),
            TerrainSample("couloir_floor", 0.0, 0.0),
        ),
    )
