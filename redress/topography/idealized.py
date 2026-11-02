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


def symmetric_v_valley_finite_wall_view_factor(
    wall_slope_degree,
    sample_distance_m,
    wall_extent_m,
    valley_half_length_m,
):
    """Return the exact view factor from a wall point to the finite opposite wall.

    The sample point lies on wall A at horizontal distance
    ``sample_distance_m`` from the valley axis. Wall B is the planar
    rectangle that rises from the axis to ``wall_extent_m`` across the
    valley and spans ``±valley_half_length_m`` along it. Wall B is fully
    visible from wall A, and the flat rims are seen edge-on from below, so
    the factor follows from the contour-integral formula for a differential
    element and a planar polygon:

    ``F = (1 / 2π) Σ γ_i n·(R_i × R_i+1) / |R_i × R_i+1|``,

    where ``R_i`` are the vectors to the polygon vertices and ``γ_i`` the
    angles they subtend. As both wall dimensions tend to infinity, the
    result tends to sin²β; see
    :func:`symmetric_v_valley_local_wall_view_factor`.
    """

    if not 0.0 < wall_slope_degree < 90.0:
        raise ValueError("wall_slope_degree must be in (0, 90) degrees")
    if not 0.0 < sample_distance_m <= wall_extent_m:
        raise ValueError("sample_distance_m must lie in (0, wall_extent_m]")
    if not valley_half_length_m > 0.0:
        raise ValueError("valley_half_length_m must be positive")

    slope = np.deg2rad(wall_slope_degree)
    tangent = np.tan(slope)
    # East, north and up. Wall A lies to the south and faces north.
    point = np.array([0.0, -sample_distance_m, sample_distance_m * tangent])
    normal = np.array([0.0, np.sin(slope), np.cos(slope)])
    rim = wall_extent_m * tangent
    vertices = np.array([
        [-valley_half_length_m, 0.0, 0.0],
        [valley_half_length_m, 0.0, 0.0],
        [valley_half_length_m, wall_extent_m, rim],
        [-valley_half_length_m, wall_extent_m, rim],
    ])
    rays = vertices - point
    total = 0.0
    for start, end in zip(rays, np.roll(rays, -1, axis=0)):
        cross = np.cross(start, end)
        cosine = start @ end / (np.linalg.norm(start) * np.linalg.norm(end))
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        total += angle * (normal @ cross) / np.linalg.norm(cross)
    return abs(total) / (2.0 * np.pi)


def solve_two_wall_irradiance(
    irradiance_a_0,
    irradiance_b_0,
    view_factor,
    reflectance_a,
    reflectance_b,
    direct_irradiance_a_0=0.0,
    direct_irradiance_b_0=0.0,
    direct_reflectance_a=None,
    direct_reflectance_b=None,
):
    """Solve diffuse inter-reflection between two Lambertian walls.

    ``irradiance_a_0`` and ``irradiance_b_0`` contain each wall's own direct
    and sky-diffuse illumination before terrain exchange. The returned
    terrain terms include every order of reflection between the walls.

    Each wall first reflects its own illumination with ``reflectance`` or,
    for the ``direct_irradiance`` part of it, with ``direct_reflectance``
    (its directional-hemispherical albedo at the local incidence). Light
    exchanged between the walls is diffuse and is reflected with
    ``reflectance``. Without direct terms both walls are fully Lambertian.
    """

    if direct_reflectance_a is None:
        direct_reflectance_a = reflectance_a
    if direct_reflectance_b is None:
        direct_reflectance_b = reflectance_b
    values = {
        "irradiance_a_0": irradiance_a_0,
        "irradiance_b_0": irradiance_b_0,
        "view_factor": view_factor,
        "reflectance_a": reflectance_a,
        "reflectance_b": reflectance_b,
        "direct_irradiance_a_0": direct_irradiance_a_0,
        "direct_irradiance_b_0": direct_irradiance_b_0,
        "direct_reflectance_a": direct_reflectance_a,
        "direct_reflectance_b": direct_reflectance_b,
    }
    values = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in values.items()
    }
    if np.any((values["view_factor"] < 0.0) | (
        values["view_factor"] > 1.0
    )):
        raise ValueError("view_factor must lie in [0, 1]")
    for name in (
        "reflectance_a", "reflectance_b",
        "direct_reflectance_a", "direct_reflectance_b",
    ):
        if np.any((values[name] < 0.0) | (values[name] > 1.0)):
            raise ValueError(f"{name} must lie in [0, 1]")
    for side in ("a", "b"):
        if np.any(
            (values[f"direct_irradiance_{side}_0"] < 0.0)
            | (
                values[f"direct_irradiance_{side}_0"]
                > values[f"irradiance_{side}_0"]
            )
        ):
            raise ValueError(
                f"direct_irradiance_{side}_0 must lie in "
                f"[0, irradiance_{side}_0]"
            )

    view_factor = values["view_factor"]
    denominator = 1.0 - (
        view_factor ** 2 * values["reflectance_a"] * values["reflectance_b"]
    )
    if np.any(denominator <= 0.0):
        raise ValueError("two-wall reflection series does not converge")

    # Exitance of each wall from its own illumination.
    exitance_0 = {
        side: (
            values[f"direct_reflectance_{side}"]
            * values[f"direct_irradiance_{side}_0"]
            + values[f"reflectance_{side}"]
            * (
                values[f"irradiance_{side}_0"]
                - values[f"direct_irradiance_{side}_0"]
            )
        )
        for side in ("a", "b")
    }
    # M_a = M_a0 + rho_a F M_b and M_b = M_b0 + rho_b F M_a.
    exitance_a = (
        exitance_0["a"] + values["reflectance_a"] * view_factor
        * exitance_0["b"]
    ) / denominator
    exitance_b = (
        exitance_0["b"] + values["reflectance_b"] * view_factor
        * exitance_0["a"]
    ) / denominator
    terrain_a = view_factor * exitance_b
    terrain_b = view_factor * exitance_a
    return {
        "irradiance_a": values["irradiance_a_0"] + terrain_a,
        "irradiance_b": values["irradiance_b_0"] + terrain_b,
        "terrain_irradiance_a": terrain_a,
        "terrain_irradiance_b": terrain_b,
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
