#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from scipy.ndimage import binary_closing
from scipy.sparse import csr_matrix


@njit
def horneslope(indem, dem_pixel_size):
    """ Calculate slope and aspect from a DEM.

    Using jit speeds up the function by a factor of 200.
    The algorithm is based on Horn's (1981) Finite Difference Weighted by
     reciprocal of Distance (Lee and Clarke, 2005). The algorithm has been
     shown to perform better than others (Lee and Clarke, 2005; Srinivasan
     and Engel, 1991; Dunn and Hickley, 1998). The Horn algorithm leads to
     a loss in local variability (smooths small dips and peaks), but doesn't
     overestimate the slopes or exagerate peaks. Jones (1998) showed that
     the algorithm performed well in comparison to reference surfaces.

    Args:
        indem (ndarray): DEM array
        pixel_size (int, int): Pixel size (x and y directions) of the DEM array
            in meters.
    Returns:
        :rtype: (ndarray, ndarray): a tuple containing a ndarray with slope
            values in radians and a ndarray with aspect values in radians."""

    # Initialise the slope and aspect arrays
    output_sl = np.full(indem.shape, np.nan, dtype=np.float64)
    output_asp = np.full(indem.shape, np.nan, dtype=np.float64)

    rows, cols = indem.shape

    for x in range(1, cols - 1):
        for y in range(1, rows - 1):
            dzx = ((indem[y - 1, x + 1] + 2 * indem[y, x + 1] +
                    indem[y + 1, x + 1]) -
                   (indem[y - 1, x - 1] + 2 *
                    indem[y, x - 1] + indem[y + 1, x - 1])) /\
                         (8 * abs(dem_pixel_size[0]))
            dzy = ((indem[y + 1, x - 1] + 2 * indem[y + 1, x] +
                    indem[y + 1, x + 1]) -
                   (indem[y - 1, x - 1] + 2 * indem[y - 1, x] +
                    indem[y - 1, x + 1])) / (8 * abs(dem_pixel_size[1]))

            slope = np.sqrt(dzx ** 2 + dzy ** 2)
            aspect = 180 / np.pi * np.arctan2(dzy, -dzx)

            output_sl[y, x] = np.arctan(slope) * 180 / np.pi

            if output_sl[y, x] == 0:
                # If no slope, set aspect to 0. Setting it to nan messes up the
                # reinterpolation later.
                output_asp[y, x] = 0
            else:
                if aspect > 90:
                    output_asp[y, x] = 360 - aspect + 90
                else:
                    output_asp[y, x] = 90 - aspect

    return np.deg2rad(output_sl), np.deg2rad(output_asp)


def skyview(
    horz_data,
    slp,
    asp,
    terrain_configuration_method="hemispheric_closure",
):
    """ Compute the sky view factor.

    Compute the sky view factor for every pixel, i.e. the ratio between diffuse
    irradiance received to that on unobstructed surface according to its slope,
    aspect and horizon. Algorithm by Dozier et al. 1981, 1990.

    Args:
        horz_data (ndarray): horizon elevation data computed from the DEM
        slp (ndarray): slope calculated from the DEM
        asp (ndarray): aspect calculated from the DEM
        terrain_configuration_method (str): ``hemispheric_closure`` uses the
            complete outward hemisphere, Ct = 1 - Vd. ``dozier_frew`` uses
            Ct = (1 + cos(slope)) / 2 - Vd. ``redress_bowl`` is retained as
            an alias for ``hemispheric_closure``.

    Returns:
        vd (ndarray): skyview factor
        ct (ndarray): terrain configuration factor
    """

    horizon = np.asarray(horz_data, dtype=np.float64)
    slope = np.asarray(slp, dtype=np.float64)
    aspect = np.asarray(asp, dtype=np.float64)
    if horizon.ndim != 3:
        raise ValueError("horizon data must have dimensions (azimuth, row, column)")
    if horizon.shape[1:] != slope.shape or slope.shape != aspect.shape:
        raise ValueError("horizon, slope and aspect grids must be spatially aligned")

    rows, columns = horizon.shape[1:]
    directions = horizon.shape[0]

    # Preallocate variables
    dphi = 2 * np.pi / directions
    cos_slope = np.cos(slope)
    sin_slope = np.sin(slope)

    horizon_zenith = np.deg2rad(90.0 - np.clip(horizon, 0.0, 90.0))
    cos_horizon = np.cos(horizon_zenith)
    sin_horizon = np.sin(horizon_zenith)

    # Compute vd (skyview factor)
    phi = np.multiply.outer(
        np.pi - np.arange(directions) * dphi,
        np.ones((rows, columns)),
    )
    cos_relative_azimuth = np.cos(phi - aspect)

    azimuth_term = cos_relative_azimuth * (
        horizon_zenith - sin_horizon * cos_horizon
    )
    integrand = (
        cos_slope * sin_horizon ** 2
        + sin_slope * azimuth_term
    )
    vd = (dphi / (2 * np.pi)) * np.sum(integrand, axis=0)
    vd = np.clip(vd, 0.0, 1.0)
    invalid = ~np.isfinite(slope) | ~np.isfinite(aspect)
    invalid |= ~np.all(np.isfinite(horizon), axis=0)
    vd[invalid] = np.nan

    if terrain_configuration_method == "dozier_frew":
        unobstructed_slope_sky = (1.0 + cos_slope) / 2.0
        ct = np.clip(unobstructed_slope_sky - vd, 0.0, 1.0)
    elif terrain_configuration_method in (
        "hemispheric_closure",
        "redress_bowl",
    ):
        ct = 1.0 - vd
    else:
        raise ValueError(
            "terrain_configuration_method must be 'hemispheric_closure', "
            "'dozier_frew', or 'redress_bowl'"
        )
    ct[invalid] = np.nan

    return vd, ct


def horizon_elevation_at_azimuth(horz_data, azimuth):
    """Interpolate horizon elevation at a compass azimuth.

    Horizon slices produced by :meth:`Dem.compute_horizon` are ordered from
    south towards decreasing compass azimuth. Interpolation is circular and
    linear in azimuth, avoiding the conservative maximum of the two adjacent
    horizon directions used by the previous implementation.
    """

    horizon = np.asarray(horz_data, dtype=np.float64)
    azimuth = np.asarray(azimuth, dtype=np.float64)
    if horizon.ndim != 3:
        raise ValueError("horizon data must have dimensions (azimuth, row, column)")
    if azimuth.shape != horizon.shape[1:]:
        azimuth = np.broadcast_to(azimuth, horizon.shape[1:])

    directions = horizon.shape[0]
    dphi = 2.0 * np.pi / directions
    position = np.mod(np.pi - np.mod(azimuth, 2.0 * np.pi), 2.0 * np.pi)
    position /= dphi
    lower = np.floor(position).astype(np.int64) % directions
    upper = (lower + 1) % directions
    weight = position - np.floor(position)

    lower_horizon = np.take_along_axis(
        horizon, lower[np.newaxis, ...], axis=0
    )[0]
    upper_horizon = np.take_along_axis(
        horizon, upper[np.newaxis, ...], axis=0
    )[0]
    return (1.0 - weight) * lower_horizon + weight * upper_horizon


def terrain_visibility(
    horz_data,
    zenith,
    azimuth,
    effective_zenith,
    min_cosine=0.0,
):
    """Return binary line-of-sight visibility above the local horizon.

    Visibility requires both a front-facing local surface and an unobstructed
    ray above the interpolated terrain horizon. The function is applicable to
    either the solar beam or the sensor viewing direction.
    """

    zenith = np.asarray(zenith, dtype=np.float64)
    effective_zenith = np.asarray(effective_zenith, dtype=np.float64)
    if not np.isfinite(min_cosine) or not 0.0 <= min_cosine < 1.0:
        raise ValueError("min_cosine must be finite and in [0, 1)")

    horizon_elevation = np.deg2rad(
        horizon_elevation_at_azimuth(horz_data, azimuth)
    )
    ray_elevation = np.pi / 2.0 - zenith
    # Treat a mathematically tangent ray as non-visible even when floating-
    # point evaluation leaves a small positive cosine. The tolerance is far
    # below any meaningful terrain or angular uncertainty.
    visibility_threshold = max(
        float(min_cosine),
        32.0 * np.finfo(np.float64).eps,
    )
    front_facing = np.cos(effective_zenith) > visibility_threshold
    above_horizon = ray_elevation > horizon_elevation
    visibility = np.logical_and(front_facing, above_horizon).astype(np.float64)

    invalid = (
        ~np.isfinite(zenith)
        | ~np.isfinite(effective_zenith)
        | ~np.isfinite(horizon_elevation)
    )
    return np.where(invalid, np.nan, visibility)


def subpixel_visibility_fraction(
    visibility,
    row_factor,
    column_factor=None,
):
    """Area-average an aligned binary visibility grid to coarser pixels.

    This operation produces the sub-pixel illuminated or visible fraction
    used by REDRESS. It is valid when the fine DEM grid is exactly nested in
    the satellite grid. General reprojection must instead use an area-weighted
    geospatial resampler.
    """

    if column_factor is None:
        column_factor = row_factor
    for name, factor in (
        ("row_factor", row_factor),
        ("column_factor", column_factor),
    ):
        if not isinstance(factor, int) or factor < 1:
            raise ValueError(f"{name} must be a positive integer")

    values = np.asarray(visibility, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("visibility must be a two-dimensional grid")
    rows, columns = values.shape
    if rows % row_factor or columns % column_factor:
        raise ValueError(
            "fine visibility dimensions must be divisible by aggregation "
            "factors"
        )
    blocks = values.reshape(
        rows // row_factor,
        row_factor,
        columns // column_factor,
        column_factor,
    )
    valid_count = np.sum(np.isfinite(blocks), axis=(1, 3))
    total = np.nansum(blocks, axis=(1, 3))
    return np.divide(
        total,
        valid_count,
        out=np.full(total.shape, np.nan, dtype=np.float64),
        where=valid_count > 0,
    )


def shadows(
    horz_data,
    slp,
    asp,
    sza,
    eff_sza,
    saa,
    min_cosine=0.0,
    clean_cast_shadows=False,
):
    """Calculate self, cast and total shadows from a DEM.

    Args:
        horz_data (ndarray): horizon elevation data computed from the DEM
        slp (ndarray): slope calculated from the DEM
        asp (ndarray): aspect calculated from the DEM
        sza (ndarray): solar zenith angles gridded as horz_data
        eff_sza (ndarray): effective solar zenith angles gridded as horz_data
        saa (ndarray): solar azimuth angles gridded as horz_data

    Returns:
        b (ndarray): combined beam visibility (1 = sunlit, 0 = shadow).
        bs (ndarray): local-surface visibility (1 = front-facing).
        bc (ndarray): terrain-horizon visibility (1 = unobstructed)."""

    del slp, asp  # Retained in the public signature for backward compatibility.

    if not np.isfinite(min_cosine) or not 0.0 <= min_cosine < 1.0:
        raise ValueError("min_cosine must be finite and in [0, 1)")

    effective_zenith = np.asarray(eff_sza, dtype=np.float64)
    visibility_threshold = max(
        float(min_cosine),
        32.0 * np.finfo(np.float64).eps,
    )
    self_visibility = np.where(
        np.isfinite(effective_zenith),
        (
            np.cos(effective_zenith) > visibility_threshold
        ).astype(np.float64),
        np.nan,
    )
    cast_visibility = terrain_visibility(
        horz_data,
        sza,
        saa,
        np.zeros_like(eff_sza),
        min_cosine=0.0,
    )
    if clean_cast_shadows:
        valid = np.isfinite(cast_visibility)
        cleaned = binary_closing(
            np.where(valid, cast_visibility, 0.0).astype(bool),
            structure=np.ones((3, 3), dtype=bool),
        ).astype(np.float64)
        cast_visibility = np.where(valid, cleaned, np.nan)

    b = self_visibility * cast_visibility
    bs = self_visibility
    bc = cast_visibility

    return (b, bs, bc)


def effective_zenith_angle(zenith, azimuth, slp, asp):
    """Compute the effective zenith angle.

    The function computes the effective zenith angle given a solar position
     determined by a zenith and azimuth angle, and the slope and aspect of the
     surface.

     Args:
        zenith (int, float, ndarray): solar zenith angle
        azimuth (int, float, ndarray): solar azimuth angle
        slp (int, float, ndarray): slope
        asp (int, float, ndarray): aspect

    Returns:
        (int, float, ndarray): effective solar zenith angle."""

    mu = np.cos(zenith) * np.cos(slp) + np.sin(zenith) * np.sin(slp) *\
        np.cos(azimuth - asp)

    return np.arccos(np.clip(mu, -1.0, 1.0))


def local_relative_azimuth(
    illumination_zenith,
    illumination_azimuth,
    viewing_zenith,
    viewing_azimuth,
    slp,
    asp,
):
    """Return the geometric relative azimuth in the local slope plane.

    The returned unsigned angle follows the common satellite convention:
    zero means that the surface-to-sun and surface-to-sensor projections have
    the same azimuth. The calculation uses the invariant angle between the
    global direction vectors and their local zenith angles.

    When either direction is normal to the local surface, relative azimuth is
    undefined and is set to zero. It then has no effect on the ART scattering
    angle because the corresponding sine-of-zenith term is zero.
    """

    illumination_zenith = np.asarray(
        illumination_zenith, dtype=np.float64
    )
    illumination_azimuth = np.asarray(
        illumination_azimuth, dtype=np.float64
    )
    viewing_zenith = np.asarray(viewing_zenith, dtype=np.float64)
    viewing_azimuth = np.asarray(viewing_azimuth, dtype=np.float64)

    local_incidence = effective_zenith_angle(
        illumination_zenith, illumination_azimuth, slp, asp
    )
    local_view_zenith = effective_zenith_angle(
        viewing_zenith, viewing_azimuth, slp, asp
    )
    cosine_direction_angle = (
        np.cos(illumination_zenith) * np.cos(viewing_zenith)
        + np.sin(illumination_zenith) * np.sin(viewing_zenith)
        * np.cos(illumination_azimuth - viewing_azimuth)
    )
    denominator = np.sin(local_incidence) * np.sin(local_view_zenith)
    cosine_relative_azimuth = np.divide(
        cosine_direction_angle
        - np.cos(local_incidence) * np.cos(local_view_zenith),
        denominator,
        out=np.ones_like(denominator, dtype=np.float64),
        where=np.abs(denominator) > 1e-12,
    )
    relative_azimuth = np.arccos(
        np.clip(cosine_relative_azimuth, -1.0, 1.0)
    )
    invalid = (
        ~np.isfinite(illumination_zenith)
        | ~np.isfinite(illumination_azimuth)
        | ~np.isfinite(viewing_zenith)
        | ~np.isfinite(viewing_azimuth)
        | ~np.isfinite(local_incidence)
        | ~np.isfinite(local_view_zenith)
    )
    return np.where(invalid, np.nan, relative_azimuth)


def art_relative_azimuth(geometric_relative_azimuth):
    """Convert geometric relative azimuth to the ART snow-BRF convention.

    The geometric convention has zero for equal sun and sensor azimuths. In
    the asymptotic radiative-transfer convention used by
    :func:`redress.optics.snow.brf_kokhanovsky`, zero denotes the
    forward-scattering side and pi denotes exact backscatter.
    """

    relative_azimuth = np.asarray(
        geometric_relative_azimuth, dtype=np.float64
    )
    unsigned = np.arccos(np.clip(np.cos(relative_azimuth), -1.0, 1.0))
    return np.pi - unsigned



# Cosine below which a neighbour is treated as seen edge-on. It corresponds
# to about 6e-5 degrees and only removes rounding residuals.
_GRAZING_COSINE = 1e-6


@dataclass(frozen=True)
class VisibleNeighbourGeometry:
    """Sparse DEM-only operator for terrain-view-factor averaging.

    Rows represent receiving pixels and columns represent visible terrain
    pixels. Matrix entries are discrete terrain view factors. The operator
    depends only on the DEM geometry, pixel size, ray radius and azimuth
    count, so it can be reused for every spectral band and reflection order.
    """

    operator: csr_matrix
    shape: tuple
    view_factor: np.ndarray


@njit
def _neighbour_view_factor(
    row,
    col,
    target_row,
    target_col,
    run,
    rise,
    east,
    north,
    angular_step,
    pixel_size,
    normal_east,
    normal_north,
    normal_up,
):
    """Return one discrete terrain view factor, or zero if back-facing."""

    distance = np.sqrt(run * run + rise * rise)
    ray_east = run * east / distance
    ray_north = run * north / distance
    ray_up = rise / distance
    cos_source = (
        normal_east[row, col] * ray_east
        + normal_north[row, col] * ray_north
        + normal_up[row, col] * ray_up
    )
    cos_target = -(
        normal_east[target_row, target_col] * ray_east
        + normal_north[target_row, target_col] * ray_north
        + normal_up[target_row, target_col] * ray_up
    )
    if not (
        cos_source > _GRAZING_COSINE
        and cos_target > _GRAZING_COSINE
    ):
        return 0.0

    # Sloped surface area of the polar sample cell.
    area = (
        run * angular_step * pixel_size
        / normal_up[target_row, target_col]
    )
    return cos_source * cos_target * area / (np.pi * distance * distance)


@njit(parallel=True)
def _visible_neighbour_counts(
    altitude,
    normal_east,
    normal_north,
    normal_up,
    pixel_size,
    radius_pixels,
    n_directions,
):
    """Count visible, mutually facing terrain samples for every pixel."""

    rows, cols = altitude.shape
    counts = np.zeros(rows * cols, dtype=np.int64)
    angular_step = 2.0 * np.pi / n_directions

    for row in prange(rows):
        for col in range(cols):
            if not (
                np.isfinite(altitude[row, col])
                and np.isfinite(normal_up[row, col])
            ):
                continue
            for direction in range(n_directions):
                azimuth = direction * angular_step
                east = np.sin(azimuth)
                north = np.cos(azimuth)
                max_tangent = -np.inf
                for step in range(1, radius_pixels + 1):
                    # Rows increase southward and columns eastward.
                    target_row = int(np.rint(row - step * north))
                    target_col = int(np.rint(col + step * east))
                    if not (
                        0 <= target_row < rows and 0 <= target_col < cols
                    ):
                        break
                    rise = altitude[target_row, target_col] - altitude[
                        row, col
                    ]
                    run = step * pixel_size
                    if not np.isfinite(rise):
                        break
                    tangent = rise / run
                    visible = tangent >= max_tangent - 1e-12
                    if tangent > max_tangent:
                        max_tangent = tangent
                    if not visible:
                        continue
                    weight = _neighbour_view_factor(
                        row,
                        col,
                        target_row,
                        target_col,
                        run,
                        rise,
                        east,
                        north,
                        angular_step,
                        pixel_size,
                        normal_east,
                        normal_north,
                        normal_up,
                    )
                    if weight > 0.0:
                        counts[row * cols + col] += 1
    return counts


@njit(parallel=True)
def _visible_neighbour_entries(
    altitude,
    normal_east,
    normal_north,
    normal_up,
    pixel_size,
    radius_pixels,
    n_directions,
    offsets,
):
    """Fill sparse target indices and view factors after the count pass."""

    rows, cols = altitude.shape
    indices = np.empty(offsets[-1], dtype=np.int64)
    weights = np.empty(offsets[-1], dtype=np.float64)
    angular_step = 2.0 * np.pi / n_directions

    for row in prange(rows):
        for col in range(cols):
            if not (
                np.isfinite(altitude[row, col])
                and np.isfinite(normal_up[row, col])
            ):
                continue
            source = row * cols + col
            position = offsets[source]
            for direction in range(n_directions):
                azimuth = direction * angular_step
                east = np.sin(azimuth)
                north = np.cos(azimuth)
                max_tangent = -np.inf
                for step in range(1, radius_pixels + 1):
                    target_row = int(np.rint(row - step * north))
                    target_col = int(np.rint(col + step * east))
                    if not (
                        0 <= target_row < rows and 0 <= target_col < cols
                    ):
                        break
                    rise = altitude[target_row, target_col] - altitude[
                        row, col
                    ]
                    run = step * pixel_size
                    if not np.isfinite(rise):
                        break
                    tangent = rise / run
                    visible = tangent >= max_tangent - 1e-12
                    if tangent > max_tangent:
                        max_tangent = tangent
                    if not visible:
                        continue
                    weight = _neighbour_view_factor(
                        row,
                        col,
                        target_row,
                        target_col,
                        run,
                        rise,
                        east,
                        north,
                        angular_step,
                        pixel_size,
                        normal_east,
                        normal_north,
                        normal_up,
                    )
                    if weight > 0.0:
                        indices[position] = target_row * cols + target_col
                        weights[position] = weight
                        position += 1
    return indices, weights


def build_visible_neighbour_geometry(
    altitude,
    slope,
    aspect,
    pixel_size,
    radius_pixels,
    n_directions=64,
):
    """Construct a reusable sparse visible-terrain view-factor operator."""

    if not isinstance(radius_pixels, (int, np.integer)) or radius_pixels < 1:
        raise ValueError("radius_pixels must be a positive integer")
    if not isinstance(n_directions, (int, np.integer)) or n_directions < 4:
        raise ValueError("n_directions must be an integer of at least 4")
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError("pixel_size must be positive and finite")

    altitude = np.asarray(altitude, dtype=np.float64)
    slope = np.asarray(slope, dtype=np.float64)
    aspect = np.asarray(aspect, dtype=np.float64)
    if altitude.ndim != 2:
        raise ValueError("altitude must be a two-dimensional array")
    if slope.shape != altitude.shape or aspect.shape != altitude.shape:
        raise ValueError("altitude, slope and aspect must have equal shapes")

    altitude = np.ascontiguousarray(altitude)
    normal_east = np.ascontiguousarray(np.sin(slope) * np.sin(aspect))
    normal_north = np.ascontiguousarray(np.sin(slope) * np.cos(aspect))
    normal_up = np.ascontiguousarray(np.cos(slope))
    counts = _visible_neighbour_counts(
        altitude,
        normal_east,
        normal_north,
        normal_up,
        float(pixel_size),
        int(radius_pixels),
        int(n_directions),
    )
    offsets = np.empty(counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    indices, weights = _visible_neighbour_entries(
        altitude,
        normal_east,
        normal_north,
        normal_up,
        float(pixel_size),
        int(radius_pixels),
        int(n_directions),
        offsets,
    )
    pixels = altitude.size
    operator = csr_matrix(
        (weights, indices, offsets),
        shape=(pixels, pixels),
    )
    operator.sum_duplicates()
    view_factor = np.asarray(operator.sum(axis=1)).reshape(altitude.shape)
    return VisibleNeighbourGeometry(
        operator=operator,
        shape=altitude.shape,
        view_factor=view_factor,
    )


def visible_neighbour_mean(
    values,
    altitude=None,
    slope=None,
    aspect=None,
    pixel_size=None,
    radius_pixels=None,
    n_directions=64,
    geometry=None,
):
    """Average neighbour fields weighted by their discrete view factor.

    Pass ``geometry`` to apply an existing sparse operator without tracing
    the DEM again. Otherwise rays are marched from every pixel in
    ``n_directions`` azimuths, one pixel per step, out to ``radius_pixels``
    and the operator is constructed first. A neighbour contributes only when
    it lies on or above the running elevation-angle maximum along the ray and
    both surfaces face each other beyond a rounding tolerance. Its weight is
    the view-factor element ``cos(theta_P) cos(theta_Q) dA / (pi d^2)``, so
    coplanar, hidden and back-facing terrain receive no weight.

    Args:
        values (ndarray): Field(s) to average, shaped (rows, cols) or
            (n_fields, rows, cols).
        altitude (ndarray): DEM in metres.
        slope (ndarray): Slope in radians.
        aspect (ndarray): Downslope azimuth in radians, clockwise from north.
        pixel_size (float): Square pixel size in metres.
        radius_pixels (int): Ray length in pixels.
        n_directions (int): Number of azimuth directions.
        geometry (VisibleNeighbourGeometry): Previously constructed sparse
            operator. When supplied, the DEM geometry arguments are ignored.

    Returns:
        (ndarray, ndarray): The weighted mean with the shape of ``values``,
            NaN where no neighbour is visible, and the summed view factor of
            the visible neighbours within the radius.
    """

    if geometry is None:
        geometry = build_visible_neighbour_geometry(
            altitude,
            slope,
            aspect,
            pixel_size,
            radius_pixels,
            n_directions=n_directions,
        )
    elif not isinstance(geometry, VisibleNeighbourGeometry):
        raise TypeError("geometry must be a VisibleNeighbourGeometry")

    values = np.asarray(values, dtype=np.float64)
    if values.ndim not in (2, 3):
        raise ValueError("values must have two or three dimensions")
    single_field = values.ndim == 2
    if single_field:
        values = values[np.newaxis]
    if values.shape[1:] != geometry.shape:
        raise ValueError("values and visible-neighbour geometry must align")

    flattened = values.reshape(values.shape[0], -1)
    finite = np.all(np.isfinite(flattened), axis=0)
    weighted_values = np.where(finite, flattened, 0.0)
    sums = np.vstack([
        geometry.operator.dot(field) for field in weighted_values
    ])
    if np.all(finite):
        weights = geometry.view_factor.ravel()
    else:
        weights = geometry.operator.dot(finite.astype(np.float64))
    mean = np.divide(
        sums,
        weights[np.newaxis],
        out=np.full_like(sums, np.nan),
        where=weights[np.newaxis] > 0,
    ).reshape(values.shape)
    return (
        mean[0] if single_field else mean,
        weights.reshape(geometry.shape),
    )
