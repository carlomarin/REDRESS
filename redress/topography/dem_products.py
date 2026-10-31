#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
import numpy as np
from numba import njit
from scipy.ndimage import binary_closing


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

