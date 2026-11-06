#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).

List of functions for the DEM horizon tool:
    - horizon
    - queues
    - horizon forward
    - horizon backward
    - slope

The algorithm is based on Dozier et al. 1980, 1990. The python implementation
 of the MATLAB code in MODimLAB (Sirguey et al. 2009) was applied here.
NB: Numba is used to speed up the process.
"""
from math import pi, cos, sin, sqrt, atan, degrees
import numpy as np
from numba import njit, prange


def queues(phi, B):
    """ Algorithm to build queues to hold profiles.

    Args:
        phi (int): the azimuth angle
        B (list): B

    Returns:
        Q (): Q
        S_list (): S_list"""

    # Set Q and S to None before they get updated
    Q = None
    S = None

    N1 = B[0]
    N2 = B[1]
    M1 = B[2]
    M2 = B[3]

    if phi <= pi / 2:
        for y in range(-M1, M2 + 1, 1):
            for x in range(-N1, N2 + 1, 1):
                xp = round(x * cos(phi) - y * sin(phi))

                # Update Q
                if Q is None:
                    Q = {int(xp + N1 + M2): [[0, 0]]}
                else:
                    if int(xp + N1 + M2) not in Q:
                        Q.update({int(xp + N1 + M2): [[0, 0]]})

                Q[int(xp + N1 + M2)].append([x, y])

                # Update S
                if S is None:
                    S = {int(xp + N1 + M2): 0}  # If not, create dictionnary
                else:
                    if max(S.keys()) < int(xp + N1 + M2):
                        S.update({int(xp + N1 + M2): 0})

                # Additional trick to replace Matlab dynamic allocation
                if not int(xp + N1 + M2) in S:
                    S.update({int(xp + N1 + M2): 0})

                S.update({int(xp + N1 + M2): S[int(xp + N1 + M2)] + 1})

    else:
        for y in range(M2, -M1 - 1, -1):  # Replaces the while loop in MODimLAB
            for x in range(-N1, N2 + 1, 1):
                xp = round(x * cos(phi) - y * sin(phi))

                # Calculate Q
                if Q is None:
                    Q = {int(xp + N1 + M2): [[0, 0]]}
                else:
                    if not int(xp + N1 + M2) in Q:
                        Q.update({int(xp + N1 + M2): [[0, 0]]})

                Q[int(xp + N1 + M2)].append([x, y])

                if S is None:
                    S = {int(xp + N1 + M2): 0}
                else:
                    if max(S.keys()) < int(xp + N1 + M2):
                        S.update({int(xp + N1 + M2): 0})

                # Additional trick to replace Matlab dynamic allocation
                if not int(xp + N1 + M2) in S:
                    S.update({int(xp + N1 + M2): 0})

                S.update({int(xp + N1 + M2): S[int(xp + N1 + M2)] + 1})

    # Convert dictionary to list
    S_list = [0] * max(S.keys())
    for i in S.keys():
        S_list[i - 1] = S[i]

    return Q, S_list


@njit
def slope_nb(i, j, A, D):
    """ Slope algorithm.

    Args:
        i (int): index of starting position
        j (int): index of current position
        A (list): Altitude
        D (list): Distance
    Returns:
        s (float): slope"""
    if A[j] <= A[i]:
        s = 0
    else:
        s = (A[j] - A[i]) / (D[j] - D[i])
    return s


@njit
def hrz_fwd_nb(A, D):
    """Fast one-dimensional algorithm for the forward direction.

    Args:
        A (list): Altitude
        D (list): Distance

    Returns:
        H (list): Horizon points in the forward direction
    """

    H = [0] * len(A)
    H[len(A) - 1] = len(A) - 1

    for i in range(len(A) - 2, 0 - 1, -1):
        j = i + 1
        found = 0
        while found == 0:
            if slope_nb(i, j, A, D) < slope_nb(j, H[j], A, D):
                j = H[j]
            else:
                found = 1
                if slope_nb(i, j, A, D) > slope_nb(j, H[j], A, D):
                    H[i] = j
                elif slope_nb(i, j, A, D) == 0:
                    H[i] = i
                else:
                    H[i] = H[j]
    return H


@njit
def hrz_bwd_nb(A, D):
    """Fast one-dimensional algorithm for the backward direction

    Args:
        A (list): Altitude
        D (list): Distance

    Returns:
        H (list): Horizon points in the forward direction"""

    H = []
    H.append(0)
    for i in range(1, len(A), 1):
        j = i - 1
        found = 0
        while found == 0:
            if slope_nb(i, j, A, D) > slope_nb(j, H[j], A, D):
                j = H[j]
            else:
                found = 1
                if slope_nb(i, j, A, D) < slope_nb(j, H[j], A, D):
                    H.append(j)
                elif slope_nb(i, j, A, D) == 0:
                    H.append(i)
                else:
                    H.append(H[j])
    return H


def dozier_horizon(dem_array, dem_pixel_size, phi):
    """ Algorithm to calculate horizon functions along profiles.

    The calculations are performed for profiles rotated by the azimuth angle
    phi.

    Args:
        dem_array (ndarray): DEM array
        pixel_size (int, int): Pixel size (x and y directions) of the DEM array
            in meters
        phi (int): azimuth angle

    Returns:
        :rtype: (list, list): a tuple containing a list with the
            elevation to the horizon for the profile and a list with the
            distance to the horizon for the profile (meters)."""

    row, col = dem_array.shape  # DEM size

    dx = dem_pixel_size  # Get pixel size
    dy = dem_pixel_size

    N1 = int(np.fix((col - 1) / 2.))  # Decimal point important!
    N2 = int(np.ceil((col - 1) / 2.))
    M1 = int(np.fix((row - 1) / 2.))
    M2 = int(np.ceil((row - 1) / 2.))

    # Call the queues function
    queuout, S = queues(phi, [N1, N2, M1, M2])

    if phi <= pi / 2:
        J1 = int(round(N1 * cos(phi) + M2 * sin(phi)))
        J2 = int(round(N2 * cos(phi) + M1 * sin(phi)))
    else:
        J1 = int(round(-N2 * cos(phi) + M2 * sin(phi)))
        J2 = int(round(-N1 * cos(phi) + M1 * sin(phi)))

    # Initialise the arrays Dh and Eh
    Dh = np.zeros(shape=(2, row, col))
    Eh = np.zeros(shape=(2, row, col))

    for xp in range(-J1, J2 + 1, 1):
        C = queuout[(xp + N1 + M2)]
        C = C[1:]

        # Initialise A and D
        A = []
        D = []

        # Populate A and D (A for altitude and D for distance)
        for j in range(0, S[int(xp + N1 + M2 - 1)], 1):
            x = C[j][0]
            y = C[j][1]

            A.append(dem_array[y + M1, x + N1])
            D.append(dx * x * sin(phi) + dy * y * cos(phi))

        # Convert profiles before entering Numba-compiled functions. Passing
        # Python lists relies on Numba's deprecated reflected-list support.
        A = np.asarray(A, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64)

        # Use fast algorithms for the forward and backward directions
        Hf = hrz_fwd_nb(A, D)
        Hb = hrz_bwd_nb(A, D)

        for j in range(0, S[int(xp + N1 + M2 - 1)], 1):
            x = C[j][0]
            y = C[j][1]

            if D[Hf[j]] - D[j] == 0:
                Dh[0][int(y + M1), int(x + N1)] = D[-1] - D[j] + sqrt(
                    dx ** 2 + dy ** 2
                )
                Eh[0][int(y + M1), int(x + N1)] = 0
            else:
                Dh[0][int(y + M1), int(x + N1)] = D[Hf[j]] - D[j] + sqrt(
                    dx ** 2 + dy ** 2
                )
                Eh[0][int(y + M1), int(x + N1)] = degrees(
                    atan((A[Hf[j]] - A[j]) / (D[Hf[j]] - D[j]))
                )
            if D[j] - D[Hb[j]] == 0:
                Dh[1][int(y + M1), int(x + N1)] = D[j] - D[0] + sqrt(
                    dx ** 2 + dy ** 2
                )
                Eh[1][int(y + M1), int(x + N1)] = 0
            else:
                Dh[1][int(y + M1), int(x + N1)] = D[j] - D[Hb[j]] + sqrt(
                    dx ** 2 + dy ** 2
                )
                Eh[1][int(y + M1), int(x + N1)] = degrees(
                    atan((A[Hb[j]] - A[j]) / (D[j] - D[Hb[j]]))
                )
    return Eh, Dh


@njit(parallel=True, cache=True)
def _ray_march_horizons(
    altitude,
    pixel_size,
    azimuths,
    minimum_distance,
):
    """Trace every crossed grid cell and maximize its bilinear surface."""

    directions = azimuths.size
    rows, columns = altitude.shape
    elevation = np.zeros((directions, rows, columns))
    distance = np.zeros((directions, rows, columns))
    epsilon = 1.0e-12
    infinity = np.inf

    for task in prange(directions * rows):
        direction = task // rows
        row = task % rows
        # Rows increase southward and columns eastward.
        row_step = -cos(azimuths[direction])
        column_step = sin(azimuths[direction])

        for column in range(columns):
            origin = altitude[row, column]
            best_tangent = 0.0
            best_distance = 0.0
            last_inside = 0.0

            # Select the first bilinear cell entered by the ray. A ray lying
            # on a grid line may use either adjacent cell because the surface
            # is continuous on the shared boundary.
            if row_step > epsilon:
                cell_row = row
            elif row_step < -epsilon:
                cell_row = row - 1
            else:
                cell_row = min(row, rows - 2)

            if column_step > epsilon:
                cell_column = column
            elif column_step < -epsilon:
                cell_column = column - 1
            else:
                cell_column = min(column, columns - 2)

            travelled = 0.0
            while (
                0 <= cell_row < rows - 1
                and 0 <= cell_column < columns - 1
            ):
                if row_step > epsilon:
                    row_crossing = (
                        cell_row + 1.0 - row
                    ) / row_step
                elif row_step < -epsilon:
                    row_crossing = (cell_row - row) / row_step
                else:
                    row_crossing = infinity

                if column_step > epsilon:
                    column_crossing = (
                        cell_column + 1.0 - column
                    ) / column_step
                elif column_step < -epsilon:
                    column_crossing = (
                        cell_column - column
                    ) / column_step
                else:
                    column_crossing = infinity

                next_crossing = min(row_crossing, column_crossing)
                if next_crossing <= travelled + epsilon:
                    break

                # Along the ray, a bilinear cell is a quadratic function
                # z(t) = q2*t^2 + q1*t + q0. The horizon tangent is
                # (z(t) - z_origin) / (t * pixel_size), whose only possible
                # interior stationary point is available analytically.
                z00 = altitude[cell_row, cell_column]
                z01 = altitude[cell_row, cell_column + 1]
                z10 = altitude[cell_row + 1, cell_column]
                z11 = altitude[cell_row + 1, cell_column + 1]
                row_offset = row - cell_row
                column_offset = column - cell_column
                cross = z11 - z10 - z01 + z00
                q2 = cross * row_step * column_step
                q1 = (
                    (z10 - z00) * row_step
                    + (z01 - z00) * column_step
                    + cross * (
                        row_offset * column_step
                        + column_offset * row_step
                    )
                )
                q0 = (
                    z00
                    + (z10 - z00) * row_offset
                    + (z01 - z00) * column_offset
                    + cross * row_offset * column_offset
                )

                segment_start = max(travelled, minimum_distance)
                if segment_start < next_crossing - epsilon:
                    start_elevation = (
                        (q2 * segment_start + q1) * segment_start + q0
                    )
                    tangent = (
                        start_elevation - origin
                    ) / (segment_start * pixel_size)
                    if tangent > best_tangent:
                        best_tangent = tangent
                        best_distance = segment_start

                    endpoint_elevation = (
                        (q2 * next_crossing + q1) * next_crossing + q0
                    )
                    tangent = (
                        endpoint_elevation - origin
                    ) / (next_crossing * pixel_size)
                    if tangent > best_tangent:
                        best_tangent = tangent
                        best_distance = next_crossing

                    if abs(q2) > epsilon:
                        critical_squared = (q0 - origin) / q2
                        if critical_squared > 0.0:
                            critical = sqrt(critical_squared)
                            if (
                                critical > segment_start + epsilon
                                and critical < next_crossing - epsilon
                            ):
                                critical_elevation = (
                                    (q2 * critical + q1) * critical + q0
                                )
                                tangent = (
                                    critical_elevation - origin
                                ) / (critical * pixel_size)
                                if tangent > best_tangent:
                                    best_tangent = tangent
                                    best_distance = critical

                travelled = next_crossing
                last_inside = next_crossing
                tolerance = epsilon * max(1.0, next_crossing)
                if abs(row_crossing - next_crossing) <= tolerance:
                    if row_step > epsilon:
                        cell_row += 1
                    elif row_step < -epsilon:
                        cell_row -= 1
                if abs(column_crossing - next_crossing) <= tolerance:
                    if column_step > epsilon:
                        cell_column += 1
                    elif column_step < -epsilon:
                        cell_column -= 1

            elevation[direction, row, column] = degrees(atan(best_tangent))
            # Without a rising horizon, report the distance to the DEM edge.
            if best_distance == 0.0:
                best_distance = last_inside
            distance[direction, row, column] = best_distance * pixel_size
    return elevation, distance


def ray_march_horizon(
    dem_array,
    dem_pixel_size,
    azimuths,
    minimum_distance_pixels=0.5,
):
    """Horizon elevation and distance on a bilinearly interpolated DEM.

    From every pixel, a straight ray is followed at each compass azimuth
    (radians, clockwise from north) until the DEM edge. Grid traversal visits
    every bilinear cell crossed by the ray. Within each cell, the terrain
    elevation is quadratic along the ray, so the maximum horizon tangent is
    evaluated at the cell-segment boundaries and at its possible analytical
    stationary point. The result is exact for the bilinearly interpolated
    raster surface beyond ``minimum_distance_pixels``; it does not depend on
    a distance step that could skip a narrow ridge. The default half-pixel
    exclusion keeps the local surface derivative in the separate self-shadow
    calculation.

    Returns horizon elevation in degrees, never below zero, and the
    horizontal distance to the horizon in metres, both shaped
    (direction, row, column).
    """

    altitude = np.ascontiguousarray(dem_array, dtype=np.float64)
    if altitude.ndim != 2 or min(altitude.shape) < 2:
        raise ValueError("dem_array must be a 2-D grid of at least 2 by 2")
    if not np.all(np.isfinite(altitude)):
        raise ValueError("dem_array must not contain missing elevations")
    if not dem_pixel_size > 0.0:
        raise ValueError("dem_pixel_size must be positive")
    if not minimum_distance_pixels > 0.0:
        raise ValueError("minimum_distance_pixels must be positive")
    return _ray_march_horizons(
        altitude,
        float(dem_pixel_size),
        np.ascontiguousarray(azimuths, dtype=np.float64),
        float(minimum_distance_pixels),
    )
