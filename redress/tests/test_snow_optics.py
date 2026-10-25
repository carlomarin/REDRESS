#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for REDRESS snow-optics wrappers."""

import numpy as np

from redress.optics.snow import albedo_kokhanovsky


def test_albedo_accepts_raster_solar_zenith_angles():
    solar_zenith = np.deg2rad(
        np.array([
            [20.0, 20.0, 45.0],
            [45.0, 65.0, 65.0],
        ])
    )
    diffuse_fraction = 0.25

    direct, diffuse, total = albedo_kokhanovsky(
        wls=800.0,
        sza=solar_zenith,
        dd=diffuse_fraction,
        ssa=20.0,
    )

    assert direct.shape == solar_zenith.shape
    assert diffuse.shape == solar_zenith.shape
    assert total.shape == solar_zenith.shape
    assert np.all(np.isfinite(direct))
    assert np.all(np.isfinite(diffuse))
    np.testing.assert_allclose(
        total,
        (1.0 - diffuse_fraction) * direct
        + diffuse_fraction * diffuse,
    )
