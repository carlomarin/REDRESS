#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Sentinel-2 snow look-up table generation."""

import numpy as np

from redress.lut import create_sentinel2_snow_brf_lut


def test_small_snow_brf_lut_has_labelled_dimensions():
    lut = create_sentinel2_snow_brf_lut(
        platform="S2B",
        band_names=("B02", "B11"),
        specific_surface_area=(10.0, 30.0),
        solar_zenith_angle=(30.0,),
        view_zenith_angle=(0.0,),
        relative_azimuth_angle=(0.0, 90.0),
    )

    assert lut["brf"].dims == (
        "specific_surface_area",
        "solar_zenith_angle",
        "view_zenith_angle",
        "relative_azimuth_angle",
        "band",
    )
    assert lut["brf"].shape == (2, 1, 1, 2, 2)
    assert lut["band"].values.tolist() == ["B02", "B11"]
    assert np.all(np.isfinite(lut["brf"]))
    assert np.all(lut["brf"] >= 0)
