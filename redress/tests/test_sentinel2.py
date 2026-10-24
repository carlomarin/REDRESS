#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Sentinel-2 MSI band definitions."""

import numpy as np
import pytest

from redress.inputs.sentinel2 import (
    SENTINEL2_BAND_NAMES,
    normalize_sentinel2_platform,
    sentinel2_bands,
)


def test_all_sentinel2b_bands_have_valid_spectral_responses():
    bands = sentinel2_bands("S2B")

    assert tuple(band.name for band in bands) == SENTINEL2_BAND_NAMES
    for band in bands:
        wavelength_nm = band.wavelength_nm
        response = band.spectral_response

        assert wavelength_nm.shape == response.shape
        assert np.all(np.diff(wavelength_nm) > 0)
        assert wavelength_nm[0] <= band.effective_wavelength_nm
        assert band.effective_wavelength_nm <= wavelength_nm[-1]


def test_band_selection_preserves_requested_order_and_resolution():
    bands = sentinel2_bands("Sentinel-2A", ["B12", "8A", "2"])

    assert tuple(band.name for band in bands) == ("B12", "B8A", "B02")
    assert tuple(band.native_resolution_m for band in bands) == (20, 20, 10)
    assert tuple(band.py6s_filter_name for band in bands) == (
        "S2A_MSI_12", "S2A_MSI_8A", "S2A_MSI_02",
    )


@pytest.mark.parametrize("platform", ["S2C", "Landsat-8", ""])
def test_unsupported_platform_is_rejected(platform):
    with pytest.raises(ValueError, match="Unsupported Sentinel-2 platform"):
        normalize_sentinel2_platform(platform)


def test_duplicate_band_selection_is_rejected():
    with pytest.raises(ValueError, match="duplicate bands"):
        sentinel2_bands("S2B", ["B02", "2"])
