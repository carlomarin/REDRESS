#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Sentinel-2 MSI band definitions."""

import numpy as np
import pytest

from redress.inputs.sentinel2 import (
    SENTINEL2_BAND_NAMES,
    normalize_sentinel2_platform,
    sentinel2_bands,
    sentinel2_metadata,
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


def test_empty_band_selection_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        sentinel2_bands("S2B", [])


def test_spectral_integration_uses_the_band_response():
    band = sentinel2_bands("S2B", ["B05"])[0]

    constant = np.full(band.wavelength_nm.size, 0.75)
    linear = band.wavelength_nm.copy()

    assert band.integrate_spectral_values(constant) == pytest.approx(0.75)
    assert band.integrate_spectral_values(linear) == pytest.approx(
        band.effective_wavelength_nm
    )


def test_metadata_contains_model_ready_band_definitions():
    metadata = sentinel2_metadata("S2B", ["B02", "B03", "B11"])

    assert metadata["sensor"] == "MSI"
    assert metadata["platform"] == "S2B"
    assert metadata["bandlist"] == ["B02", "B03", "B11"]
    assert metadata["native_resolution_m"] == [10, 10, 20]
    assert len(metadata["wavelengths"]) == 3
    assert len(metadata["band_definitions"]) == 3
