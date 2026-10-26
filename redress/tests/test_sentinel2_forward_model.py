#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Sentinel-2 spectral integration in the REDRESS model."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from redress.inputs.products import Model


class _FakeRadiativeTransferModel:
    wavelengths = []

    def __init__(self):
        self.outputs = None

    def run(self, sza, saa, vza, vaa, wavelength, altitude, aerosol, **kwargs):
        self.wavelengths.append(wavelength)
        self.outputs = SimpleNamespace(
            direct_solar_irradiance=3.0,
            diffuse_solar_irradiance=1.0,
        )


def _constant_scene(shape=(2, 2)):
    zeros = np.zeros(shape, dtype=np.float64)
    angles = xr.Dataset(
        {
            "SZA": (("x", "y"), zeros.copy()),
            "SAA": (("x", "y"), zeros.copy()),
            "VZA": (("x", "y"), zeros.copy()),
            "VAA": (("x", "y"), zeros.copy()),
        }
    )
    topography = xr.Dataset(
        {
            "altitude": (("x", "y"), zeros.copy()),
            "eff_sza": (("x", "y"), zeros.copy()),
            "eff_vza": (("x", "y"), zeros.copy()),
        }
    )
    return angles, topography


def test_model_integrates_snow_optics_over_sentinel2_response(monkeypatch):
    shape = (2, 2)
    received_impurities = []

    def fake_albedo(wavelength_nm, sza, diffuse_ratio, ssa, **kwargs):
        received_impurities.append(kwargs["impurities"])
        direct = np.full(shape, wavelength_nm)
        diffuse = np.full(shape, 2.0 * wavelength_nm)
        total = np.full(shape, 3.0 * wavelength_nm)
        return direct, diffuse, total

    def fake_brf(theta_i, theta_v, phi, wavelength_nm, ssa, **kwargs):
        received_impurities.append(kwargs["impurities"])
        return np.full(shape, wavelength_nm)

    monkeypatch.setattr(
        "redress.inputs.products.snow.albedo_kokhanovsky", fake_albedo
    )
    monkeypatch.setattr(
        "redress.inputs.products.snow.brf_kokhanovsky", fake_brf
    )

    _FakeRadiativeTransferModel.wavelengths = []
    model = Model()
    model.set_sentinel2("S2B", ["B05"])
    model.ssa = 20.0
    model.set_lap(black_carbon_ng_g=50.0, dust_ug_g=20.0)
    model.angles, model.topo_bands = _constant_scene(shape)
    model.rt_model = _FakeRadiativeTransferModel
    model.set_rt_options(aero="None")

    model.compute_albedo()
    model.compute_brf()

    effective_wavelength = model.meta["wavelengths"][0]
    np.testing.assert_allclose(
        model.albedo["direct"][0], effective_wavelength
    )
    np.testing.assert_allclose(
        model.albedo["diffuse"][0], 2.0 * effective_wavelength
    )
    np.testing.assert_allclose(
        model.albedo["total"][0], 3.0 * effective_wavelength
    )
    np.testing.assert_allclose(model.brf[0][1], effective_wavelength)

    assert len(_FakeRadiativeTransferModel.wavelengths) == 1
    assert _FakeRadiativeTransferModel.wavelengths[0] is \
        model.meta["band_definitions"][0].py6s_filter
    assert received_impurities
    assert all(
        impurities == model.snow_impurities
        for impurities in received_impurities
    )
