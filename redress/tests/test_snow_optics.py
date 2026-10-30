#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for REDRESS snow-optics wrappers."""

import numpy as np

from redress.optics.snow import (
    albedo_kokhanovsky,
    brf_kokhanovsky,
    directional_albedo_kokhanovsky,
    lap_impurities,
)


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


def test_hdr_by_reciprocity_uses_local_view_zenith(monkeypatch):
    """Evaluate direct albedo at eff_vza to obtain Eq. 5's a_v."""

    def fake_compute_alpha(wavelength_m, ssa, **kwargs):
        return 0.49

    monkeypatch.setattr(
        "redress.optics.snow.so.compute_alpha",
        fake_compute_alpha,
    )
    local_view_zenith = np.deg2rad(
        np.array([
            [5.0, 25.0],
            [25.0, 40.0],
        ])
    )

    hdr = directional_albedo_kokhanovsky(
        wls=800.0,
        zenith=local_view_zenith,
        ssa=20.0,
    )

    expected = np.exp(
        -0.7
        * (3.0 / 7.0)
        * (1.0 + 2.0 * np.cos(local_view_zenith))
    )
    np.testing.assert_allclose(hdr, expected)


def test_lap_concentrations_are_converted_to_mass_fractions():
    impurities = lap_impurities(
        black_carbon_ng_g=50.0,
        dust_ug_g=20.0,
        dust_density_kg_m3=2600.0,
    )

    assert np.isclose(impurities["BC"], 50e-9)
    assert np.isclose(impurities["dust"][0], 20e-6)
    assert impurities["dust"][1] == 2600.0

    skiles_impurities = lap_impurities(
        dust_ug_g=20.0,
        dust_model="skiles2014",
    )
    assert "dust_skiles" in skiles_impurities


def test_black_carbon_reduces_visible_albedo_and_brf():
    solar_zenith = np.deg2rad(45.0)
    clean_albedo = albedo_kokhanovsky(
        wls=500.0,
        sza=solar_zenith,
        dd=0.25,
        ssa=20.0,
    )[2]
    polluted_albedo = albedo_kokhanovsky(
        wls=500.0,
        sza=solar_zenith,
        dd=0.25,
        ssa=20.0,
        impurities=lap_impurities(black_carbon_ng_g=200.0),
    )[2]

    clean_brf = brf_kokhanovsky(
        solar_zenith, 0.0, 0.0, 500.0, 20.0
    )
    polluted_brf = brf_kokhanovsky(
        solar_zenith,
        0.0,
        0.0,
        500.0,
        20.0,
        impurities=lap_impurities(black_carbon_ng_g=200.0),
    )

    assert polluted_albedo < clean_albedo
    assert polluted_brf < clean_brf
