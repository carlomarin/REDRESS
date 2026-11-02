#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for REDRESS snow-optics wrappers."""

import numpy as np
import pytest

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


def _hemispheric_brf_integral(theta_i, wavelength_nm, ssa, **options):
    view_zenith = np.linspace(0.0, np.pi / 2.0, 361)[1:-1]
    azimuth = np.linspace(0.0, 2.0 * np.pi, 721)
    zenith_grid, azimuth_grid = np.meshgrid(
        view_zenith, azimuth, indexing="ij"
    )
    brf = brf_kokhanovsky(
        theta_i, zenith_grid, azimuth_grid, wavelength_nm, ssa, **options
    )
    integrand = brf * np.cos(zenith_grid) * np.sin(zenith_grid)
    return np.trapezoid(
        np.trapezoid(integrand, azimuth, axis=1), view_zenith
    ) / np.pi


def test_brf_integrates_to_the_directional_albedo():
    # B11 with coarse snow is where the asymptotic BRF overshoots most.
    for wavelength_nm in (560.0, 1610.0):
        for incidence_degree in (0.0, 45.0, 75.0):
            incidence = np.deg2rad(incidence_degree)
            albedo = directional_albedo_kokhanovsky(
                np.array(wavelength_nm), incidence, 20.0
            )
            assert _hemispheric_brf_integral(
                incidence, wavelength_nm, 20.0
            ) == pytest.approx(float(albedo), rel=2e-3)


def test_unclosed_brf_overshoots_the_albedo_in_swir():
    incidence = np.deg2rad(45.0)
    albedo = directional_albedo_kokhanovsky(np.array(1610.0), incidence, 20.0)

    unclosed = _hemispheric_brf_integral(
        incidence, 1610.0, 20.0, close_on_albedo=False
    )

    assert unclosed > 1.2 * float(albedo)


def test_closed_brf_is_reciprocal():
    forward = brf_kokhanovsky(0.4, 1.1, 0.7, 1610.0, 20.0)
    reverse = brf_kokhanovsky(1.1, 0.4, 0.7, 1610.0, 20.0)

    assert forward == pytest.approx(reverse, rel=1e-12)


def test_brf_shares_the_albedo_absorption(monkeypatch):
    def fake_compute_alpha(wavelength_m, ssa, **kwargs):
        return 0.49

    monkeypatch.setattr(
        "redress.optics.snow.so.compute_alpha", fake_compute_alpha
    )

    brf = brf_kokhanovsky(
        0.3, 0.2, 1.0, 800.0, 20.0, close_on_albedo=False
    )

    cos_i, cos_v = np.cos(0.3), np.cos(0.2)
    theta = np.degrees(np.arccos(
        -cos_i * cos_v + np.sin(0.3) * np.sin(0.2) * np.cos(1.0)
    ))
    phase = 11.1 * np.exp(-0.087 * theta) + 1.1 * np.exp(-0.014 * theta)
    r0 = (
        1.247 + 1.186 * (cos_i + cos_v) + 5.157 * cos_i * cos_v + phase
    ) / (4.0 * (cos_i + cos_v))
    u_i = 3.0 / 7.0 * (1.0 + 2.0 * cos_i)
    u_v = 3.0 / 7.0 * (1.0 + 2.0 * cos_v)
    assert brf == pytest.approx(r0 * np.exp(-0.7 * u_i * u_v / r0))


def test_tartes_directional_albedo_matches_tartes():
    import tartes

    zenith = np.deg2rad(np.array([[0.0, 30.0], [60.0, 80.0]]))

    albedo = directional_albedo_kokhanovsky(
        1610.0, zenith, 60.0, albedo_model="tartes"
    )

    expected = [
        tartes.albedo(
            1610e-9, 60.0, density=300.0,
            shape_parameterization="constant", B0=1.6, g0=0.845,
            refrac_index="w2008", dir_frac=1.0, sza=angle,
        )
        for angle in np.degrees(zenith).ravel()
    ]
    np.testing.assert_allclose(albedo.ravel(), expected, rtol=2e-3)


def test_brf_closes_on_the_tartes_albedo():
    incidence = np.deg2rad(30.0)
    albedo = directional_albedo_kokhanovsky(
        1610.0, incidence, 60.0, albedo_model="tartes"
    )

    assert _hemispheric_brf_integral(
        incidence, 1610.0, 60.0, albedo_model="tartes"
    ) == pytest.approx(float(albedo), rel=2e-3)
    # TARTES is brighter than ART where ice absorbs strongly.
    assert float(albedo) > 1.3 * float(
        directional_albedo_kokhanovsky(1610.0, incidence, 60.0)
    )


def test_tartes_albedo_rejects_impurities_and_unknown_models():
    with pytest.raises(NotImplementedError):
        albedo_kokhanovsky(
            500.0, 0.5, 0.25, 20.0,
            impurities=lap_impurities(black_carbon_ng_g=50.0),
            albedo_model="tartes",
        )
    with pytest.raises(ValueError, match="albedo_model"):
        directional_albedo_kokhanovsky(500.0, 0.5, 20.0, albedo_model="x")
