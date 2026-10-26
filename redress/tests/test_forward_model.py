#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for REDRESS forward-model utilities."""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from redress.topography.forward_model import (
    Cases,
    _anisotropic_sky_factor,
    _window_mean,
    _solar_irradiance_density,
    iterative_radiance,
    window_size_from_distance,
)


def test_anisotropic_sky_factor_respects_sky_view_in_shadow():
    beam_visibility = np.array([0.0, 1.0])
    direct_transmittance = 0.8
    cos_local_incidence = np.cos(np.deg2rad([70.0, 30.0]))
    cos_solar_zenith = np.cos(np.deg2rad(45.0))
    sky_view_factor = np.array([0.35, 0.35])

    result = _anisotropic_sky_factor(
        beam_visibility,
        direct_transmittance,
        cos_local_incidence,
        cos_solar_zenith,
        sky_view_factor,
    )

    expected_illuminated = (
        direct_transmittance
        * cos_local_incidence[1]
        / cos_solar_zenith
        + (1 - direct_transmittance) * sky_view_factor[1]
    )
    assert result[0] == pytest.approx(sky_view_factor[0])
    assert result[1] == pytest.approx(expected_illuminated)


@pytest.mark.parametrize(
    "width_m,resolution_m,expected_pixels",
    (
        (1500.0, 300.0, 5),
        (2100.0, 300.0, 7),
        (1500.0, 20.0, 75),
        (2100.0, 20.0, 105),
        (1000.0, 20.0, 51),
    ),
)
def test_window_size_is_derived_from_physical_scale(
    width_m, resolution_m, expected_pixels
):
    assert window_size_from_distance(width_m, resolution_m) == \
        expected_pixels


@pytest.mark.parametrize("width_m,resolution_m", ((0, 20), (1000, 0)))
def test_window_size_rejects_non_positive_values(width_m, resolution_m):
    with pytest.raises(ValueError, match="positive finite"):
        window_size_from_distance(width_m, resolution_m)


def test_window_mean_supports_a_window_larger_than_the_scene():
    values = np.full((9, 9), 0.6)

    result = _window_mean(values, 105, boundary="nearest")

    np.testing.assert_allclose(result, values)


def test_monochromatic_solar_irradiance_is_returned_directly():
    outputs = SimpleNamespace(values={"solar_spectrum": 1450.0})

    assert _solar_irradiance_density(outputs) == pytest.approx(1450.0)


def test_filtered_solar_irradiance_is_normalized_by_filter_integral():
    outputs = SimpleNamespace(values={
        "int_solar_spectrum": 72.5,
        "int_funct_filt": 0.05,
    })

    assert _solar_irradiance_density(outputs) == pytest.approx(1450.0)


def test_missing_solar_irradiance_reports_available_outputs():
    outputs = SimpleNamespace(values={"pixel_radiance": 1.0})

    with pytest.raises(ValueError, match="pixel_radiance"):
        _solar_irradiance_density(outputs)


class _TransparentAtmosphere:
    """Minimal RT model with direct transmission and no additive terms."""

    outputs = None

    @classmethod
    def run(cls, sza, saa, vza, vaa, wavelength, altitude, aerosol, **kwargs):
        solar_irradiance = 1000.0
        cls.outputs = SimpleNamespace(
            direct_solar_irradiance=solar_irradiance * np.cos(sza),
            diffuse_solar_irradiance=0.0,
            spherical_albedo=SimpleNamespace(total=0.0),
            atmospheric_intrinsic_radiance=0.0,
            values={"solar_spectrum": solar_irradiance},
        )


def _uniform_geometry(local_incidence_degree, shape=(9, 9)):
    global_sza = np.deg2rad(45.0)
    local_incidence = np.deg2rad(local_incidence_degree)
    zeros = np.zeros(shape, dtype=np.float64)
    angles = xr.Dataset({
        "SZA": (("x", "y"), np.full(shape, global_sza)),
        "SAA": (("x", "y"), zeros.copy()),
        "VZA": (("x", "y"), zeros.copy()),
        "VAA": (("x", "y"), zeros.copy()),
    })
    topography = xr.Dataset({
        "altitude": (("x", "y"), zeros.copy()),
        "eff_sza": (("x", "y"), np.full(shape, local_incidence)),
        "eff_vza": (("x", "y"), zeros.copy()),
        "vt": (("x", "y"), np.ones(shape)),
        "ct": (("x", "y"), zeros.copy()),
        "all_shadows": (("x", "y"), np.ones(shape)),
    })
    return angles, topography


def test_local_incidence_cosine_scales_every_wavelength_equally():
    case = Cases()
    case.create_cases(2)
    ratios = []

    for wavelength in (500.0, 1600.0, 2200.0):
        reflectances = []
        for incidence in (20.0, 70.0):
            angles, topography = _uniform_geometry(incidence)
            shape = topography["altitude"].shape
            radiance, reflectance, components = iterative_radiance(
                topography,
                angles,
                wavelength,
                hdr=np.zeros(shape),
                bhr=np.full(shape, 0.5),
                rt_model=_TransparentAtmosphere,
                rt_options={
                    "aerosol_model": "None",
                    "aod": 0.0,
                    "refl": 0.0,
                    "water": 0.0,
                    "ozone": 0.0,
                    "atmo_model": None,
                    "atcor": False,
                },
                brf=np.full(shape, 0.5),
                case=case,
                return_reflectance=True,
                return_components=True,
                verbose=False,
            )
            component_sum = sum(components.values())
            np.testing.assert_allclose(component_sum, radiance)
            reflectances.append(reflectance[4, 4])
        ratios.append(reflectances[0] / reflectances[1])

    expected_ratio = np.cos(np.deg2rad(20.0)) / np.cos(np.deg2rad(70.0))
    np.testing.assert_allclose(ratios, expected_ratio)
