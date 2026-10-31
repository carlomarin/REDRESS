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
    build_atmospheric_altitude_lookup,
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


class _CoupledAtmosphere:
    """RT model exposing diffuse irradiance and atmospheric albedo."""

    outputs = None
    ground_reflectances = []

    @classmethod
    def run(cls, sza, saa, vza, vaa, wavelength, altitude, aerosol, **kwargs):
        solar_irradiance = 1000.0
        direct_transmittance = 0.8
        cls.ground_reflectances.append(kwargs["refl"])
        cls.outputs = SimpleNamespace(
            direct_solar_irradiance=(
                solar_irradiance * np.cos(sza) * direct_transmittance
            ),
            diffuse_solar_irradiance=100.0,
            spherical_albedo=SimpleNamespace(total=0.2),
            atmospheric_intrinsic_radiance=0.0,
            values={"solar_spectrum": solar_irradiance},
        )


class _AltitudeDependentAtmosphere:
    """Deterministic RT model used to test elevation interpolation."""

    outputs = None
    altitudes_m = []

    @classmethod
    def run(cls, sza, saa, vza, vaa, wavelength, altitude, aerosol, **kwargs):
        del sza, saa, vza, vaa, wavelength, aerosol, kwargs
        cls.altitudes_m.append(float(altitude))
        cls.outputs = SimpleNamespace(
            direct_solar_irradiance=500.0 + 0.05 * altitude,
            diffuse_solar_irradiance=100.0 - 0.01 * altitude,
            spherical_albedo=SimpleNamespace(
                total=0.20 - 0.00001 * altitude
            ),
            atmospheric_intrinsic_radiance=20.0 - 0.005 * altitude,
            values={"solar_spectrum": 1000.0},
        )


def _uniform_geometry(
    local_incidence_degree,
    shape=(9, 9),
    sky_view_factor=1.0,
):
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
        "vt": (("x", "y"), np.full(shape, sky_view_factor)),
        "ct": (("x", "y"), zeros.copy()),
        "all_shadows": (("x", "y"), np.ones(shape)),
    })
    return angles, topography


def test_atmospheric_lookup_interpolates_every_field_over_altitude():
    angles, topography = _uniform_geometry(45.0, shape=(3, 3))
    altitude = np.asarray([
        [1000.0, 1125.0, 1250.0],
        [1375.0, 1500.0, 1625.0],
        [1750.0, 1875.0, 2000.0],
    ])
    topography["altitude"] = (("x", "y"), altitude)
    _AltitudeDependentAtmosphere.altitudes_m = []

    lookup = build_atmospheric_altitude_lookup(
        topography,
        angles,
        rt_wavelength=550.0,
        rt_model=_AltitudeDependentAtmosphere,
        rt_options={
            "aerosol_model": "None",
            "aod": 0.0,
            "water": 0.0,
            "ozone": 0.0,
            "atmo_model": None,
            "atcor": False,
            "month": None,
            "day": None,
        },
        atmosphere_altitude_step_m=500.0,
    )

    np.testing.assert_allclose(
        lookup["altitude_nodes_m"], [1000.0, 1500.0, 2000.0]
    )
    np.testing.assert_allclose(
        lookup["forward"]["direct_solar_irradiance"],
        500.0 + 0.05 * altitude,
    )
    np.testing.assert_allclose(
        lookup["forward"]["atmospheric_intrinsic_radiance"],
        20.0 - 0.005 * altitude,
    )
    assert _AltitudeDependentAtmosphere.altitudes_m == [
        1000.0,
        1500.0,
        2000.0,
        1000.0,
        1500.0,
        2000.0,
    ]


@pytest.mark.parametrize("sky_view_factor", (1.0, 0.35))
def test_surface_atmosphere_coupling_is_sky_view_masked(sky_view_factor):
    # Keep ct at zero to isolate the sky mask from terrain reillumination.
    angles, topography = _uniform_geometry(
        45.0, sky_view_factor=sky_view_factor
    )
    shape = topography["altitude"].shape
    components = {}

    for case_number in (3, 4):
        case = Cases()
        case.create_cases(case_number)
        _CoupledAtmosphere.ground_reflectances = []
        _, _, case_components, diagnostics = iterative_radiance(
            topography,
            angles,
            wavelength=550.0,
            hdr=np.full(shape, 0.5),
            bhr=np.full(shape, 0.5),
            rt_model=_CoupledAtmosphere,
            rt_options={
                "aerosol_model": "None",
                "aod": 0.0,
                "refl": 0.9,
                "water": 0.0,
                "ozone": 0.0,
                "atmo_model": None,
                "atcor": False,
            },
            brf=np.full(shape, 0.5),
            case=case,
            return_reflectance=True,
            return_components=True,
            return_diagnostics=True,
            verbose=False,
        )
        components[case_number] = case_components
        assert _CoupledAtmosphere.ground_reflectances == [0.0, 0.0]
        assert diagnostics["terrain_reflected_irradiance"].shape == shape
        assert diagnostics["total_diffuse_irradiance"].shape == shape

    direct_flat = 1000.0 * np.cos(np.deg2rad(45.0)) * 0.8
    total_flat = direct_flat + 100.0
    coupling = total_flat * (0.5 * 0.2) / (1 - 0.5 * 0.2)
    expected_radiance_increment = (
        0.5 / np.pi * coupling * sky_view_factor * 0.8
    )
    actual_radiance_increment = (
        components[4]["diffuse_surface"][4, 4]
        - components[3]["diffuse_surface"][4, 4]
    )
    assert actual_radiance_increment == pytest.approx(
        expected_radiance_increment
    )


def test_neighbour_beam_weighting_suppresses_complete_terrain_term():
    angles, topography = _uniform_geometry(45.0)
    shape = topography["altitude"].shape
    topography["ct"] = (("x", "y"), np.full(shape, 0.4))
    visibility = np.ones(shape)
    visibility[:, :4] = 0.0
    topography["all_shadows"] = (("x", "y"), visibility)
    case = Cases()
    case.create_cases(3)
    terrain_irradiance = {}

    for weighting in (True, False):
        _, _, _, diagnostics = iterative_radiance(
            topography,
            angles,
            wavelength=550.0,
            hdr=np.full(shape, 0.5),
            bhr=np.full(shape, 0.5),
            rt_model=_CoupledAtmosphere,
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
            return_diagnostics=True,
            weight_terrain_by_neighbour_beam_visibility=weighting,
            verbose=False,
        )
        terrain_irradiance[weighting] = diagnostics[
            "terrain_reflected_irradiance"
        ][4, 4]

    assert terrain_irradiance[False] > terrain_irradiance[True]


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
