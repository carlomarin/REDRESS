#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for REDRESS forward-model utilities."""

from types import SimpleNamespace

import pytest

from redress.topography.forward_model import _solar_irradiance_density


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
