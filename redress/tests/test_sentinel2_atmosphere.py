#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Sentinel-2 L1C atmospheric auxiliary-data handling."""

import numpy as np
import pytest

from redress.readers.sentinel2_atmosphere import (
    _bilinear_sample,
    _parse_baseline,
    _read_processing_baseline,
    cams_aerosol_components,
    elevation_factor,
    geopotential_to_elevation_m,
    ozone_to_cm_atm,
    water_to_g_cm2,
)


def test_official_ecmwf_units_are_converted_for_py6s():
    assert water_to_g_cm2(10.0, "kg m-2") == pytest.approx(1.0)
    assert ozone_to_cm_atm(0.0064245, "kg m-2") == pytest.approx(0.3)
    assert geopotential_to_elevation_m(
        19613.3, "m2 s-2"
    ) == pytest.approx(2000.0)


def test_elevation_factor_uses_the_requested_scale_height():
    factor = elevation_factor(
        site_elevation_m=3000.0,
        model_elevation_m=1000.0,
        scale_height_m=2000.0,
    )
    assert factor == pytest.approx(np.exp(-1.0))


def test_cams_species_map_to_normalized_py6s_components():
    components = cams_aerosol_components({
        "black_carbon": 0.01,
        "dust": 0.04,
        "organic_matter": 0.02,
        "sea_salt": 0.02,
        "sulphate": 0.01,
    })

    assert components == pytest.approx({
        "dust": 0.4,
        "water": 0.3,
        "oceanic": 0.2,
        "soot": 0.1,
    })
    assert sum(components.values()) == pytest.approx(1.0)


def test_bilinear_sampling_uses_pixel_centres():
    array = np.asarray([[0.0, 2.0], [2.0, 4.0]])
    geotransform = (10.0, 1.0, 0.0, 50.0, 0.0, -1.0)

    sampled = _bilinear_sample(
        array,
        geotransform,
        x_coordinate=11.0,
        y_coordinate=49.0,
    )

    assert sampled == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("03.01", (3, 1)), ("04.00", (4, 0)), ("05.11", (5, 11))],
)
def test_processing_baseline_parser(text, expected):
    assert _parse_baseline(text) == expected


def test_sentinel2_product_type_and_baseline_are_read_without_namespace(
    tmp_path,
):
    metadata = tmp_path / "MTD_MSIL1C.xml"
    metadata.write_text(
        """<?xml version="1.0"?>
        <root xmlns="urn:example:sentinel2">
          <PRODUCT_TYPE>S2MSI1C</PRODUCT_TYPE>
          <PROCESSING_BASELINE>05.11</PROCESSING_BASELINE>
        </root>
        """,
        encoding="utf-8",
    )

    assert _read_processing_baseline(metadata) == "05.11"
