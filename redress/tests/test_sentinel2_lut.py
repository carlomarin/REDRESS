#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for Sentinel-2 snow look-up table generation."""

import numpy as np
import xarray as xr

from redress.lut import (
    Sentinel2EndmemberScenario,
    create_sentinel2_snow_brf_lut,
    create_sentinel2_toa_lut,
)


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


def test_toa_lut_preserves_scenario_parameters(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.meta = None

        def set_sentinel2(self, platform, band_names):
            self.meta = {
                "platform": platform,
                "bandlist": list(band_names),
                "wavelengths": [500.0, 1600.0],
                "native_resolution_m": [10, 20],
            }

        def import_rt_model(self, name):
            self.rt_model_name = name

        def set_rt_options(self, **kwargs):
            self.rt_options = kwargs

        def simulate_TOA_radiance(self, case_number, verbose=True):
            shape = self.topo_bands["altitude"].shape
            bands = self.meta["bandlist"]
            self.toa_rad = xr.Dataset({
                band: (("x", "y"), np.full(shape, index + 1.0))
                for index, band in enumerate(bands)
            })
            self.toa_reflectance = xr.Dataset({
                band: (("x", "y"), np.full(shape, 0.1 * (index + 1)))
                for index, band in enumerate(bands)
            })
            self.brf = [
                (wavelength, np.full(shape, 0.2 * (index + 1)))
                for index, wavelength in enumerate(self.meta["wavelengths"])
            ]
            self.albedo = {
                "direct": [np.full(shape, 0.3), np.full(shape, 0.4)],
                "diffuse": [np.full(shape, 0.5), np.full(shape, 0.6)],
            }

    monkeypatch.setattr("redress.lut.sentinel2.Model", FakeModel)

    scenarios = (
        Sentinel2EndmemberScenario(name="baseline", category="Baseline"),
        Sentinel2EndmemberScenario(
            name="high_aod", category="Atmosphere", aod550=0.3
        ),
    )
    lut = create_sentinel2_toa_lut(
        platform="S2B",
        band_names=("B02", "B11"),
        scenarios=scenarios,
        month=3,
        day=28,
    )

    assert lut["toa_reflectance"].shape == (2, 2)
    assert lut["category"].values.tolist() == ["Baseline", "Atmosphere"]
    assert lut["aod550"].values.tolist() == [0.1, 0.3]
    np.testing.assert_allclose(lut["toa_reflectance"][0], [0.1, 0.2])
