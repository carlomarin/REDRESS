#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the georeferenced Sentinel-2 scene workflow."""

import numpy as np
import pytest

from redress.simulation.sentinel2_scene import (
    fill_missing_dem_cells,
    read_mtd_tl_geometry,
)


def test_small_dem_gaps_are_filled_from_nearest_valid_elevation():
    altitude = np.asarray([
        [100.0, 100.0, 100.0],
        [100.0, np.nan, 200.0],
        [200.0, 200.0, 200.0],
    ])

    filled, count, distance = fill_missing_dem_cells(
        altitude,
        pixel_size_m=20.0,
        maximum_fill_distance_m=20.0,
    )

    assert count == 1
    assert distance == pytest.approx(20.0)
    assert np.isfinite(filled).all()
    assert filled[1, 1] in (100.0, 200.0)


def test_large_dem_gap_is_rejected():
    altitude = np.ones((7, 7), dtype=float)
    altitude[1:-1, 1:-1] = np.nan

    with pytest.raises(ValueError, match="largest gap"):
        fill_missing_dem_cells(
            altitude,
            pixel_size_m=20.0,
            maximum_fill_distance_m=20.0,
        )


def test_mtd_tl_mean_geometry_uses_requested_viewing_bands(tmp_path):
    metadata = tmp_path / "MTD_TL.xml"
    metadata.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<n1:Level-1C_Tile_ID xmlns:n1="urn:sentinel2:test">
  <n1:General_Info>
    <SENSING_TIME>2026-03-28T14:37:39.123Z</SENSING_TIME>
  </n1:General_Info>
  <n1:Geometric_Info>
    <Tile_Angles>
      <Mean_Sun_Angle>
        <ZENITH_ANGLE>46.5</ZENITH_ANGLE>
        <AZIMUTH_ANGLE>12.0</AZIMUTH_ANGLE>
      </Mean_Sun_Angle>
      <Mean_Viewing_Incidence_Angle_List>
        <Mean_Viewing_Incidence_Angle bandId="1">
          <ZENITH_ANGLE>4.0</ZENITH_ANGLE>
          <AZIMUTH_ANGLE>359.0</AZIMUTH_ANGLE>
        </Mean_Viewing_Incidence_Angle>
        <Mean_Viewing_Incidence_Angle bandId="2">
          <ZENITH_ANGLE>4.0</ZENITH_ANGLE>
          <AZIMUTH_ANGLE>1.0</AZIMUTH_ANGLE>
        </Mean_Viewing_Incidence_Angle>
      </Mean_Viewing_Incidence_Angle_List>
    </Tile_Angles>
  </n1:Geometric_Info>
</n1:Level-1C_Tile_ID>
""",
        encoding="utf-8",
    )

    geometry, acquisition_time = read_mtd_tl_geometry(
        metadata, band_names=("B02", "B03")
    )

    assert acquisition_time.year == 2026
    assert acquisition_time.month == 3
    assert acquisition_time.day == 28
    assert geometry.solar_zenith_deg == pytest.approx(46.5)
    assert geometry.solar_azimuth_deg == pytest.approx(12.0)
    assert geometry.view_zenith_deg == pytest.approx(4.0, abs=1e-3)
    assert min(
        abs(geometry.view_azimuth_deg),
        abs(geometry.view_azimuth_deg - 360.0),
    ) < 1e-6


def test_mtd_tl_geometry_reports_a_missing_requested_band(tmp_path):
    metadata = tmp_path / "MTD_TL.xml"
    metadata.write_text(
        """<Tile>
  <SENSING_TIME>2026-03-28T14:37:39Z</SENSING_TIME>
  <Mean_Sun_Angle>
    <ZENITH_ANGLE>45</ZENITH_ANGLE>
    <AZIMUTH_ANGLE>180</AZIMUTH_ANGLE>
  </Mean_Sun_Angle>
  <Mean_Viewing_Incidence_Angle bandId="1">
    <ZENITH_ANGLE>4</ZENITH_ANGLE>
    <AZIMUTH_ANGLE>0</AZIMUTH_ANGLE>
  </Mean_Viewing_Incidence_Angle>
</Tile>""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="B03"):
        read_mtd_tl_geometry(metadata, band_names=("B02", "B03"))
