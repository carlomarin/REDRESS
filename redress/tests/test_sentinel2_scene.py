#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the georeferenced Sentinel-2 scene workflow."""

import numpy as np
import pytest

import xarray as xr

from redress.inputs.products import Dem
from redress.simulation.sentinel2_scene import (
    Sentinel2BandViewingGeometry,
    Sentinel2Geometry,
    Sentinel2TileGrid,
    _constant_angles,
    _jp2_precision,
    crop_scene,
    edge_independent_window,
    fill_missing_dem_cells,
    read_l1c_toa_reflectance,
    read_mtd_tl_geometry,
    read_mtd_tl_grid,
    read_snow_fraction_on_grid,
    reflectance_comparison_table,
    resample_dem_to_sentinel2_grid,
    write_multiband_geotiff,
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


def test_mtd_tl_geometry_retains_each_requested_viewing_band(tmp_path):
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
    assert geometry.viewing_by_band == {
        "B02": Sentinel2BandViewingGeometry(4.0, 359.0),
        "B03": Sentinel2BandViewingGeometry(4.0, 1.0),
    }
    assert geometry.viewing_for_band("B02") == pytest.approx((4.0, 359.0))
    assert geometry.viewing_for_band("B03") == pytest.approx((4.0, 1.0))
    # The cross-band vector mean remains available only as a summary.
    assert geometry.view_zenith_deg == pytest.approx(4.0, abs=1e-3)
    assert min(
        abs(geometry.view_azimuth_deg),
        abs(geometry.view_azimuth_deg - 360.0),
    ) < 1e-6


def test_constant_angles_select_the_requested_band_viewing_direction():
    dem = _grid_dem(3, 4)
    geometry = Sentinel2Geometry(
        solar_zenith_deg=45.0,
        solar_azimuth_deg=180.0,
        view_zenith_deg=0.0,
        view_azimuth_deg=0.0,
        viewing_by_band={
            "B02": Sentinel2BandViewingGeometry(3.0, 95.0),
            "B11": Sentinel2BandViewingGeometry(8.0, 105.0),
        },
    )

    blue = _constant_angles(dem, geometry, "B02")
    swir = _constant_angles(dem, geometry, "B11")

    np.testing.assert_allclose(blue["VZA"], np.deg2rad(3.0))
    np.testing.assert_allclose(blue["VAA"], np.deg2rad(95.0))
    np.testing.assert_allclose(swir["VZA"], np.deg2rad(8.0))
    np.testing.assert_allclose(swir["VAA"], np.deg2rad(105.0))


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


def _grid_dem(rows, columns, geotransform=(600000.0, 20.0, 0.0, 5200020.0, 0.0, -20.0)):
    dem = Dem()
    dem.bands["altitude"] = xr.DataArray(
        np.zeros((rows, columns)), dims=("x", "y")
    )
    dem.geotransform = geotransform
    dem.pixel_size = (geotransform[1], geotransform[5])
    dem.crs_wkt = ""
    return dem


def _scene(dem, bands=("B03", "B11")):
    rows, columns = dem.bands["altitude"].shape
    x = dem.geotransform[0] + (np.arange(columns) + 0.5) * dem.geotransform[1]
    y = dem.geotransform[3] + (np.arange(rows) + 0.5) * dem.geotransform[5]
    values = np.arange(len(bands) * rows * columns, dtype=np.float32)
    return xr.Dataset(
        {"toa_reflectance": (
            ("band", "y", "x"), values.reshape(len(bands), rows, columns)
        )},
        coords={"band": list(bands), "x": x, "y": y},
    )


def test_mtd_tl_grid_reads_the_requested_resolution(tmp_path):
    metadata = tmp_path / "MTD_TL.xml"
    metadata.write_text(
        """<Tile>
  <HORIZONTAL_CS_CODE>EPSG:32632</HORIZONTAL_CS_CODE>
  <Geoposition resolution="10"><ULX>600000</ULX><ULY>5200020</ULY></Geoposition>
  <Geoposition resolution="20"><ULX>600000</ULX><ULY>5200020</ULY></Geoposition>
</Tile>""",
        encoding="utf-8",
    )

    grid = read_mtd_tl_grid(metadata, 20.0)

    assert grid == Sentinel2TileGrid(600000.0, 5200020.0, 20.0, 32632)


def test_dem_is_averaged_onto_whole_tile_pixels(tmp_path):
    from osgeo import gdal, osr

    # A 1 m DEM whose origin is offset from the 20 m tile grid and whose
    # elevation equals its easting offset, so 20 m means are pixel centres.
    source = tmp_path / "source.tif"
    west, north = 600007.0, 5200013.0
    columns, rows = 95, 70
    raster = gdal.GetDriverByName("GTiff").Create(
        str(source), columns, rows, 1, gdal.GDT_Float32
    )
    raster.SetGeoTransform((west, 1.0, 0.0, north, 0.0, -1.0))
    reference = osr.SpatialReference()
    reference.ImportFromEPSG(32632)
    raster.SetProjection(reference.ExportToWkt())
    easting = west + np.arange(columns) + 0.5
    raster.GetRasterBand(1).WriteArray(
        np.tile(easting - 600000.0, (rows, 1)).astype(np.float32)
    )
    raster = None

    output = resample_dem_to_sentinel2_grid(
        source,
        tmp_path / "aligned.tif",
        Sentinel2TileGrid(600000.0, 5200020.0, 20.0, 32632),
    )

    dataset = gdal.Open(str(output))
    assert dataset.GetGeoTransform() == pytest.approx(
        (600020.0, 20.0, 0.0, 5200000.0, 0.0, -20.0)
    )
    # 600007-600102 by 5199943-5200013 holds whole pixels from easting
    # 600020 to 600100 and from northing 5200000 down to 5199960.
    assert (dataset.RasterXSize, dataset.RasterYSize) == (4, 2)
    np.testing.assert_allclose(
        dataset.ReadAsArray(),
        np.tile([30.0, 50.0, 70.0, 90.0], (2, 1)),
        atol=1e-4,
    )


def test_edge_independent_window_keeps_half_a_window_from_each_edge():
    dem = _grid_dem(160, 140)

    rows, columns = edge_independent_window(dem, 2100.0)

    # 2100 m is 105 pixels, so 52 pixels are lost on each side.
    assert (rows.start, rows.stop) == (52, 108)
    assert (columns.start, columns.stop) == (52, 88)
    with pytest.raises(ValueError, match="no pixel"):
        edge_independent_window(_grid_dem(104, 200), 2100.0)


def test_edge_window_margin_can_be_set_explicitly():
    dem = _grid_dem(160, 140)

    rows, columns = edge_independent_window(dem, 2100.0, margin_m=410.0)

    # 410 m rounds up to 21 pixels.
    assert (rows.start, rows.stop) == (21, 139)
    assert (columns.start, columns.stop) == (21, 119)


def test_cropped_scene_geotiff_keeps_its_own_origin(tmp_path):
    from osgeo import gdal

    dem = _grid_dem(10, 12)
    scene = crop_scene(_scene(dem), slice(3, 7), slice(2, 9))

    path = write_multiband_geotiff(scene, dem, tmp_path / "crop.tif")

    dataset = gdal.Open(str(path))
    assert dataset.GetGeoTransform() == pytest.approx(
        (600040.0, 20.0, 0.0, 5199960.0, 0.0, -20.0)
    )
    assert (dataset.RasterCount, dataset.RasterYSize, dataset.RasterXSize) == (
        2, 4, 7
    )
    assert dataset.GetRasterBand(2).GetDescription() == "B11"
    np.testing.assert_array_equal(
        dataset.ReadAsArray(), scene["toa_reflectance"].values
    )


def test_jp2_precision_is_read_from_the_siz_marker(tmp_path):
    path = tmp_path / "header.jp2"
    siz = bytearray(b"\xff\x4f\xff\x51" + bytes(38) + b"\x0e")
    path.write_bytes(b"jp2 boxes" + bytes(siz))

    assert _jp2_precision(path) == 15


def test_l1c_reflectance_is_scaled_and_averaged_onto_the_grid(tmp_path):
    from PIL import Image

    safe = tmp_path / "S2B_MSIL1C_TEST.SAFE"
    images = safe / "GRANULE" / "L1C_TEST" / "IMG_DATA"
    images.mkdir(parents=True)
    (safe / "MTD_MSIL1C.xml").write_text(
        """<Product>
  <QUANTIFICATION_VALUE unit="none">10000</QUANTIFICATION_VALUE>
  <RADIO_ADD_OFFSET band_id="2">-1000</RADIO_ADD_OFFSET>
  <RADIO_ADD_OFFSET band_id="11">-1000</RADIO_ADD_OFFSET>
</Product>""",
        encoding="utf-8",
    )
    # B03 at 10 m: each 2 by 2 block averages to 3000 + 1000 * block.
    fine = np.zeros((8, 8), dtype=np.uint16)
    for block_row in range(4):
        for block_column in range(4):
            level = 3000 + 1000 * (4 * block_row + block_column)
            fine[
                2 * block_row:2 * block_row + 2,
                2 * block_column:2 * block_column + 2,
            ] = np.array([[level - 100, level + 100], [level, level]])
    Image.fromarray(fine).save(images / "T_TEST_B03.jp2")
    coarse = (1000 + 100 * np.arange(16, dtype=np.uint16)).reshape(4, 4)
    Image.fromarray(coarse).save(images / "T_TEST_B11.jp2")

    grid = Sentinel2TileGrid(600000.0, 5200020.0, 20.0, 32632)
    x = 600000.0 + 20.0 * np.array([1.5, 2.5])
    y = 5200020.0 - 20.0 * np.array([0.5, 1.5, 2.5])

    observed = read_l1c_toa_reflectance(
        safe, grid, x, y, band_names=("B03", "B11")
    )

    blocks = (np.arange(16).reshape(4, 4) * 1000 + 2000) / 10000
    expected_b03 = blocks[0:3, 1:3]
    assert observed["toa_reflectance"].sel(band="B03").values == pytest.approx(
        expected_b03
    )
    expected_b11 = (coarse[0:3, 1:3] - 1000.0) / 10000
    assert observed["toa_reflectance"].sel(band="B11").values == pytest.approx(
        expected_b11
    )


def test_comparison_table_reports_bias_and_rmse():
    dem = _grid_dem(2, 2)
    simulated = _scene(dem, bands=("B03",))
    observed = simulated.copy(deep=True)
    observed["toa_reflectance"] = simulated["toa_reflectance"] - 0.1
    observed["toa_reflectance"][0, 0, 0] = np.nan

    table = reflectance_comparison_table(simulated, observed)

    row = table.iloc[0]
    assert row["band"] == "B03"
    assert row["pixels"] == 3
    assert row["bias"] == pytest.approx(0.1)
    assert row["rmse"] == pytest.approx(0.1)


def test_snow_fraction_is_read_on_the_simulated_grid(tmp_path):
    from osgeo import gdal, osr

    path = tmp_path / "scf.tif"
    raster = gdal.GetDriverByName("GTiff").Create(
        str(path), 4, 3, 1, gdal.GDT_Float32
    )
    # ETRS89 / UTM 32N, offset by whole pixels from the simulated grid.
    raster.SetGeoTransform((600000.0, 20.0, 0.0, 5200020.0, 0.0, -20.0))
    reference = osr.SpatialReference()
    reference.ImportFromEPSG(25832)
    raster.SetProjection(reference.ExportToWkt())
    percent = np.arange(12, dtype=np.float32).reshape(3, 4) * 10.0
    percent[0, 1] = np.nan
    raster.GetRasterBand(1).WriteArray(percent)
    raster.GetRasterBand(1).SetNoDataValue(np.nan)
    raster = None

    x = 600000.0 + 20.0 * np.array([1.5, 2.5, 4.5])
    y = 5200020.0 - 20.0 * np.array([0.5, 1.5])

    fraction = read_snow_fraction_on_grid(path, x, y)

    expected = np.array([
        [np.nan, 0.2, np.nan],
        [0.5, 0.6, np.nan],
    ])
    np.testing.assert_allclose(fraction, expected)

    with pytest.raises(ValueError, match="grid"):
        read_snow_fraction_on_grid(path, x + 7.0, y)


def test_comparison_table_uses_only_masked_pixels():
    dem = _grid_dem(2, 2)
    simulated = _scene(dem, bands=("B03",))
    observed = simulated.copy(deep=True)
    observed["toa_reflectance"] = simulated["toa_reflectance"] - 0.1
    observed["toa_reflectance"][0, 1, 1] = 9.0
    mask = np.array([[True, True], [True, False]])

    table = reflectance_comparison_table(simulated, observed, mask=mask)

    assert table.iloc[0]["pixels"] == 3
    assert table.iloc[0]["bias"] == pytest.approx(0.1)
