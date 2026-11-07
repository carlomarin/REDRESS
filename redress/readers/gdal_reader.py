#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gdal_reader.

This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
from osgeo import gdal
import xarray as xr
import numpy as np
from pyproj import CRS, Transformer
from redress.inputs.products import Dem

gdal.UseExceptions()


def open_product(inpath):
    """Open product with GDAL.

    Open a raster and return the Gdal dataset and geotransform information.

    :param inpath: Path to the raster
    :type inpath: pathlib.PosixPath
    :return: a tuple containing the Gdal dataset and geotransform
    :rtype: tuple (osgeo.gdal.Dataset, tuple)
    """
    try:
        product = gdal.Open(str(inpath), gdal.GA_ReadOnly)
    except RuntimeError as error:
        raise OSError(f"GDAL cannot read {inpath}: {error}") from error
    if product is None:
        raise OSError(f"GDAL cannot read {inpath}")
    geotransform = product.GetGeoTransform()

    return (product, geotransform)


def image_extents(product, geotransform, step=1):
    """Get the corner coordinates from the opened raster.

    Fetches a list of latitude and longitude values for the boundary
    of a given satellite product with a given pixel step along
    the border.

    :param product: the opened Gdal dataset
    :type product: osgeo.gdal.Dataset
    :param geotransform: the raster's geotransform information
    :type geotransform: tuple
    :param step: the step size in pixels
    :type step: int, optional
    :return: a list containing the boundary coordinates
    :rtype: list
    """
    # Get Dem product size
    size = product.RasterXSize, product.RasterYSize

    # Set empty extent file and retrieve lat/lon values
    ext = []
    xarr = [0, size[0]]
    yarr = [0, size[1]]

    for px in xarr:
        for py in yarr:
            x = geotransform[0] + (px * geotransform[1]) +\
                (py * geotransform[2])
            y = geotransform[3] + (px * geotransform[4]) +\
                (py * geotransform[5])
            ext.append((x, y))
        yarr.reverse()

    return ext


def reproject(product, epsg, resolution=None):
    """Reproject a product.

    Reprojects a Gdal product in memory based on a given EPSG code. The
     operation is performed in-place.

    :param epsg: the epsg code to reproject the product to
    :type epsg: int
    """
    options = {
        "dstSRS": f"EPSG:{epsg}",
        "dstNodata": np.nan,
        "outputType": gdal.GDT_Float32,
        "resampleAlg": gdal.GRA_Bilinear,
        "format": "MEM",
    }
    if resolution is not None:
        if not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be finite and greater than zero")
        options.update(xRes=resolution, yRes=resolution)
    reprojected = gdal.Warp('', product, **options)
    if reprojected is None:
        raise RuntimeError(f"GDAL failed to reproject the DEM to EPSG:{epsg}")

    return (reprojected, reprojected.GetGeoTransform())


def clip(
    product,
    geo_extent,
    native_epsg=4326,
    epsg=4326,
    buffer_distance=0.0,
):
    """Clip a product.

    Clips a Gdal product in memory to a given extent. The
     operation is performed in-place.

    :param geo_extent: the extent that the product will be clipped to
    :type geo_extent: osgeo.ogr.Geometry
    :param epsg: the epsg code of the extent
    :type epsg: int
    """
    # Get max min coords of the shape
    minX, maxX, minY, maxY = geo_extent.GetEnvelope()
    transformer = Transformer.from_crs(
        CRS.from_epsg(native_epsg),
        CRS.from_epsg(epsg),
        always_xy=True,
    )
    corners = (
        (minX, minY), (minX, maxY), (maxX, minY), (maxX, maxY),
    )
    transformed = [transformer.transform(x, y) for x, y in corners]
    x_coordinates, y_coordinates = zip(*transformed)
    xmin = min(x_coordinates) - buffer_distance
    xmax = max(x_coordinates) + buffer_distance
    ymin = min(y_coordinates) - buffer_distance
    ymax = max(y_coordinates) + buffer_distance

    clipped = gdal.Translate(
        '',
        product,
        format="MEM",
        projWin=[xmin, ymax, xmax, ymin],
    )
    if clipped is None:
        raise RuntimeError("GDAL failed to clip the DEM to the requested extent")

    return (clipped, clipped.GetGeoTransform())


def extract_band(product, geotransform):
    """Extract a band from the product.

    Extracts the first band values from the product as a
    DataArray. The units are added as an attribute.

    :return: an xarray containing the band values.
    :rtype: xarray
    """
    try:
        currentband = product.GetRasterBand(1)  # Open band
    except (RuntimeError, ValueError) as error:
        raise OSError("GDAL cannot read the first DEM band") from error

    # Extract band array
    arr = currentband.ReadAsArray().astype(np.float64)
    nodata = currentband.GetNoDataValue()
    if nodata is not None:
        arr[np.isclose(arr, nodata, equal_nan=True)] = np.nan
    scale = currentband.GetScale()
    offset = currentband.GetOffset()
    if scale is not None:
        arr *= scale
    if offset is not None:
        arr += offset
    band = xr.DataArray(arr, dims=['x', 'y'])

    # Calculate longitude array
    indices = np.indices(arr.shape)
    longitude = geotransform[1] * indices[1] + geotransform[2] *\
        indices[0] + geotransform[1] * 0.5 + geotransform[2] *\
        0.5 + geotransform[0]

    # Calculate latitude array
    latitude = geotransform[4] * indices[1] + geotransform[5] *\
        indices[0] + geotransform[4] * 0.5 + geotransform[5] *\
        0.5 + geotransform[3]

    # Set to xarray
    band.coords['lon'] = xr.DataArray(longitude, dims=['x', 'y'])
    band.coords['lat'] = xr.DataArray(latitude, dims=['x', 'y'])

    return band


def dem_generic(
    inpath,
    extent,
    epsg,
    resolution=None,
    horizon_buffer_m=0.0,
    extent_epsg=4326,
):
    """Read, project and clip a DEM for REDRESS terrain calculations.

    ``epsg`` must identify a projected CRS with metre horizontal units.
    ``extent_epsg`` identifies the coordinate reference system of ``extent``.
    The optional horizon buffer retains terrain outside the satellite
    footprint; it should be at least as large as the intended horizon search
    distance.
    """
    # Instantiate DEM class
    dem = Dem()

    # Feed the class with data
    dem.meta["path"] = inpath

    # Open the image using the imported reader
    raw_product, raw_geotrans = open_product(inpath)

    if epsg is None:
        raise ValueError(
            "A projected metric EPSG code is required for DEM processing"
        )
    target_crs = CRS.from_epsg(epsg)
    if not target_crs.is_projected:
        raise ValueError("DEM target CRS must be projected")
    unit_names = {
        axis.unit_name.lower() for axis in target_crs.axis_info
        if axis.unit_name is not None
    }
    if not unit_names.intersection({"metre", "meter"}):
        raise ValueError("DEM target CRS horizontal units must be metres")
    if not np.isfinite(horizon_buffer_m) or horizon_buffer_m < 0:
        raise ValueError("horizon_buffer_m must be finite and non-negative")

    # Reproject into a local metric CRS before calculating terrain gradients.
    reprojected_product, reprojected_geotrans = reproject(
        raw_product, epsg, resolution=resolution
    )

    # Subset the DEM to an area of interest
    clipped_product, clipped_geotrans = clip(
        reprojected_product,
        extent,
        native_epsg=extent_epsg,
        epsg=epsg,
        buffer_distance=horizon_buffer_m,
    )
    if clipped_product.RasterXSize == 0 or clipped_product.RasterYSize == 0:
        raise ValueError("The requested extent does not overlap the DEM")

    # Import the band and geotransform
    dem.bands["altitude"] = extract_band(clipped_product, clipped_geotrans)
    dem.geotransform = clipped_geotrans
    dem.pixel_size = (clipped_geotrans[1], clipped_geotrans[-1])

    return dem

