#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
import importlib
from datetime import date
import xarray as xr
import numpy as np
from scipy.ndimage import median_filter
from redress.topography.horizon import dozier_horizon
from redress.topography import dem_products
from redress.optics import snow
from redress.inputs.sentinel2 import sentinel2_metadata
from redress.topography.forward_model import (
    DEFAULT_ATMOSPHERE_WINDOW_WIDTH_M,
    DEFAULT_PIXEL_RESOLUTION_M,
    DEFAULT_TERRAIN_WINDOW_WIDTH_M,
    Cases,
    iterative_radiance,
)


_GRAZING_ZENITH = np.nextafter(np.pi / 2.0, 0.0)


def _surface_optics_zenith(zenith):
    """Limit angular snow optics to directions above the local horizon.

    Self-shadowed or sensor-hidden facets can have effective zenith angles
    greater than 90 degrees. Their direct radiance is removed later by the
    beam- and view-visibility masks. Evaluating their otherwise-unused snow
    optics at the grazing limit keeps the analytical functions in their
    physical domain and prevents invalid intermediate values.
    """

    return np.clip(
        np.asarray(zenith, dtype=np.float64),
        0.0,
        _GRAZING_ZENITH,
    )


class Sat(object):
    """ Satellite image class

    This class is composed of the necessary bands and metadata to run the
    forward model. To populate this class, a specific reader has to be
    specified in order to import the satellite image.
     """
    def __init__(self):

        self.meta = {"path": None, "bandlist": [], "angles": [],
                     "wavelengths": []}
        self.bands = xr.Dataset()
        self.altitude = xr.Dataset()
        self.geotransform = None
        self.pixel_size = None
        self.angles = xr.Dataset()
        self.topo_bands = xr.Dataset()


class Dem(object):
    """ DEM image class

    This class stores the DEM altitude band, the geocoding information and the
    resampled solar / viewing angles provided.
    """
    def __init__(self):
        self.meta = {"path": None, "bandlist": [],}
        self.bands = xr.Dataset()
        self.geotransform = None
        self.pixel_size = None
        self.angles = xr.Dataset()

    def compute_horizon(self, N=32, spatial_filter_size=None):
        """Compute horizon elevation and distance.

        The algorithm, based on Dozier et al. (1981) calculates the elevation
         and distance to the horizon for each pixel in a DEM, for a number of
        directions. Sirguey et al., 2009 showed although Dozier et al. 1981
        suggested that 32 directions were sufficient to obtain a decent skyview
        factor, 64 directions help obtain a more accurate representation of
        shadows. Here N results in 2N directions (N32 = 64 directions).

        Args:
            dem (self): dem object
            N (int): Number of directions to be calculated divided by 2.

        """

        if not isinstance(N, int) or N < 2:
            raise ValueError("N must be an integer greater than or equal to 2")

        # Test if the pixels are square
        if not np.isclose(
            abs(self.pixel_size[0]), abs(self.pixel_size[1]), rtol=1e-6
        ):
            raise RuntimeError("Error: Horizon algorithm only works with"
                               " square pixels!")

        if "altitude" not in self.bands:
            raise ValueError("DEM dataset must contain an 'altitude' variable")

        altitude = np.asarray(
            self.bands["altitude"].data, dtype=np.float64
        )
        if not np.all(np.isfinite(altitude)):
            missing = np.count_nonzero(~np.isfinite(altitude))
            raise ValueError(
                f"DEM contains {missing} missing elevation cells. Fill the "
                "buffered DEM before horizon calculation; replacing missing "
                "terrain with zero creates false horizons."
            )

        # Pre-allocate elevation and distance rasters. These rasters will
        # contain the angle in degrees to the horizon and the distance in
        # meters for each pixel. Each slice in the 3rd dimension represents
        # a direction.
        eh = np.full((2 * N, altitude.shape[0],
                      altitude.shape[1]), np.nan,
                     dtype=np.float32,
                     )
        dh = np.full((2 * N, altitude.shape[0],
                      altitude.shape[1]), np.nan,
                     dtype=np.float32,
                     )

        # Run the horizon algorithm for each direction
        for ii in range(1, N + 1, 1):
            phi = (ii - 1) * np.pi / N
            ehtemp, dhtemp = dozier_horizon(altitude,
                                            abs(self.pixel_size[0]), phi)
            dh[ii - 1, :, :] = dhtemp[0, :, :]
            eh[ii - 1, :, :] = ehtemp[0, :, :]
            dh[ii - 1 + N, :, :] = dhtemp[1, :, :]
            eh[ii - 1 + N, :, :] = ehtemp[1, :, :]

        if spatial_filter_size is not None:
            if (
                not isinstance(spatial_filter_size, int)
                or spatial_filter_size < 1
                or spatial_filter_size % 2 == 0
            ):
                raise ValueError(
                    "spatial_filter_size must be a positive odd integer"
                )
            filter_shape = (1, spatial_filter_size, spatial_filter_size)
            eh = median_filter(eh, size=filter_shape)
            dh = median_filter(dh, size=filter_shape)

        # Convert numpy arrays to xarray
        horizon_azimuth = np.mod(
            np.pi - np.arange(2 * N) * np.pi / N,
            2.0 * np.pi,
        )
        self.bands["horizon_ele"] = xr.DataArray(
            eh,
            dims=['phi', 'x', 'y'],
            coords={"phi": horizon_azimuth},
            attrs={"units": "degree", "long_name": "terrain horizon elevation"},
        )
        self.bands["horizon_dist"] = xr.DataArray(
            dh,
            dims=['phi', 'x', 'y'],
            coords={"phi": horizon_azimuth},
            attrs={"units": "m", "long_name": "distance to terrain horizon"},
        )
        self.bands.coords["phi"].attrs.update(
            units="radian",
            long_name="terrain horizon compass azimuth",
        )

    def compute_topo_bands(
        self,
        self_shadow_cosine_threshold=0.0,
        clean_cast_shadows=False,
        terrain_configuration_method="hemispheric_closure",
    ):
        """Calculate all topographic products from DEM

        Calculates all the topographic products based on the input DEM and the
        derived products.

        Parameters
        ----------
        self_shadow_cosine_threshold : float
            Minimum cosine of local solar incidence considered illuminated.
            The physical default is zero. The original REDRESS empirical
            value can be reproduced with 0.035.
        clean_cast_shadows : bool
            Apply a 3 by 3 binary closing operation to cast visibility.
            Disabled by default because this changes the physical mask.
        terrain_configuration_method : {"hemispheric_closure",
            "dozier_frew", "redress_bowl"}
            Definition used for the terrain configuration factor. The default
            closes the outward hemisphere with Ct = 1 - Vd. ``redress_bowl``
            is a backward-compatible alias for the same convention.
        """

        # Temporarily store products in a dictionnary
        topo_prods = {}

        # Check if horizon exists
        if "horizon_ele" not in self.bands.keys():
            raise ValueError("Please calculate or load the horizon product"
                             " before computing other topographic products!")

        # Compute aspect and slope of DEM
        print("Calculating Slope and Aspect from DEM\n")
        topo_prods["slope"], topo_prods["aspect"] = dem_products.horneslope(
            self.bands["altitude"].data, self.pixel_size)

        # Calculate effective sza
        print("Calculating Effective Solar and Viewing Angles\n")
        topo_prods["eff_sza"] = dem_products.effective_zenith_angle(
            self.angles["SZA"],
            self.angles["SAA"],
            topo_prods["slope"], topo_prods["aspect"])

        # Calculate effective oza
        topo_prods["eff_vza"] = dem_products.effective_zenith_angle(
            self.angles["VZA"], self.angles["VAA"],
            topo_prods["slope"], topo_prods["aspect"])
        topo_prods["local_relative_azimuth"] = (
            dem_products.local_relative_azimuth(
                self.angles["SZA"].data,
                self.angles["SAA"].data,
                self.angles["VZA"].data,
                self.angles["VAA"].data,
                topo_prods["slope"],
                topo_prods["aspect"],
            )
        )
        topo_prods["local_art_relative_azimuth"] = (
            dem_products.art_relative_azimuth(
                topo_prods["local_relative_azimuth"]
            )
        )

        # Compute skyview and terrain configuration factors
        print("Calculating Skyview Factor\n")
        topo_prods["vt"], topo_prods["ct"] = dem_products.skyview(
            self.bands["horizon_ele"].data,
            topo_prods["slope"],
            topo_prods["aspect"],
            terrain_configuration_method=terrain_configuration_method,
        )

        # Calculate shadow product
        print("Calculating pixels based shadows\n")
        topo_prods["all_shadows"], topo_prods["self_shadows"],\
            topo_prods["cast_shadows"] = dem_products.shadows(
                self.bands["horizon_ele"].data,
                topo_prods["slope"], topo_prods["aspect"],
                self.angles["SZA"].data,
                topo_prods["eff_sza"],
                self.angles["SAA"].data,
                min_cosine=self_shadow_cosine_threshold,
                clean_cast_shadows=clean_cast_shadows,
            )
        topo_prods["beam_visibility"] = topo_prods["all_shadows"]

        # A pixel must face the sensor and lie above the terrain horizon in
        # the viewing direction. This is the binary function 8 in the paper.
        topo_prods["view_ground"] = dem_products.terrain_visibility(
            self.bands["horizon_ele"].data,
            self.angles["VZA"].data,
            self.angles["VAA"].data,
            topo_prods["eff_vza"],
            min_cosine=0.0,
        )

        # The 3-by-3 slope stencil is undefined at the outermost DEM cells.
        # Preserve these cells as missing rather than assigning physical zero.
        spatial_coords = {
            name: coordinate
            for name, coordinate in self.bands.coords.items()
            if set(coordinate.dims).issubset({"x", "y"})
        }
        for key, value in topo_prods.items():
            padded_value = np.pad(value[1:-1, 1:-1], ((1, 1), (1, 1)),
                                  mode="constant", constant_values=np.nan)
            self.bands[key] = xr.DataArray(padded_value,
                                           dims=['x', 'y'],
                                           coords=spatial_coords)

        units = {
            "slope": "radian",
            "aspect": "radian",
            "eff_sza": "radian",
            "eff_vza": "radian",
            "local_relative_azimuth": "radian",
            "local_art_relative_azimuth": "radian",
            "vt": "1",
            "ct": "1",
            "all_shadows": "1",
            "self_shadows": "1",
            "cast_shadows": "1",
            "beam_visibility": "1",
            "view_ground": "1",
        }
        for name, unit in units.items():
            self.bands[name].attrs["units"] = unit
        self.bands["ct"].attrs["method"] = terrain_configuration_method
        self.bands["local_relative_azimuth"].attrs["long_name"] = (
            "geometric local solar-view azimuth separation; zero is equal "
            "azimuth"
        )
        self.bands["local_art_relative_azimuth"].attrs["long_name"] = (
            "local ART relative azimuth; zero is forward scattering and pi "
            "is backscatter"
        )

        print("Done\n")


class Model(object):
    """ Model output class

    This class is composed of the necessary bands and metadata to run the
    forward model. To populate this class, a specific reader has to be
    specified in order to import the satellite image.
     """
    def __init__(self):

        self.ssa = None
        self.snow_impurities = {}
        self.lap = {
            "black_carbon_ng_g": 0.0,
            "dust_ug_g": 0.0,
            "dust_density_kg_m3": snow.DEFAULT_DUST_DENSITY_KG_M3,
            "dust_model": "muller2011",
        }
        self.angles = None
        self.dem = None
        self.meta = None
        self.rt_model = None
        self.rt_options = {}
        self.difftot = []
        self.brf = []
        self.albedo = {
            "total": [],
            "direct": [],
            "diffuse": [],
            "hdr": [],
        }
        self.case = Cases()
        self.toa_rad = xr.Dataset()
        self.toa_reflectance = xr.Dataset()
        self.toa_components = {}
        self.geotransform = None
    # TODO: make a copy attributes function in class to copy attributes from other class
    def import_rt_model(self, rt_name):
        """ Import a model"""
        # Try to import the model in the rt_models folder.
        # Return the reader class in the module.
        try:
            module = importlib.import_module(
                "redress.rtmodel.%s" % rt_name)
        except ImportError as e:
            raise IOError("Unable to find the rt model %s. %s." % (rt_name, e))

        self.rt_model = getattr(module, "rtmodel")

    def set_sentinel2(self, platform, band_names=None):
        """Configure the model for Sentinel-2 MSI spectral bands.

        Parameters
        ----------
        platform : str
            Sentinel-2 platform identifier, currently ``S2A`` or ``S2B``.
        band_names : iterable of str, optional
            MSI bands to simulate. All 13 bands are used when omitted.
        """

        self.meta = sentinel2_metadata(platform, band_names)

    def set_lap(
        self,
        black_carbon_ng_g=0.0,
        dust_ug_g=0.0,
        dust_density_kg_m3=snow.DEFAULT_DUST_DENSITY_KG_M3,
        dust_model="muller2011",
    ):
        """Set light-absorbing particles in the simulated snow.

        Concentrations are specified as mass of impurity per mass of snow.
        The Snowoptics representation used by REDRESS permits black carbon,
        mineral dust, or their simultaneous presence.

        Parameters
        ----------
        black_carbon_ng_g : float
            Black-carbon concentration in ng g-1.
        dust_ug_g : float
            Mineral-dust concentration in microgram g-1.
        dust_density_kg_m3 : float
            Mineral-dust bulk density in kg m-3.
        dust_model : {"muller2011", "skiles2014"}
            Dust optical formulation implemented by Snowoptics.
        """

        self.snow_impurities = snow.lap_impurities(
            black_carbon_ng_g=black_carbon_ng_g,
            dust_ug_g=dust_ug_g,
            dust_density_kg_m3=dust_density_kg_m3,
            dust_model=dust_model,
        )
        self.lap = {
            "black_carbon_ng_g": float(black_carbon_ng_g),
            "dust_ug_g": float(dust_ug_g),
            "dust_density_kg_m3": float(dust_density_kg_m3),
            "dust_model": dust_model,
        }

    def set_rt_options(self, aero, aod=0.1, refl=0.99, water=0.05,
                       ozone=0.3, atmos=None, atcor=False,
                       month=None, day=None):
        """Set the atmopheric conditions as entry options for the RT Model."""

        if (month is None) != (day is None):
            raise ValueError("month and day must be provided together")
        if month is not None:
            date(2000, int(month), int(day))

        self.rt_options.update({"aerosol_model": aero,
                                "aod": aod, "refl": refl, "water": water,
                                "ozone": ozone, "atmo_model": atmos,
                                "atcor": atcor,
                                "month": month, "day": day})

    def compute_albedo(self):
        if self.ssa is None:
            raise ValueError("The ssa of snow hasn't been set!")

        if not self.rt_model:
            raise ValueError("No RT model specified!")

        if not self.rt_options:
            raise ValueError("The input options to run the RT model haven't"
                             " been set.")

        self.difftot = []
        self.albedo = {
            "total": [],
            "direct": [],
            "diffuse": [],
            "hdr": [],
        }
        band_definitions = self.meta.get("band_definitions")

        if band_definitions is not None:
            if len(band_definitions) != len(self.meta["wavelengths"]):
                raise ValueError(
                    "band_definitions and wavelengths must have equal length"
                )
        else:
            band_definitions = [None] * len(self.meta["wavelengths"])

        # Calculate the direct and diffuse illumination ratio using the mean
        # geometry and elevation of the scene.
        for wvl, band_definition in zip(self.meta["wavelengths"],
                                        band_definitions):
            rt = self.rt_model()
            rt_wavelength = (band_definition.py6s_filter
                             if band_definition is not None else wvl)
            # Global satellite geometry is used here only for the 6S
            # atmospheric irradiance calculation. The snow DHR and HDR below
            # use eff_sza and eff_vza, respectively.
            rt.run(self.angles["SZA"].data.mean(),
                   self.angles["SAA"].data.mean(),
                   self.angles["VZA"].data.mean(),
                   self.angles["VAA"].data.mean(),
                   rt_wavelength,
                   np.nanmean(self.topo_bands["altitude"].data),
                   self.rt_options["aerosol_model"],
                   aod=self.rt_options["aod"],
                   # Only the atmosphere-only direct/diffuse ratio is used.
                   # REDRESS calculates surface coupling explicitly.
                   refl=0.0,
                   water=self.rt_options["water"],
                   ozone=self.rt_options["ozone"],
                   atmo=self.rt_options["atmo_model"],
                   atcor=self.rt_options["atcor"],
                   month=self.rt_options.get("month"),
                   day=self.rt_options.get("day"),
                   )
            dir_illum = rt.outputs.direct_solar_irradiance
            diff_illum = rt.outputs.diffuse_solar_irradiance

            diffuse_total_ratio = 1 / (1 + dir_illum / diff_illum)

            self.difftot.append(diffuse_total_ratio)

            local_incidence = _surface_optics_zenith(
                self.topo_bands["eff_sza"].data
            )
            local_view_zenith = _surface_optics_zenith(
                self.topo_bands["eff_vza"].data
            )

            # The direct albedo is the DHR at local incidence. Equation 5 of
            # Lamare et al. (2020) instead requires the pixel HDR at the local
            # viewing angle. BRDF reciprocity permits calculation of that HDR
            # with the same directional-albedo function evaluated at eff_vza.
            if band_definition is None:
                alb_dir, alb_diff, alb_tot = snow.albedo_kokhanovsky(
                    wvl,
                    local_incidence,
                    diffuse_total_ratio,
                    self.ssa,
                    impurities=self.snow_impurities,
                    B=1.6, g=0.845)
                hdr = snow.directional_albedo_kokhanovsky(
                    wvl,
                    local_view_zenith,
                    self.ssa,
                    impurities=self.snow_impurities,
                    B=1.6,
                    g=0.845,
                )
            else:
                spectral_albedo = [
                    snow.albedo_kokhanovsky(
                        spectral_wavelength,
                        local_incidence,
                        diffuse_total_ratio,
                        self.ssa,
                        impurities=self.snow_impurities,
                        B=1.6,
                        g=0.845,
                    )
                    for spectral_wavelength in band_definition.wavelength_nm
                ]
                spectral_hdr = [
                    snow.directional_albedo_kokhanovsky(
                        spectral_wavelength,
                        local_view_zenith,
                        self.ssa,
                        impurities=self.snow_impurities,
                        B=1.6,
                        g=0.845,
                    )
                    for spectral_wavelength in band_definition.wavelength_nm
                ]
                alb_dir = band_definition.integrate_spectral_values(
                    np.stack([value[0] for value in spectral_albedo]), axis=0
                )
                alb_diff = band_definition.integrate_spectral_values(
                    np.stack([value[1] for value in spectral_albedo]), axis=0
                )
                alb_tot = band_definition.integrate_spectral_values(
                    np.stack([value[2] for value in spectral_albedo]), axis=0
                )
                hdr = band_definition.integrate_spectral_values(
                    np.stack(spectral_hdr), axis=0
                )

            self.albedo["direct"].append(alb_dir)
            self.albedo["diffuse"].append(alb_diff)
            self.albedo["total"].append(alb_tot)
            self.albedo["hdr"].append(hdr)

    def compute_brf(self):
        self.brf = []
        band_definitions = self.meta.get("band_definitions")
        if band_definitions is None:
            band_definitions = [None] * len(self.meta["wavelengths"])
        if "local_art_relative_azimuth" in self.topo_bands:
            relative_azimuth = self.topo_bands[
                "local_art_relative_azimuth"
            ]
        else:
            geometric_relative_azimuth = self.topo_bands.get(
                "local_relative_azimuth",
                self.angles["SAA"] - self.angles["VAA"],
            )
            relative_azimuth = dem_products.art_relative_azimuth(
                geometric_relative_azimuth
            )

        illumination_zenith = _surface_optics_zenith(
            self.topo_bands["eff_sza"]
        )
        viewing_zenith = _surface_optics_zenith(
            self.topo_bands["eff_vza"]
        )

        # Calculate BRF for the local geometry, wavelength and snow SSA.
        for wvl, band_definition in zip(self.meta["wavelengths"],
                                        band_definitions):
            if band_definition is None:
                band_brf = snow.brf_kokhanovsky(
                    illumination_zenith,
                    viewing_zenith,
                    relative_azimuth,
                    wvl,
                    self.ssa,
                    impurities=self.snow_impurities,
                )
            else:
                spectral_brf = [
                    snow.brf_kokhanovsky(
                        illumination_zenith,
                        viewing_zenith,
                        relative_azimuth,
                        spectral_wavelength,
                        self.ssa,
                        impurities=self.snow_impurities,
                    )
                    for spectral_wavelength in band_definition.wavelength_nm
                ]
                band_brf = band_definition.integrate_spectral_values(
                    np.stack(spectral_brf), axis=0
                )

            self.brf.append((wvl, band_brf))

    def simulate_TOA_radiance(
        self,
        casenum,
        verbose=True,
        pixel_resolution_m=DEFAULT_PIXEL_RESOLUTION_M,
        terrain_window_width_m=DEFAULT_TERRAIN_WINDOW_WIDTH_M,
        atmosphere_window_width_m=DEFAULT_ATMOSPHERE_WINDOW_WIDTH_M,
    ):
        """Simulate TOA radiance using physically scaled neighbourhoods."""

        self.toa_rad = xr.Dataset()
        self.toa_reflectance = xr.Dataset()
        component_names = (
            "direct_surface",
            "diffuse_surface",
            "neighbour_atmosphere",
            "atmospheric_path",
        )
        self.toa_components = {
            name: xr.Dataset() for name in component_names
        }

        # Compute albedo
        self.compute_albedo()

        # Compute BRF
        self.compute_brf()

        case = Cases()
        case.create_cases(casenum)

        band_definitions = self.meta.get("band_definitions")
        if band_definitions is None:
            band_definitions = [None] * len(self.meta["bandlist"])

        for band, wvl, definition, hdr, bhr, brf in zip(
                self.meta["bandlist"],
                self.meta["wavelengths"],
                band_definitions,
                self.albedo["hdr"],
                self.albedo["diffuse"],
                self.brf):

            rt_wavelength = (definition.py6s_filter
                             if definition is not None else wvl)

            result = iterative_radiance(
                self.topo_bands,
                self.angles,
                wvl,
                hdr,
                bhr,
                self.rt_model(),
                self.rt_options,
                np.asarray(brf[1]),
                case,
                dif_anis=False,
                rt_wavelength=rt_wavelength,
                return_reflectance=True,
                return_components=True,
                verbose=verbose,
                pixel_resolution_m=pixel_resolution_m,
                terrain_window_width_m=terrain_window_width_m,
                atmosphere_window_width_m=atmosphere_window_width_m,
            )
            (synthetic_toa_radiance,
             synthetic_toa_reflectance,
             radiance_components) = result
            self.toa_rad[band] = xr.DataArray(synthetic_toa_radiance,
                                              dims=['x', 'y'],
                                              coords=self.topo_bands.coords)
            self.toa_reflectance[band] = xr.DataArray(
                synthetic_toa_reflectance,
                dims=['x', 'y'],
                coords=self.topo_bands.coords,
            )
            for component_name, component_values in (
                radiance_components.items()
            ):
                self.toa_components[component_name][band] = xr.DataArray(
                    component_values,
                    dims=['x', 'y'],
                    coords=self.topo_bands.coords,
                )

