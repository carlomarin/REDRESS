#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
import numpy as np
from scipy.ndimage import uniform_filter


DEFAULT_TERRAIN_WINDOW_WIDTH_M = 1500.0
DEFAULT_ATMOSPHERE_WINDOW_WIDTH_M = 2100.0
DEFAULT_PIXEL_RESOLUTION_M = 300.0


def window_size_from_distance(window_width_m, pixel_resolution_m):
    """Convert a physical moving-window width to an odd pixel count.

    The returned window covers at least ``window_width_m`` and always has a
    central pixel. At the original 300 m OLCI resolution, the REDRESS widths
    of 1.5 and 2.1 km therefore reproduce the legacy 5- and 7-pixel windows.
    At 20 m Sentinel-2 resolution, they become 75 and 105 pixels.
    """

    window_width_m = float(window_width_m)
    pixel_resolution_m = float(pixel_resolution_m)
    if not np.isfinite(window_width_m) or window_width_m <= 0:
        raise ValueError("window_width_m must be a positive finite value")
    if not np.isfinite(pixel_resolution_m) or pixel_resolution_m <= 0:
        raise ValueError(
            "pixel_resolution_m must be a positive finite value"
        )

    window_size = int(np.ceil(window_width_m / pixel_resolution_m))
    if window_size % 2 == 0:
        window_size += 1
    return window_size


def _validate_window_size(window_size, name):
    """Validate an explicit moving-window size in pixels."""

    if isinstance(window_size, bool) or int(window_size) != window_size:
        raise ValueError(f"{name} must be an odd positive integer")
    window_size = int(window_size)
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError(f"{name} must be an odd positive integer")
    return window_size


def _window_mean(values, window_size, boundary="nearest"):
    """Calculate an efficient NaN-aware square moving-window mean."""

    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    filtered_values = uniform_filter(
        np.where(finite, values, 0.0),
        size=window_size,
        mode=boundary,
        cval=0.0,
    )
    filtered_weights = uniform_filter(
        finite.astype(np.float64),
        size=window_size,
        mode=boundary,
        cval=0.0,
    )
    return np.divide(
        filtered_values,
        filtered_weights,
        out=np.full_like(filtered_values, np.nan),
        where=filtered_weights > 0,
    )


def _anisotropic_sky_factor(
    beam_visibility,
    direct_transmittance,
    cos_local_incidence,
    cos_solar_zenith,
    sky_view_factor,
):
    """Return the Hay anisotropic diffuse-sky scaling factor.

    The circumsolar component follows the local incidence angle. The
    remaining isotropic component is reduced by the fraction of sky visible
    from the pixel. ``beam_visibility`` ranges from one for a fully
    illuminated pixel to zero for a fully shaded pixel.
    """

    anisotropy_index = beam_visibility * direct_transmittance
    circumsolar = anisotropy_index * (
        cos_local_incidence / cos_solar_zenith
    )
    isotropic = (1 - anisotropy_index) * sky_view_factor
    return circumsolar + isotropic


def _solar_irradiance_density(outputs):
    """Return Py6S extraterrestrial solar irradiance in W m-2 um-1.

    Monochromatic 6S runs expose ``solar_spectrum`` directly. Filtered sensor
    runs instead expose the solar-spectrum integral and the filter integral;
    their ratio is the response-weighted irradiance density.
    """

    values = outputs.values
    if "solar_spectrum" in values:
        solar_irradiance = values["solar_spectrum"]
    elif {"int_solar_spectrum", "int_funct_filt"} <= values.keys():
        filter_integral = values["int_funct_filt"]
        if filter_integral <= 0:
            raise ValueError("Py6S returned a non-positive filter integral")
        solar_irradiance = values["int_solar_spectrum"] / filter_integral
    else:
        available = ", ".join(sorted(values))
        raise ValueError(
            "Py6S did not return the solar irradiance required by REDRESS. "
            f"Available outputs: {available}"
        )

    if not np.isfinite(solar_irradiance) or solar_irradiance <= 0:
        raise ValueError("Py6S returned invalid extraterrestrial irradiance")
    return float(solar_irradiance)


class Cases:
    """Terrain configuration class.

    The class allows the user to set a specific terrain configuration to run
     the forward iterative model. The configurations are divided into 5
     scenarii that set different options (see the class method for more
     information).

    Attributes:
        flat_terrain (bool): Switch for flat terrain.
        atm2sensor (bool): Switch for the atmospheric intrinsic radiance.
        terrain_contribution (bool): Switch for terrain reillumination.
        atm_coupl (bool): Switch for the coupled atmospheric neighbor effects.

    """
    def __init__(self):
        self.flat_terrain = bool
        self.atm2sensor = bool
        self.terrain_contribution = bool
        self.atm_coupl = bool

    def create_cases(self, case_number):
        """Set the terrain configuration.

        The five possible terrain configurations allow to configure the boolean
         switches which populate the class. Case 1: consider a flat terrain
         only (no slopes, terrain effects, or atmosphere). Case 2: consider
         slopes only (no terrain effects or atmosphere). Case 3: consider
         slopes and terrain effects (reillumination), but no coupling with the
         atmosphere. Case 4: consider slopes, terrain effects and coupling with
         the atmosphere but no atmospheric intrinsic radiance. Case 5: full
         rugged terrain effects (slopes, reillumination, atmospheric coupling,
         atmosphere intrinsic radiance).

        Args:
            case_number: The first parameter.

        Raises:
            ValueError: If 'case_number' is not between 1 and 5.
        """
        if case_number == 1:
            # Case 1: flat terrain, no terrain contribution
            self.flat_terrain = True
            self.atm2sensor = False
            self.terrain_contribution = False
            self.atm_coupl = False
        elif case_number == 2:
            # Case 3: tilted terrain, no terrain contribution
            self.flat_terrain = False
            self.atm2sensor = False
            self.terrain_contribution = False
            self.atm_coupl = False
        elif case_number == 3:
            # Case 3: tilted terrain, no atmospheric coupling
            self.flat_terrain = False
            self.atm2sensor = False
            self.terrain_contribution = True
            self.atm_coupl = False
        elif case_number == 4:
            # Case : tilted terrain, coupling and terrain reflected
            self.flat_terrain = False
            self.atm2sensor = False
            self.terrain_contribution = True
            self.atm_coupl = True
        elif case_number == 5:
            # Case 5: tilted terrain, coupling and terrain reflected, and
            # atmosphere intrinsic radiance
            self.flat_terrain = False
            self.atm2sensor = True
            self.terrain_contribution = True
            self.atm_coupl = True
        else:
            raise ValueError("Case number not available")


def iterative_radiance(
    topo_bands,
    angles,
    wavelength,
    hdr,
    bhr,
    rt_model,
    rt_options,
    brf,
    case,
    tw=None,
    aw=None,
    dif_anis=False,
    rt_wavelength=None,
    return_reflectance=False,
    return_components=False,
    max_iterations=100,
    verbose=True,
    pixel_resolution_m=DEFAULT_PIXEL_RESOLUTION_M,
    terrain_window_width_m=DEFAULT_TERRAIN_WINDOW_WIDTH_M,
    atmosphere_window_width_m=DEFAULT_ATMOSPHERE_WINDOW_WIDTH_M,
):
    """Iterative solver for TOA radiance.

    Neighbourhood windows are derived from their physical widths and the
    raster resolution. The ``tw`` and ``aw`` arguments remain available as
    explicit pixel-count overrides for reproducing earlier calculations.
    """

    if tw is None:
        tw = window_size_from_distance(
            terrain_window_width_m, pixel_resolution_m
        )
    else:
        tw = _validate_window_size(tw, "tw")
    if aw is None:
        aw = window_size_from_distance(
            atmosphere_window_width_m, pixel_resolution_m
        )
    else:
        aw = _validate_window_size(aw, "aw")

    # Calculate the cosine of the SZA and effective SZA
    cos_sza = np.cos(angles["SZA"].data)
    cos_sza_eff = np.cos(topo_bands["eff_sza"].data)
    # Set negative values to 0
    cos_sza_eff = np.where(cos_sza_eff < 0, 0, cos_sza_eff)

    # Calculate the cosine of effective viewing angle
    cos_vza = np.cos(angles["VZA"].data)
    cos_vza_eff = np.cos(topo_bands["eff_vza"].data)

    # Calculate the pixels viewed by the satellite (not hidden)
    view_ground = np.where(cos_vza_eff <= 0, 0, cos_vza_eff)
    # Set - values to 0
    view_ground = np.where(view_ground > 0, 1, view_ground)  # >0 = 1

    # Use convolution to obtain the mean terrain configuration factor
    ct_nbh = _window_mean(
        topo_bands["ct"].data, tw, boundary="constant"
    )
    # Shadows, in this product bb = 0 if a shadow, 1 if no shadow
    bb = topo_bands["all_shadows"]

    # Shadows in the neighborhood
    bb_nbh = _window_mean(bb, tw, boundary="constant")

    # Run the iterative reflectance for all bands
    if verbose:
        print("Running complex terrain reflectance calculation"
              " for band: %s nm" % wavelength)

    # Initialise environmental reflectance, terrain reflected irradiance
    # iterative reflectance, and iterative radiance.
    # The bi-hemispherical reflectance (spherical albedo) calculated from
    # in-situ SSA measurements is used to initialize both the
    # environmental and terrain reflectance.
    # The environemental reflectance is averaged over 3X3 pixels (diam =
    # 900 m) and terrain reflectance over 7x7 pixels (diam = 2.1 km)
    re = _window_mean(bhr, aw, boundary="nearest")
    re = np.expand_dims(re, axis=2)

    rt = _window_mean(bhr, tw, boundary="nearest")
    rt = np.expand_dims(rt, axis=2)

    r = np.zeros_like(bhr)
    r = np.expand_dims(r, axis=2)

    l_toa = np.zeros_like(bhr)
    l_toa = np.expand_dims(l_toa, axis=2)

    # The scalar wavelength is used for diagnostics and backward
    # compatibility. A sensor response can be supplied separately for Py6S.
    if rt_wavelength is None:
        rt_wavelength = wavelength

    # Call the radiative transfer class
    rt_model.run(angles["SZA"].data.mean(),
                 angles["SAA"].data.mean(),
                 angles["VZA"].data.mean(),
                 angles["VAA"].data.mean(),
                 rt_wavelength,
                 np.nanmean(topo_bands["altitude"].data),
                 rt_options["aerosol_model"],
                 aod=rt_options["aod"],
                 refl=bhr.mean(),
                 water=rt_options["water"],
                 ozone=rt_options["ozone"],
                 atmo=rt_options["atmo_model"],
                 atcor=rt_options["atcor"],
                 month=rt_options.get("month"),
                 day=rt_options.get("day"),
                 )

    atmospheric_data = rt_model.outputs

    rt_model.run(angles["VZA"].data.mean(),
                 angles["VAA"].data.mean(),
                 angles["SZA"].data.mean(),
                 angles["SAA"].data.mean(),
                 rt_wavelength,
                 np.nanmean(topo_bands["altitude"].data),
                 rt_options["aerosol_model"],
                 aod=rt_options["aod"],
                 refl=bhr.mean(),
                 water=rt_options["water"],
                 ozone=rt_options["ozone"],
                 atmo=rt_options["atmo_model"],
                 atcor=rt_options["atcor"],
                 month=rt_options.get("month"),
                 day=rt_options.get("day"),
                 )
    atmospheric_inv_data = rt_model.outputs

    # Get the variables of interest from the 6s run
    # Downward direct solar flux attenuated by the atmosphere
    # e_dir = mu_s * E_s * e{-tau/mu_s}
    EdP_flat = atmospheric_data.direct_solar_irradiance

    # SOlar flux
    Eo = _solar_irradiance_density(atmospheric_data)

    # For viewing direction
    EdP_thetav = atmospheric_inv_data.direct_solar_irradiance

    # Direct atmospheric transmittance in illumination direction
    T_dir_dn = EdP_flat / (Eo * cos_sza)

    # Direct atmospheric transmittance in viewing direction
    T_dir_up = EdP_thetav / (Eo * cos_vza)

    # Downward diffuse solar flux for a flat surface:
    EhP_flat = atmospheric_data.diffuse_solar_irradiance
    # Downward diffuse solar flux for a flat surface: viewing direction
    EhP_flat_thetav = atmospheric_inv_data.diffuse_solar_irradiance

    # Total downward flux for a flat surface
    EtP_flat = EdP_flat + EhP_flat

    # Atmospheric spherical albedo
    rs = atmospheric_data.spherical_albedo.total

    # Atmospheric path radiance
    LtA = atmospheric_data.atmospheric_intrinsic_radiance

    # Atmospheric diffuse transmittance. In viewing direction
    td = EhP_flat_thetav / (Eo * cos_vza)

    # Flat or tilted terrain options
    if case.flat_terrain:

        # Direct solar irradiance at surface
        EdP = EdP_flat  # Same quantity as 6S

        # Sky view factor
        vd = 1

    else:  # tilted terrain
        EdP = bb * Eo * cos_sza_eff * T_dir_dn

        # Sky view factor: with or without anisotropy of diffuse irradiance
        # at grazing angles
        if not dif_anis:
            # Assume isotropic irradiance, eq. 15
            vd = topo_bands["vt"]
        else:
            # Account for anisotropy of diffuse irradiance at grazing
            # angles, eq. 16
            vd = _anisotropic_sky_factor(
                beam_visibility=bb,
                direct_transmittance=T_dir_dn,
                cos_local_incidence=cos_sza_eff,
                cos_solar_zenith=cos_sza,
                sky_view_factor=topo_bands["vt"],
            )

    # Contribution of the neighbouring slopes to the satellite signal
    # Set to zero (modified in iteration for full run)
    # LtNa = LtGA + LtGGA + LtGAGA
    LtNA = 0

    # Direct radiation reflected by pixel to the satellite sensor
    # (eq. 2)
    LdP = view_ground * (brf / np.pi) * EdP * T_dir_up

    # Iterative calculation
    # Initialise the convergence and iterator
    l_difference = 1
    i = 1

    while l_difference > 1e-3 and i <= max_iterations:

        if verbose:
            print("Iteration number: %s" % i)

        #  Atmospheric contribution to the pixel (multple reflections),
        #  eq. 20
        # e_flat_ground = e_dir + e_diff_flat; s_atm = rho_s in Modimlab
        # R(k-1)(M)dSm = rho_e
        EtGAP = EtP_flat * ((re[:, :, i - 1] * rs) /
                            (1 - re[:, :, i - 1] * rs))

        # Atmospheric coupling switch
        if case.atm_coupl:
            EtP = EtP_flat + EtGAP
        else:
            EtP = EtP_flat

        # Terrain contribution switch
        if case.terrain_contribution:

            # Add shadow off option in eq. 18
            shad = bb_nbh  # Disable reillumination from shadows

            # # Terrain re-illumination, eq. 18 substituted by eq. 11 from
            # # Sirguey et al. 2011
            # Terrain reflected irradiance
            EtGP = EtP * (rt[:, :, i - 1] * shad * topo_bands["ct"]) / (
                1 - rt[:, :, i - 1] * shad * ct_nbh)
        else:

            EtGP = 0

        # Combine parts of the total diffuse irradiance incoming at the
        # surface of the pixel, built from eq. 3 (rewritten based on grey
        # recaps)
        EhP = EhP_flat * vd + EtGP

        # If considering the neighbouring slopes contributions directly to
        # the satellite signal, "l_dif_dir" and "l_dif_ref_coupl_dif"
        # become != 0
        if case.atm2sensor:
            # LtNA = LtGA + LtGGA + LtGAGA, eq. 23
            LtNA = (td * re[:, :, i - 1] *
                    (EdP_flat + EhP_flat + EtGAP)) / np.pi

        # Diffuse radiation reflected by pixel to the satellite sensor
        # Eq. 4
        LhP = view_ground * (hdr / np.pi) * EhP * T_dir_up

        # TOA radiance (equation 1)
        l_total = LdP + LhP + LtNA + LtA

        # Update the surface hemispherical-conical reflectance
        r_current_dividend = np.pi * (l_total - LtNA - LtA)

        r_current_divisor = T_dir_up * view_ground * (EdP + EhP)

        r_currentstep = np.divide(r_current_dividend, r_current_divisor,
                                  out=np.zeros_like(r_current_dividend),
                                  where=r_current_divisor != 0)

        # Update the reflectance stack
        r = np.dstack((r, r_currentstep))

        # Apply the averaging of the updated reflectance
        re_i = _window_mean(r[:, :, i], aw, boundary="nearest")
        re = np.dstack((re, re_i))
        rt_i = _window_mean(r[:, :, i], tw, boundary="nearest")
        rt = np.dstack((rt, rt_i))

        # Update the radiance stack
        l_toa = np.dstack((l_toa, l_total))

        # Update the convergence indicator (removing the edges)
        l_difference = np.abs(
            np.nanmean(l_toa[2:-2, 2:-2, i] - l_toa[2:-2, 2:-2, i - 1])
        )
        if verbose:
            print(
                "radiance difference with previous"
                " iteration = %s" % l_difference
            )
        i += 1

    if l_difference > 1e-3:
        raise RuntimeError(
            f"Terrain radiance did not converge after {max_iterations} "
            "iterations"
        )

    toa_radiance = l_toa[:, :, -1]
    components = {
        "direct_surface": LdP,
        "diffuse_surface": LhP,
        "neighbour_atmosphere": LtNA,
        "atmospheric_path": LtA,
    }
    components = {
        name: np.broadcast_to(
            np.asarray(value, dtype=np.float64), toa_radiance.shape
        ).copy()
        for name, value in components.items()
    }
    if not return_reflectance:
        if return_components:
            return toa_radiance, components
        return toa_radiance

    reflectance_denominator = Eo * cos_sza
    toa_reflectance = np.divide(
        np.pi * toa_radiance,
        reflectance_denominator,
        out=np.full_like(toa_radiance, np.nan, dtype=np.float64),
        where=reflectance_denominator > 0,
    )
    if return_components:
        return toa_radiance, toa_reflectance, components
    return toa_radiance, toa_reflectance

