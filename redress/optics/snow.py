#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snow optics tools.

This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
import numpy as np
import snowoptics as so


DEFAULT_DUST_DENSITY_KG_M3 = 2600.0
DUST_MODELS = ("muller2011", "skiles2014")


def lap_impurities(
    black_carbon_ng_g=0.0,
    dust_ug_g=0.0,
    dust_density_kg_m3=DEFAULT_DUST_DENSITY_KG_M3,
    dust_model="muller2011",
):
    """Build a Snowoptics impurity dictionary from common LAP units.

    Parameters
    ----------
    black_carbon_ng_g : float
        Black-carbon concentration in ng g-1 of snow.
    dust_ug_g : float
        Mineral-dust concentration in microgram g-1 of snow.
    dust_density_kg_m3 : float
        Bulk density used by Snowoptics to derive the dust mass absorption
        coefficient. The default represents a generic mineral dust.
    dust_model : {"muller2011", "skiles2014"}
        Snowoptics dust refractive-index formulation. Müller et al. (2011)
        represents the more absorbing formulation.

    Returns
    -------
    dict
        Snowoptics impurity concentrations expressed as kg kg-1. Zero-valued
        species are omitted.
    """

    concentrations = {
        "black_carbon_ng_g": float(black_carbon_ng_g),
        "dust_ug_g": float(dust_ug_g),
        "dust_density_kg_m3": float(dust_density_kg_m3),
    }
    if not all(np.isfinite(value) for value in concentrations.values()):
        raise ValueError("LAP concentrations and density must be finite")
    if concentrations["black_carbon_ng_g"] < 0:
        raise ValueError("black_carbon_ng_g cannot be negative")
    if concentrations["dust_ug_g"] < 0:
        raise ValueError("dust_ug_g cannot be negative")
    if concentrations["dust_density_kg_m3"] <= 0:
        raise ValueError("dust_density_kg_m3 must be greater than zero")
    if dust_model not in DUST_MODELS:
        choices = ", ".join(DUST_MODELS)
        raise ValueError(f"dust_model must be one of: {choices}")

    impurities = {}
    if concentrations["black_carbon_ng_g"] > 0:
        impurities["BC"] = concentrations["black_carbon_ng_g"] * 1e-9
    if concentrations["dust_ug_g"] > 0:
        species = "dust" if dust_model == "muller2011" else "dust_skiles"
        impurities[species] = (
            concentrations["dust_ug_g"] * 1e-6,
            concentrations["dust_density_kg_m3"],
        )
    return impurities


def albedo_kokhanovsky(
    wls,
    sza,
    dd,
    ssa,
    impurities=None,
    B=1.6,
    g=0.845,
):
    """Calculate black sky, white sky and blue sky albedo.

    Using the Asymptotic Radiative Transfer theory from Kokhanovsky and Zege,
    2004 (implemented in the snowoptics library by Ghislain Picard:
     https://github.com/ghislainp/snowoptics), calculate direct (black sky),
    diffuse (white sky) and total (blue sky) albedo for a given snow SSA, SZA,
    and direct-to-diffuse ratio. The calculations rely on the B and g values
    found in Libois et al. 2013 (doi:10.5194/tc-7-1803-2013).

    :param wls: Array of wavelength values at which to compute the albedo (nm)
    :type wls: ndarray
    :param sza: Solar Zenith Angle grid (rad)
    :type sza: ndarray
    :param dd: Direct-to-diffuse ratio values at the given wavelengths
    :type dd: ndarray
    :param ssa: Specific surface area of the snow (m2/kg)
    :type ssa: float
    :param impurities: Snowoptics LAP concentrations in kg/kg
    :type impurities: dict, optional
    :param B: Snow absorption enhancement parameter
    :type B: float, optional
    :param g: Snow asymmetry factor
    :type g: float, optional
    :return: black sky, white sky and blue sky albedos
    :rtype: tuple [3]
    """
    # Convert wavelengths from nanometers to meters (done here to avoid
    # confusion).
    wls_m = wls * 1e-9
    sza = np.asarray(sza, dtype=np.float64)
    if np.any((sza < 0) | (sza > np.pi / 2)):
        raise ValueError("solar zenith angles must be in [0, pi/2] radians")

    # White-sky albedo does not depend on solar zenith angle.
    alb_diff_flat_val = so.albedo_diffuse_KZ04(
        wls_m,
        ssa,
        impurities=impurities,
        ni="w2008",
        B=B,
        g=g,
    )

    alb_dir_flat = directional_albedo_kokhanovsky(
        wls,
        sza,
        ssa,
        impurities=impurities,
        B=B,
        g=g,
    )

    # Make diffuse albedo and diffuse fraction conform to the angle grid.
    alb_diff_flat = np.full(sza.shape, alb_diff_flat_val, dtype=np.float64)
    diffuse_fraction = np.broadcast_to(
        np.asarray(dd, dtype=np.float64), sza.shape
    )

    # Blue-sky albedo is the direct/diffuse irradiance-weighted combination.
    alb_kok_flat = (
        (1.0 - diffuse_fraction) * alb_dir_flat
        + diffuse_fraction * alb_diff_flat
    )

    return (alb_dir_flat, alb_diff_flat, alb_kok_flat)


def directional_albedo_kokhanovsky(
    wls,
    zenith,
    ssa,
    impurities=None,
    B=1.6,
    g=0.845,
):
    """Calculate directional-hemispherical snow reflectance.

    Through BRDF reciprocity, evaluating this quantity at the local viewing
    zenith also gives the hemispherical-directional reflectance required for
    diffuse illumination in the REDRESS forward model.
    """

    wavelength_m = wls * 1e-9
    zenith = np.asarray(zenith, dtype=np.float64)
    if np.any((zenith < 0) | (zenith > np.pi / 2)):
        raise ValueError("zenith angles must be in [0, pi/2] radians")

    # Snowoptics 0.99.2 implements this expression directly but validates the
    # cosine with a scalar assertion. Reproduce the same ART equation here so
    # full rasters of local angles are evaluated without a Python pixel loop.
    alpha = so.compute_alpha(
        wavelength_m,
        ssa,
        impurities=impurities,
        ni="w2008",
        B=B,
        g=g,
    )
    angular_factor = (3.0 / 7.0) * (1.0 + 2.0 * np.cos(zenith))
    return np.exp(-np.sqrt(alpha) * angular_factor)


# Quadrature used to close the BRF on the directional albedo.
_CLOSURE_ZENITH_NODES = 32
_CLOSURE_AZIMUTH_SAMPLES = 72


def _art_brf(theta_i, theta_v, phi, absorption):
    """Evaluate the ART BRF for a given absorption exponent.

    ``absorption`` is the square root of the Snowoptics ``alpha``, the same
    term that sets the plane albedo exp(-absorption u(mu)).
    """

    cos_i = np.cos(theta_i)
    cos_v = np.cos(theta_v)
    # Scattering angle with phi = 0 on the forward-scattering side.
    theta = np.degrees(np.arccos(np.clip(
        -cos_i * cos_v + np.sin(theta_i) * np.sin(theta_v) * np.cos(phi),
        -1.0,
        1.0,
    )))
    phase = 11.1 * np.exp(-0.087 * theta) + 1.1 * np.exp(-0.014 * theta)
    r0 = (
        1.247 + 1.186 * (cos_i + cos_v) + 5.157 * cos_i * cos_v + phase
    ) / (4.0 * (cos_i + cos_v))
    u_i = 3.0 / 7.0 * (1.0 + 2.0 * cos_i)
    u_v = 3.0 / 7.0 * (1.0 + 2.0 * cos_v)
    return r0 * np.exp(-absorption * u_i * u_v / r0)


def _brf_closure_factors(absorption):
    """Return reciprocal factors that close the BRF on the albedo.

    The factors f solve f(mu_i) * (1/pi) * integral of f(mu_v) R mu_v
    over the view hemisphere = exp(-absorption u(mu_i)) on Gauss-Legendre
    nodes in mu. Scaling R by f(mu_i) f(mu_v) keeps it reciprocal.
    """

    nodes, weights = np.polynomial.legendre.leggauss(_CLOSURE_ZENITH_NODES)
    mu = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    zenith = np.arccos(mu)
    azimuth = (
        (np.arange(_CLOSURE_AZIMUTH_SAMPLES) + 0.5)
        * 2.0 * np.pi / _CLOSURE_AZIMUTH_SAMPLES
    )
    brf = _art_brf(
        zenith[:, np.newaxis, np.newaxis],
        zenith[np.newaxis, :, np.newaxis],
        azimuth[np.newaxis, np.newaxis, :],
        absorption,
    )
    # (1/pi) times the azimuth integral is twice the azimuthal mean.
    kernel = 2.0 * brf.mean(axis=2) * (mu * weights)[np.newaxis, :]
    albedo = np.exp(-absorption * 3.0 / 7.0 * (1.0 + 2.0 * mu))

    factors = np.ones_like(mu)
    for _ in range(500):
        updated = np.sqrt(factors * albedo / (kernel @ factors))
        converged = np.max(np.abs(updated - factors)) < 1e-12
        factors = updated
        if converged:
            break
    return mu, factors


def brf_kokhanovsky(
    theta_i,
    theta_v,
    phi,
    wvl,
    SSA,
    impurities=None,
    B=1.6,
    g=0.845,
    close_on_albedo=True,
):
    """Calculate snow BRF.

    Snow BRF function from Kokhanovsky et al. Described in DOI:
    10.1109/LgrS.2012.2185775. Calculate BRF value for given illumination /
    viewing angles, wavelength and SSA.

    The absorption term is the Snowoptics ``alpha`` used by
    :func:`directional_albedo_kokhanovsky`, so the BRF and the albedos
    share B, g, the ice refractive index and any impurities. With
    ``close_on_albedo`` the BRF is further scaled by reciprocal factors
    f(theta_i) f(theta_v) so that its integral over the view hemisphere
    equals that directional albedo. Without it, the asymptotic BRF
    reflects up to about twice the albedo in strongly absorbing bands such
    as B11 and B12.

    :param theta_i: illumination zenith angle
    :type theta_i: float
    :param theta_v: viewing zenith angle
    :type theta_v: float
    :param phi: ART relative azimuth angle: zero on the forward-scattering
        side and pi in exact backscatter
    :type phi: float
    :param wvl: wavelength (nm), a scalar
    :type wvl: float
    :param SSA: Specific surface area of snow
    :type SSA: float
    :param impurities: Snowoptics LAP concentrations in kg/kg
    :type impurities: dict, optional
    :param B: Snow absorption enhancement parameter
    :type B: float, optional
    :param g: Snow asymmetry factor
    :type g: float, optional
    :param close_on_albedo: scale the BRF to integrate to the albedo
    :type close_on_albedo: bool, optional
    :return: BRF
    :rtype: ndarray
    """

    if np.ndim(wvl) != 0:
        raise ValueError("brf_kokhanovsky expects one wavelength")
    absorption = float(np.sqrt(so.compute_alpha(
        float(wvl) * 1e-9,
        SSA,
        impurities=impurities,
        ni="w2008",
        B=B,
        g=g,
    )))
    rr = _art_brf(theta_i, theta_v, phi, absorption)
    if not close_on_albedo:
        return rr

    mu, factors = _brf_closure_factors(absorption)
    factor_i = np.interp(np.cos(theta_i), mu, factors)
    factor_v = np.interp(np.cos(theta_v), mu, factors)
    return rr * factor_i * factor_v

    # r0 in kokhanovsky's paper
    r = brf0(theta_i, theta_v, phi)

    # k0 for theta_v and theta i
    k0v = 3. / 7. * (1. + 2. * np.cos(theta_v))
    k0i = 3. / 7. * (1. + 2. * np.cos(theta_i))

    # get refractive index
    n, ni = so.refractive_index.refice2008(wvl_m)

    gamma = 4 * np.pi * (ni + M) / wvl_m

    if impurities:
        # Convert the Snowoptics co-single-scattering albedo back to an
        # equivalent absorption coefficient. This retains the original
        # REDRESS BRF expression while using the same LAP optical properties
        # and mass concentrations as the direct and diffuse albedo.
        rho_ice = 917.0
        cossalb, _ = so.compute_single_scattering_properties(
            wvl_m,
            SSA,
            impurities=impurities,
            ni="w2008",
            B=B,
            g=g,
        )
        gamma = cossalb * SSA * rho_ice / (2.0 * B)

    # Alpha = sqrt(gamma * L), with L approximately 13d, where
    # d is the average optical diameter of snow:
    # d = 6 / refract_ind_ice * SSA
    alpha = np.sqrt(gamma * b * 6. / (917 * SSA))

    # r(theta_i, theta_v, phi)
    rr = r * np.exp(-alpha * k0i * k0v / r)

    return rr

