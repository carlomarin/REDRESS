#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snow optics tools.

This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
from functools import lru_cache

import numpy as np
import snowoptics as so


DEFAULT_DUST_DENSITY_KG_M3 = 2600.0
DUST_MODELS = ("muller2011", "skiles2014")
# "art" uses the asymptotic Kokhanovsky-Zege albedo. "tartes" takes the
# albedo from the TARTES two-stream model, which stays accurate when ice
# absorbs strongly, as in Sentinel-2 B11 and B12.
SNOW_ALBEDO_MODELS = ("art", "tartes")
# TARTES grain-shape parameterization and ice refractive index. "constant"
# with "w2008" matches the ART albedo; TARTES's own defaults are
# "robledano23", which sets B from the wavelength-dependent refractive
# index, and "p2016".
TARTES_SHAPE_PARAMETERIZATION = "constant"
TARTES_REFRACTIVE_INDEX = "w2008"


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


def _validate_albedo_model(albedo_model):
    if albedo_model not in SNOW_ALBEDO_MODELS:
        raise ValueError(
            "albedo_model must be one of: " + ", ".join(SNOW_ALBEDO_MODELS)
        )


class _SnowopticsImpurity:
    """TARTES impurity type with the Snowoptics mass absorption.

    TARTES and Snowoptics add the same impurity co-albedo,
    2 / SSA * concentration * MAE, so this gives the TARTES albedo the
    same LAP optics as the ART albedo and BRF.
    """

    def __init__(self, species, density_kg_m3):
        if species in ("BC", "soot"):
            self._imaginary = so.refractive_index.refsoot_imag
        elif species == "dust":
            self._imaginary = so.refractive_index.refdust_imag
        elif species == "dust_skiles":
            self._imaginary = (
                lambda wavelength: so.refractive_index.refdust_imag(
                    wavelength, formulation="skiles2014"
                )
            )
        else:
            raise ValueError(f"unsupported impurity species: {species}")
        self.density = density_kg_m3

    def MAE(self, wavelength_m):
        return (
            6.0 * np.pi / wavelength_m / self.density
            * np.abs(self._imaginary(wavelength_m))
        )


def _impurity_key(impurities):
    """Hashable (species, concentration, density) tuple for caching."""

    key = []
    for species, value in sorted((impurities or {}).items()):
        try:
            concentration, density = value
        except TypeError:
            # Snowoptics' default black-carbon density.
            concentration, density = value, 1270.0
        key.append((species, float(concentration), float(density)))
    return tuple(key)


def _tartes_albedo(wavelength_nm, ssa, zenith_degree, B, g,
                   shape_parameterization=None, refractive_index=None,
                   impurity_key=()):
    """Semi-infinite TARTES albedo; ``zenith_degree=None`` is diffuse.

    By default B and g are held constant and the ice refractive index is
    the one used by the ART albedo, so the two models differ only in their
    radiative transfer. See ``TARTES_SHAPE_PARAMETERIZATION``.
    ``impurity_key`` comes from ``_impurity_key``.
    """

    import tartes

    options = {
        "density": 300.0,  # A semi-infinite albedo does not depend on it.
        "shape_parameterization": (
            shape_parameterization or TARTES_SHAPE_PARAMETERIZATION
        ),
        "B0": B,
        "g0": g,
        "refrac_index": refractive_index or TARTES_REFRACTIVE_INDEX,
    }
    if impurity_key:
        # One snow layer: a list over layers of per-species concentrations.
        options["impurities"] = [[item[1] for item in impurity_key]]
        options["impurities_type"] = [
            _SnowopticsImpurity(species, density)
            for species, _, density in impurity_key
        ]
    if zenith_degree is None:
        return float(tartes.albedo(
            wavelength_nm * 1e-9, ssa, dir_frac=0.0, **options
        ))
    return float(tartes.albedo(
        wavelength_nm * 1e-9, ssa, dir_frac=1.0, sza=zenith_degree,
        **options
    ))


@lru_cache(maxsize=4096)
def _tartes_directional_table(wavelength_nm, ssa, B, g,
                              shape_parameterization, refractive_index,
                              impurity_key):
    """TARTES directional albedo on the closure nodes plus mu = 0 and 1."""

    mu, _ = _closure_nodes()
    mu = np.concatenate(([0.0], mu, [1.0]))
    albedo = np.array([
        _tartes_albedo(
            wavelength_nm, ssa, float(np.degrees(np.arccos(value))), B, g,
            shape_parameterization, refractive_index, impurity_key,
        )
        for value in mu
    ])
    return mu, albedo


def _directional_albedo_tartes(wavelength_nm, zenith, ssa, B, g,
                               impurities=None):
    mu, albedo = _tartes_directional_table(
        float(wavelength_nm), float(ssa), float(B), float(g),
        TARTES_SHAPE_PARAMETERIZATION, TARTES_REFRACTIVE_INDEX,
        _impurity_key(impurities),
    )
    return np.interp(np.cos(zenith), mu, albedo)


def albedo_kokhanovsky(
    wls,
    sza,
    dd,
    ssa,
    impurities=None,
    B=1.6,
    g=0.845,
    albedo_model="art",
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
    :param albedo_model: "art" or "tartes", see ``SNOW_ALBEDO_MODELS``
    :type albedo_model: str, optional
    :return: black sky, white sky and blue sky albedos
    :rtype: tuple [3]
    """
    # Convert wavelengths from nanometers to meters (done here to avoid
    # confusion).
    wls_m = wls * 1e-9
    sza = np.asarray(sza, dtype=np.float64)
    if np.any((sza < 0) | (sza > np.pi / 2)):
        raise ValueError("solar zenith angles must be in [0, pi/2] radians")

    _validate_albedo_model(albedo_model)
    # White-sky albedo does not depend on solar zenith angle.
    if albedo_model == "tartes":
        alb_diff_flat_val = _tartes_albedo(
            float(wls), ssa, None, B, g,
            impurity_key=_impurity_key(impurities),
        )
    else:
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
        albedo_model=albedo_model,
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
    albedo_model="art",
):
    """Calculate directional-hemispherical snow reflectance.

    Through BRDF reciprocity, evaluating this quantity at the local viewing
    zenith also gives the hemispherical-directional reflectance required for
    diffuse illumination in the REDRESS forward model. With
    ``albedo_model="tartes"`` it is interpolated in cos(zenith) from TARTES
    values at one wavelength.
    """

    _validate_albedo_model(albedo_model)
    wavelength_m = wls * 1e-9
    zenith = np.asarray(zenith, dtype=np.float64)
    if np.any((zenith < 0) | (zenith > np.pi / 2)):
        raise ValueError("zenith angles must be in [0, pi/2] radians")
    if albedo_model == "tartes":
        return _directional_albedo_tartes(
            wls, zenith, ssa, B, g, impurities
        )

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


def _closure_nodes():
    """Gauss-Legendre nodes and weights in mu on [0, 1]."""

    nodes, weights = np.polynomial.legendre.leggauss(_CLOSURE_ZENITH_NODES)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def _brf_closure_factors(absorption, albedo):
    """Return reciprocal factors that close the BRF on the albedo.

    The factors f solve f(mu_i) * (1/pi) * integral of f(mu_v) R mu_v
    over the view hemisphere = albedo(mu_i) on Gauss-Legendre nodes in mu,
    where R is the ART BRF with the given absorption exponent. Scaling R by
    f(mu_i) f(mu_v) keeps it reciprocal.
    """

    mu, weights = _closure_nodes()
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

    factors = np.ones_like(mu)
    for _ in range(500):
        updated = np.sqrt(factors * albedo / (kernel @ factors))
        converged = np.max(np.abs(updated - factors)) < 1e-12
        factors = updated
        if converged:
            break
    return factors


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
    albedo_model="art",
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
    as B11 and B12. With ``albedo_model="tartes"`` the closure targets the
    TARTES directional albedo instead, so ART supplies only the angular
    shape.

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
    :param albedo_model: "art" or "tartes", see ``SNOW_ALBEDO_MODELS``
    :type albedo_model: str, optional
    :return: BRF
    :rtype: ndarray
    """

    if np.ndim(wvl) != 0:
        raise ValueError("brf_kokhanovsky expects one wavelength")
    _validate_albedo_model(albedo_model)
    if albedo_model == "tartes" and not close_on_albedo:
        raise ValueError("the TARTES albedo requires close_on_albedo=True")
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

    mu, _ = _closure_nodes()
    if albedo_model == "tartes":
        albedo = _directional_albedo_tartes(
            wvl, np.arccos(mu), SSA, B, g, impurities
        )
    else:
        albedo = np.exp(-absorption * 3.0 / 7.0 * (1.0 + 2.0 * mu))
    factors = _brf_closure_factors(absorption, albedo)
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

