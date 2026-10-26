#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sentinel-2 look-up tables generated with REDRESS snow optics."""

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import xarray as xr

from redress.inputs.products import Model


@dataclass(frozen=True)
class Sentinel2EndmemberScenario:
    """Physical and observational parameters of one LUT endmember."""

    name: str
    category: str
    specific_surface_area: float = 20.0
    black_carbon_ng_g: float = 0.0
    dust_ug_g: float = 0.0
    dust_density_kg_m3: float = 2600.0
    dust_model: str = "muller2011"
    solar_zenith_angle: float = 45.0
    local_incidence_angle: float = 45.0
    view_zenith_angle: float = 0.0
    local_view_zenith_angle: float = 0.0
    relative_azimuth_angle: float = 0.0
    altitude_m: float = 2000.0
    aod550: float = 0.10
    water_vapour_g_cm2: float = 1.0
    ozone_cm_atm: float = 0.30
    sky_view_factor: float = 1.0
    terrain_configuration_factor: float = 0.0
    shadowed: bool = False

    def validate(self, enforce_view_factor_closure: bool = True) -> None:
        """Validate physical ranges needed by REDRESS and Py6S.

        REDRESS computes the terrain configuration factor from a DEM as
        ``ct = 1 - vt``. Synthetic scenarios follow the same bowl closure by
        default. Closure can be disabled explicitly when the two factors are
        varied independently in a controlled sensitivity experiment.
        """

        if not self.name.strip():
            raise ValueError("scenario name cannot be empty")
        if not self.category.strip():
            raise ValueError("scenario category cannot be empty")
        for name, value in (
            ("specific_surface_area", self.specific_surface_area),
            ("black_carbon_ng_g", self.black_carbon_ng_g),
            ("dust_ug_g", self.dust_ug_g),
            ("dust_density_kg_m3", self.dust_density_kg_m3),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.specific_surface_area <= 0:
            raise ValueError("specific_surface_area must be greater than zero")
        if self.black_carbon_ng_g < 0:
            raise ValueError("black_carbon_ng_g cannot be negative")
        if self.dust_ug_g < 0:
            raise ValueError("dust_ug_g cannot be negative")
        if self.dust_density_kg_m3 <= 0:
            raise ValueError("dust_density_kg_m3 must be greater than zero")
        if self.dust_model not in ("muller2011", "skiles2014"):
            raise ValueError(
                "dust_model must be 'muller2011' or 'skiles2014'"
            )
        for name, value in (
            ("solar_zenith_angle", self.solar_zenith_angle),
            ("local_incidence_angle", self.local_incidence_angle),
            ("view_zenith_angle", self.view_zenith_angle),
            ("local_view_zenith_angle", self.local_view_zenith_angle),
        ):
            if not 0 <= value < 90:
                raise ValueError(f"{name} must be in [0, 90) degrees")
        if not 0 <= self.relative_azimuth_angle <= 360:
            raise ValueError(
                "relative_azimuth_angle must be in [0, 360] degrees"
            )
        if self.altitude_m < 0:
            raise ValueError("altitude_m cannot be negative")
        if self.aod550 < 0:
            raise ValueError("aod550 cannot be negative")
        if self.water_vapour_g_cm2 < 0:
            raise ValueError("water_vapour_g_cm2 cannot be negative")
        if self.ozone_cm_atm < 0:
            raise ValueError("ozone_cm_atm cannot be negative")
        for name, value in (
            ("sky_view_factor", self.sky_view_factor),
            ("terrain_configuration_factor",
             self.terrain_configuration_factor),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        view_factor_sum = (
            self.sky_view_factor + self.terrain_configuration_factor
        )
        if enforce_view_factor_closure and not np.isclose(
            view_factor_sum, 1.0, rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                "sky_view_factor and terrain_configuration_factor must sum "
                "to 1 under the REDRESS bowl approximation; set "
                "enforce_view_factor_closure=False only for an intentional "
                "independent-factor sensitivity experiment"
            )


def _coordinate(values: Iterable[float], name: str) -> np.ndarray:
    """Return a validated, one-dimensional floating-point coordinate."""

    coordinate = np.asarray(tuple(values), dtype=np.float64)
    if coordinate.ndim != 1 or coordinate.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(coordinate)):
        raise ValueError(f"{name} contains a non-finite value")
    if np.unique(coordinate).size != coordinate.size:
        raise ValueError(f"{name} contains duplicate values")
    return coordinate


def _geometry(
    solar_zenith_deg: float,
    view_zenith_deg: float,
    relative_azimuth_deg: float,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build a one-pixel flat-terrain geometry for the REDRESS model."""

    solar_zenith = np.deg2rad(solar_zenith_deg)
    view_zenith = np.deg2rad(view_zenith_deg)
    relative_azimuth = np.deg2rad(relative_azimuth_deg)

    shape = (1, 1)
    angles = xr.Dataset(
        {
            "SZA": (("x", "y"), np.full(shape, solar_zenith)),
            "SAA": (("x", "y"), np.full(shape, relative_azimuth)),
            "VZA": (("x", "y"), np.full(shape, view_zenith)),
            "VAA": (("x", "y"), np.zeros(shape)),
        }
    )
    topography = xr.Dataset(
        {
            "eff_sza": (("x", "y"), np.full(shape, solar_zenith)),
            "eff_vza": (("x", "y"), np.full(shape, view_zenith)),
        }
    )
    return angles, topography


def _forward_scene(
    scenario: Sentinel2EndmemberScenario,
    scene_size: int,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Create a uniform synthetic scene for diagnostic model experiments.

    Production scenes should obtain ``vt`` and ``ct`` from
    :meth:`redress.inputs.products.Dem.compute_topo_bands`, which derives both
    fields from the DEM slope, aspect and horizon.
    """

    if scene_size < 9 or scene_size % 2 == 0:
        raise ValueError("scene_size must be an odd integer of at least 9")

    shape = (scene_size, scene_size)
    sza = np.deg2rad(scenario.solar_zenith_angle)
    incidence = np.deg2rad(scenario.local_incidence_angle)
    vza = np.deg2rad(scenario.view_zenith_angle)
    local_vza = np.deg2rad(scenario.local_view_zenith_angle)
    relative_azimuth = np.deg2rad(scenario.relative_azimuth_angle)

    angles = xr.Dataset(
        {
            "SZA": (("x", "y"), np.full(shape, sza)),
            "SAA": (("x", "y"), np.full(shape, relative_azimuth)),
            "VZA": (("x", "y"), np.full(shape, vza)),
            "VAA": (("x", "y"), np.zeros(shape)),
        },
        coords={"x": np.arange(scene_size), "y": np.arange(scene_size)},
    )
    topography = xr.Dataset(
        {
            "altitude": (("x", "y"),
                         np.full(shape, scenario.altitude_m)),
            "eff_sza": (("x", "y"), np.full(shape, incidence)),
            "eff_vza": (("x", "y"), np.full(shape, local_vza)),
            "vt": (("x", "y"),
                   np.full(shape, scenario.sky_view_factor)),
            "ct": (("x", "y"), np.full(
                shape, scenario.terrain_configuration_factor
            )),
            "all_shadows": (("x", "y"), np.full(
                shape, 0.0 if scenario.shadowed else 1.0
            )),
        },
        coords=angles.coords,
    )
    return angles, topography


def create_sentinel2_snow_brf_lut(
    platform: str,
    band_names: Iterable[str],
    specific_surface_area: Iterable[float],
    solar_zenith_angle: Iterable[float],
    view_zenith_angle: Iterable[float],
    relative_azimuth_angle: Iterable[float],
) -> xr.Dataset:
    """Create a band-integrated Sentinel-2 snow BRF look-up table.

    This first LUT represents a locally flat, fully illuminated snow surface.
    It isolates the effects of snow specific surface area and observation
    geometry. Atmospheric attenuation, terrain re-illumination and cast
    shadows are outside its scope.

    Parameters
    ----------
    platform
        Sentinel-2 platform, currently ``S2A`` or ``S2B``.
    band_names
        Ordered MSI bands to include.
    specific_surface_area
        Snow specific surface area values in m2 kg-1.
    solar_zenith_angle
        Solar zenith angles in degrees.
    view_zenith_angle
        Sensor viewing zenith angles in degrees.
    relative_azimuth_angle
        Solar-minus-view azimuth angles in degrees.

    Returns
    -------
    xarray.Dataset
        Dataset containing the dimensionless band reflectance factor ``brf``.
    """

    ssa = _coordinate(specific_surface_area, "specific_surface_area")
    sza = _coordinate(solar_zenith_angle, "solar_zenith_angle")
    vza = _coordinate(view_zenith_angle, "view_zenith_angle")
    raa = _coordinate(relative_azimuth_angle, "relative_azimuth_angle")

    if np.any(ssa <= 0):
        raise ValueError("specific_surface_area must be greater than zero")
    if np.any((sza < 0) | (sza >= 90)):
        raise ValueError("solar_zenith_angle must be in [0, 90) degrees")
    if np.any((vza < 0) | (vza >= 90)):
        raise ValueError("view_zenith_angle must be in [0, 90) degrees")
    if np.any((raa < 0) | (raa > 360)):
        raise ValueError("relative_azimuth_angle must be in [0, 360] degrees")

    model = Model()
    model.set_sentinel2(platform, band_names)
    band_count = len(model.meta["bandlist"])

    shape = (ssa.size, sza.size, vza.size, raa.size, band_count)
    brf = np.empty(shape, dtype=np.float32)

    for ssa_index, ssa_value in enumerate(ssa):
        model.ssa = float(ssa_value)
        for sza_index, sza_value in enumerate(sza):
            for vza_index, vza_value in enumerate(vza):
                for raa_index, raa_value in enumerate(raa):
                    model.angles, model.topo_bands = _geometry(
                        sza_value, vza_value, raa_value
                    )
                    model.compute_brf()
                    brf[ssa_index, sza_index, vza_index, raa_index, :] = [
                        np.asarray(value[1]).item() for value in model.brf
                    ]

    dataset = xr.Dataset(
        data_vars={
            "brf": (
                (
                    "specific_surface_area",
                    "solar_zenith_angle",
                    "view_zenith_angle",
                    "relative_azimuth_angle",
                    "band",
                ),
                brf,
            )
        },
        coords={
            "specific_surface_area": ssa,
            "solar_zenith_angle": sza,
            "view_zenith_angle": vza,
            "relative_azimuth_angle": raa,
            "band": model.meta["bandlist"],
            "wavelength_nm": (
                "band", np.asarray(model.meta["wavelengths"])
            ),
            "native_resolution_m": (
                "band", np.asarray(model.meta["native_resolution_m"])
            ),
        },
        attrs={
            "title": "REDRESS Sentinel-2 snow BRF look-up table",
            "platform": model.meta["platform"],
            "sensor": "MSI",
            "surface_geometry": "locally flat and fully illuminated",
            "spectral_integration": "Py6S MSI relative spectral response",
        },
    )

    dataset["brf"].attrs.update(
        long_name="bidirectional reflectance factor",
        units="1",
    )
    dataset["specific_surface_area"].attrs.update(
        long_name="snow specific surface area",
        units="m2 kg-1",
    )
    for coordinate_name in (
        "solar_zenith_angle",
        "view_zenith_angle",
        "relative_azimuth_angle",
    ):
        dataset[coordinate_name].attrs["units"] = "degree"
    dataset["wavelength_nm"].attrs.update(
        long_name="response-weighted effective wavelength",
        units="nm",
    )
    dataset["native_resolution_m"].attrs.update(
        long_name="native Sentinel-2 spatial resolution",
        units="m",
    )

    return dataset


def create_sentinel2_toa_lut(
    platform: str,
    band_names: Iterable[str],
    scenarios: Iterable[Sentinel2EndmemberScenario],
    month: int,
    day: int,
    aerosol_model: str = "Continental",
    atmospheric_profile: str | None = None,
    case_number: int = 5,
    scene_size: int = 9,
    enforce_view_factor_closure: bool = True,
) -> xr.Dataset:
    """Run REDRESS for named Sentinel-2 TOA endmember scenarios.

    The resulting TOA reflectance is derived from the REDRESS TOA radiance,
    the Py6S band solar irradiance and the cosine of the global solar zenith
    angle. A uniform scene is used so the center pixel is unaffected by the
    convolution boundaries of the iterative terrain model. The synthetic sky
    view and terrain configuration factors must satisfy the REDRESS bowl
    closure by default. Set ``enforce_view_factor_closure=False`` only when
    deliberately varying the two factors independently for sensitivity work.
    """

    scenarios = tuple(scenarios)
    if not scenarios:
        raise ValueError("at least one endmember scenario is required")
    if len({scenario.name for scenario in scenarios}) != len(scenarios):
        raise ValueError("scenario names must be unique")
    for scenario in scenarios:
        scenario.validate(
            enforce_view_factor_closure=enforce_view_factor_closure
        )

    model = Model()
    model.set_sentinel2(platform, band_names)
    model.import_rt_model("python_6S")

    scenario_count = len(scenarios)
    band_count = len(model.meta["bandlist"])
    toa_radiance = np.empty((scenario_count, band_count), dtype=np.float32)
    toa_reflectance = np.empty_like(toa_radiance)
    surface_brf = np.empty_like(toa_radiance)
    direct_albedo = np.empty_like(toa_radiance)
    diffuse_albedo = np.empty_like(toa_radiance)
    component_names = (
        "direct_surface",
        "diffuse_surface",
        "neighbour_atmosphere",
        "atmospheric_path",
    )
    radiance_components = {
        name: np.empty_like(toa_radiance) for name in component_names
    }

    center = scene_size // 2
    for scenario_index, scenario in enumerate(scenarios):
        print(
            f"Running scenario {scenario_index + 1}/{scenario_count}: "
            f"{scenario.name}"
        )
        model.ssa = scenario.specific_surface_area
        model.set_lap(
            black_carbon_ng_g=scenario.black_carbon_ng_g,
            dust_ug_g=scenario.dust_ug_g,
            dust_density_kg_m3=scenario.dust_density_kg_m3,
            dust_model=scenario.dust_model,
        )
        model.angles, model.topo_bands = _forward_scene(
            scenario, scene_size=scene_size
        )
        model.set_rt_options(
            aero=aerosol_model,
            aod=scenario.aod550,
            water=scenario.water_vapour_g_cm2,
            ozone=scenario.ozone_cm_atm,
            atmos=atmospheric_profile,
            atcor=False,
            month=month,
            day=day,
        )
        model.simulate_TOA_radiance(case_number, verbose=False)

        for band_index, band_name in enumerate(model.meta["bandlist"]):
            toa_radiance[scenario_index, band_index] = \
                model.toa_rad[band_name].values[center, center]
            toa_reflectance[scenario_index, band_index] = \
                model.toa_reflectance[band_name].values[center, center]
            surface_brf[scenario_index, band_index] = \
                np.asarray(model.brf[band_index][1])[center, center]
            direct_albedo[scenario_index, band_index] = \
                np.asarray(model.albedo["direct"][band_index])[center, center]
            diffuse_albedo[scenario_index, band_index] = \
                np.asarray(model.albedo["diffuse"][band_index])[center, center]
            for component_name in component_names:
                radiance_components[component_name][
                    scenario_index, band_index
                ] = model.toa_components[component_name][band_name].values[
                    center, center
                ]

    scenario_fields = asdict(scenarios[0]).keys()
    coordinates = {
        "scenario": [scenario.name for scenario in scenarios],
        "band": model.meta["bandlist"],
        "wavelength_nm": ("band", np.asarray(model.meta["wavelengths"])),
        "native_resolution_m": (
            "band", np.asarray(model.meta["native_resolution_m"])
        ),
    }
    for field in scenario_fields:
        if field == "name":
            continue
        coordinates[field] = (
            "scenario", [getattr(scenario, field) for scenario in scenarios]
        )

    data_variables = {
        "surface_brf": (("scenario", "band"), surface_brf),
        "direct_albedo": (("scenario", "band"), direct_albedo),
        "diffuse_albedo": (("scenario", "band"), diffuse_albedo),
        "toa_radiance": (("scenario", "band"), toa_radiance),
        "toa_reflectance": (("scenario", "band"), toa_reflectance),
    }
    data_variables.update({
        f"toa_{name}_radiance": (("scenario", "band"), values)
        for name, values in radiance_components.items()
    })

    dataset = xr.Dataset(
        data_vars=data_variables,
        coords=coordinates,
        attrs={
            "title": "REDRESS Sentinel-2 TOA snow endmember LUT",
            "platform": model.meta["platform"],
            "sensor": "MSI",
            "month": month,
            "day": day,
            "aerosol_model": aerosol_model,
            "atmospheric_profile": (
                atmospheric_profile
                if atmospheric_profile is not None
                else "user water vapour and ozone"
            ),
            "redress_case_number": case_number,
            "scene_size": scene_size,
            "view_factor_relationship": "terrain factor = 1 - sky view",
            "view_factor_source": (
                "prescribed uniform diagnostic scenarios; production "
                "endmembers must use DEM-derived vt and ct"
            ),
            "view_factor_closure_enforced": (
                "true" if enforce_view_factor_closure else "false"
            ),
            "lap_optical_model": (
                "Snowoptics 0.99.2 Kokhanovsky-Zege impurity absorption"
            ),
            "black_carbon_optics": (
                "Bond and Bergstrom (2006), Snowoptics BC formulation"
            ),
        },
    )

    for variable in (
        "surface_brf", "direct_albedo", "diffuse_albedo", "toa_reflectance"
    ):
        dataset[variable].attrs["units"] = "1"
    dataset["toa_radiance"].attrs.update(
        long_name="top-of-atmosphere spectral radiance",
        units="W m-2 sr-1 um-1",
    )
    dataset["toa_reflectance"].attrs["long_name"] = \
        "top-of-atmosphere reflectance"
    dataset["surface_brf"].attrs["long_name"] = \
        "surface bidirectional reflectance factor"
    component_long_names = {
        "direct_surface": "direct surface-reflected TOA radiance",
        "diffuse_surface": "diffuse surface-reflected TOA radiance",
        "neighbour_atmosphere": (
            "neighbour-atmosphere coupling TOA radiance"
        ),
        "atmospheric_path": "atmospheric path radiance",
    }
    for component_name, long_name in component_long_names.items():
        dataset[f"toa_{component_name}_radiance"].attrs.update(
            long_name=long_name,
            units="W m-2 sr-1 um-1",
        )

    coordinate_units = {
        "specific_surface_area": "m2 kg-1",
        "black_carbon_ng_g": "ng g-1",
        "dust_ug_g": "ug g-1",
        "dust_density_kg_m3": "kg m-3",
        "solar_zenith_angle": "degree",
        "local_incidence_angle": "degree",
        "view_zenith_angle": "degree",
        "local_view_zenith_angle": "degree",
        "relative_azimuth_angle": "degree",
        "altitude_m": "m",
        "aod550": "1",
        "water_vapour_g_cm2": "g cm-2",
        "ozone_cm_atm": "cm-atm",
        "sky_view_factor": "1",
        "terrain_configuration_factor": "1",
    }
    for coordinate_name, units in coordinate_units.items():
        dataset[coordinate_name].attrs["units"] = units
    dataset["black_carbon_ng_g"].attrs["long_name"] = \
        "black-carbon mass concentration in snow"
    dataset["dust_ug_g"].attrs["long_name"] = \
        "mineral-dust mass concentration in snow"
    dataset["dust_density_kg_m3"].attrs["long_name"] = \
        "assumed mineral-dust bulk density"
    dataset["dust_model"].attrs["long_name"] = \
        "Snowoptics mineral-dust optical formulation"
    dataset["wavelength_nm"].attrs.update(
        long_name="response-weighted effective wavelength",
        units="nm",
    )
    dataset["native_resolution_m"].attrs["units"] = "m"

    return dataset
