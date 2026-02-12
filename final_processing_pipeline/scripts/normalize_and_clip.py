import numpy as np
import xarray as xr
from typing import List, Dict


def normalize_dem(ds: xr.Dataset, max_earth_height: float = 8849.0) -> xr.Dataset:
    """Scales COP_DEM to [0, 1] based on a physical maximum."""
    assert "COP_DEM" in ds.data_vars, "COP_DEM not found in dataset variables."

    normalized = ds["COP_DEM"] / max_earth_height

    # Assert physical plausibility
    dem_max = float(normalized.max())
    assert (
        dem_max <= 1.0
    ), f"DEM normalization failed: Max is {dem_max}, expected <= 1.0"
    assert (
        normalized.min() >= -0.05
    ), f"DEM has unexpected negative values: {normalized.min().values}"

    ds["COP_DEM"] = normalized.astype("float32")
    return ds


def calculate_global_era5_stats(
    cubes: Dict[str, xr.Dataset], vars_to_norm: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calculates global percentiles for ERA5 variables across all cubes.
    Expects squeezed 1D scalar time series (time dimension only).
    """
    stats = {}

    for var in vars_to_norm:
        all_time_values = []

        for key, ds in cubes.items():
            if var not in ds.data_vars:
                continue

            # Verification: Ensure it's 1D (squeezed) before calculating stats
            assert len(ds[var].dims) == 1, f"Variable {var} in cube {key} is not 1D."

            sample = ds[var].values
            valid_sample = sample[~np.isnan(sample)]

            if valid_sample.size > 0:
                all_time_values.append(valid_sample)

        if all_time_values:
            combined_values = np.concatenate(all_time_values)

            # Robust scaling using percentiles to mitigate ERA5 outliers
            p_min = np.nanpercentile(combined_values, 0.01)
            p_max = np.nanpercentile(combined_values, 99.99)

            # Assert that range is valid to avoid division by zero
            assert (
                p_max > p_min
            ), f"Invalid range for variable {var}: min {p_min} == max {p_max}"

            stats[var] = {"p_min": float(p_min), "p_max": float(p_max)}
        else:
            print(f"Warning: No valid data found for variable {var} across all cubes.")

    return stats


def normalize_era5(ds: xr.Dataset, stats: Dict[str, Dict[str, float]]) -> xr.Dataset:
    """Applies robust min-max scaling and clips outliers to [0, 1]."""
    for var, s in stats.items():
        if var not in ds.data_vars:
            continue

        p_min = s["p_min"]
        p_max = s["p_max"]

        # Apply normalization
        normalized = (ds[var] - p_min) / (p_max - p_min)

        # Assert: Check if we are creating infinite values
        assert not np.isinf(
            normalized
        ).any(), f"Normalization produced Inf values in {var}"

        ds[var] = normalized.astype("float32")

        # # Clip to [0, 1] to handle values outside the 0.01-99.99 percentile range
        # ds[var] = normalized.clip(0, 1).astype("float32")

    return ds


def final_clipping_instance(ds: xr.Dataset) -> xr.Dataset:
    """Safety net: Clips all variables except ESA_Lc to interval of [-5; 5]."""

    for var in ds.data_vars:
        if var != "ESA_LC":
            ds[var] = ds[var].clip(-5, 5)

    return ds


def check_standardization(ds: xr.Dataset, vars_to_check: List[str]):
    """Prints range and mean for verification of the normalization process."""
    print(f"{'Variable':<25} | {'Min':<8} | {'Max':<8} | {'Mean':<8}")
    print("-" * 60)
    for var in vars_to_check:
        if var not in ds.data_vars:
            continue

        v_min = float(ds[var].min())
        v_max = float(ds[var].max())
        v_mean = float(ds[var].mean())
        print(f"{var:<25} | {v_min:8.4f} | {v_max:8.4f} | {v_mean:8.4f}")
