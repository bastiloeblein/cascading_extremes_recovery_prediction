from typing import Dict
import numpy as np
import xarray as xr
import spyndex
import gc


# -------------------------------------------------------------- CONSTANTS --------------------------------------------------------------------------------------

# SCL: 0=No Data, 1=Saturated/Defective, 2=Dark Area/Shadow
SCL_INVALID = [0, 1, 2]
# SCL: 3=Cloud Shadow, 8=Cloud Medium Prob, 9=Cloud High Prob, 10=Cirrus
SCL_CLOUDS = [3, 8, 9, 10]
# cloud_mask: 1, 2, 3 represent different cloud levels
CM_CLOUDS = [1, 2, 3]

# ESA vegetation classes
VEGETATION_CLASSES = [10, 20, 30, 40, 60, 90, 95, 100]

# Sentinel-2 L2A reflectance values usually range from 0 to 10000 (0-100% reflectance)
REFLECTANCE_MIN = 0
REFLECTANCE_MAX = 10000
NORM_FACTOR = 10000.0

# Expected physical range for NDVI
NDVI_MIN = -1.0
NDVI_MAX = 1.0

# S2 Bandnamen für spyndex
BAND_MAP_S2: Dict[str, str] = {
    "B": "B02_normalized",
    "G": "B03_normalized",
    "R": "B04_normalized",
    "RE1": "B05_normalized",
    "RE2": "B06_normalized",
    "RE3": "B07_normalized",
    "N": "B08_normalized",
    "S1": "B11_normalized",
    "S2": "B12_normalized",
}

BAND_MAP_S2_BASIC: Dict[str, str] = {
    "B": "B02_basic",
    "G": "B03_basic",
    "R": "B04_basic",
    "RE1": "B05_basic",
    "RE2": "B06_basic",
    "RE3": "B07_basic",
    "N": "B08_basic",
    "S1": "B11_basic",
    "S2": "B12_basic",
}

BAND_MAP_S2_STRICT: Dict[str, str] = {
    "B": "B02_strict",
    "G": "B03_strict",
    "R": "B04_strict",
    "RE1": "B05_strict",
    "RE2": "B06_strict",
    "RE3": "B07_strict",
    "N": "B08_strict",
    "S1": "B11_strict",
    "S2": "B12_strict",
}

# S1 Bandnamen (VH/VV)
BAND_MAP_S1: Dict[str, str] = {
    "VV": "vv",  #'vv_speckle',
    "VH": "vh",  #'vh_speckle',
}

# ---------------------------------------------------------------------- S2 Functions ---------------------------------------------------------------------------


def get_s2_quality_masks(ds: xr.Dataset) -> xr.Dataset:
    """
    Creates binary physical quality masks and removes source classification bands.
    """

    # Check for hardware issues and general cloud mask noise
    is_hardware_ok = ~ds.SCL.isin(SCL_INVALID)
    is_cm_ok = ~ds.cloud_mask.isin(CM_CLOUDS)

    # Check for clouds using the Scene Classification Layer (SCL)
    is_scl_cloud_free = ~ds.SCL.isin(SCL_CLOUDS)

    # Generate binary masks (1 = Valid, 0 = Masked)
    # Basic: Hardware and Cloud Mask check
    ds["mask_phys_basic"] = (is_hardware_ok & is_cm_ok).astype("uint8")

    # Strict: Basic check + strict SCL cloud filtering
    ds["mask_phys_strict"] = (ds["mask_phys_basic"] & is_scl_cloud_free).astype("uint8")

    # --- SANITY CHECKS ---
    # 1. Binary check: Ensure values are strictly 0 or 1
    unique_vals = np.unique(ds["mask_phys_basic"].values)
    assert set(unique_vals).issubset(
        {0, 1}
    ), f"Basic Mask contains non-binary values: {unique_vals}!"

    # 2. Hierarchy invariant: Strict mask must be a subset of basic mask
    if int(ds["mask_phys_strict"].sum()) > int(ds["mask_phys_basic"].sum()):
        raise ValueError(
            "Logic Error: Strict mask has more valid pixels than basic mask!"
        )

    # --- CLEANUP ---
    ds = ds.drop_vars(["SCL", "cloud_mask"], errors="ignore")

    return ds


def get_vegetation_mask(ds: xr.Dataset) -> xr.Dataset:
    """
    Creates a binary vegetation mask based on ESA WorldCover classes
    and removes the source land cover band.
    """

    # Create mask: 1 for vegetation classes, 0 for everything else
    # We take the first (and usually only) timestamp of WorldCover
    mask = (
        ds.ESA_LC.isel(time_esa_worldcover=0).isin(VEGETATION_CLASSES).astype("uint8")
    )

    # Assign to dataset
    ds["is_veg"] = mask

    # --- SANITY CHECK ---
    # Ensure the mask is strictly binary (0 or 1)
    unique_vals = np.unique(ds["is_veg"].values)
    assert set(unique_vals).issubset(
        {0, 1}
    ), f"Vegetation mask contains non-binary values: {unique_vals}!"

    return ds


def clean_and_normalize_bands(ds: xr.Dataset) -> xr.Dataset:
    """
    Cleans B04 and B08 based on physical limits and masks,
    normalizes values to [0, 1], and removes raw source bands.
    """

    # 1. Get list of all bands
    all_b_bands = [v for v in ds.data_vars if v.startswith("B") and v[1:].isalnum()]

    # 2. Create validity mask: Reflectance must be within physical limits
    core_is_valid = (
        (ds.B04 > REFLECTANCE_MIN)
        & (ds.B04 <= REFLECTANCE_MAX)
        & (ds.B08 > REFLECTANCE_MIN)
        & (ds.B08 <= REFLECTANCE_MAX)
    )

    # 3. Update existing quality masks with physical validity
    ds["quality_mask_basic"] = (ds["mask_phys_basic"] & core_is_valid).astype("uint8")
    ds["quality_mask_strict"] = (ds["mask_phys_strict"] & core_is_valid).astype("uint8")
    del core_is_valid
    gc.collect()

    # 4. Apply masks and normalize to [0, 1] range
    for band in all_b_bands:
        # is_this_band_valid = (ds[band] > REFLECTANCE_MIN) & (
        #     ds[band] <= REFLECTANCE_MAX
        # )

        # num_invalid = (
        #     (~is_this_band_valid & ds.quality_mask_basic & ds.is_veg).sum().values
        # )
        # total_pixels = (ds.quality_mask_basic & ds.is_veg).sum().values
        # percent_affected = (num_invalid / total_pixels) * 100

        # print(f"--- {band} ---")
        # print(f"Invalid pixels: {num_invalid} / {total_pixels}")
        # print(f"Affected: {percent_affected:.4f}%")

        ds[f"{band}_normalized"] = (ds[band] / NORM_FACTOR).astype("float32")

    gc.collect()

    # --- CLEANUP ---
    # Drop original raw bands and intermediate masks to save memory and bands that will not be used
    # We keep the new masks
    # ds = ds.drop_vars(all_b_bands, errors='ignore')

    # ds = ds.drop_vars([ "mask_phys_basic", "mask_phys_strict"])

    return ds


def calculate_s2_index(ds: xr.Dataset, index_name: str) -> xr.DataArray:
    """
    Calculates spectral indices (e.g., NDVI) and verifies NaN consistency
    against input masks.
    """

    # 1. Select the correct band mapping based on prior cleaning steps
    bands_map = BAND_MAP_S2

    # Build parameters for spyndex computation
    params = {code: ds[band] for code, band in bands_map.items() if band in ds}

    # 2. Handle Special Cases like kNDVI
    if index_name == "kNDVI":
        params["kNN"] = 1.0
        # Kernel computation requires NIR (N) and Red (R)
        params["kNR"] = spyndex.computeKernel(
            kernel="RBF",
            params={
                "a": params["N"],
                "b": params["R"],
                "sigma": 0.5 * (params["N"] + params["R"]),
            },
        )

    # 3. Compute Index
    ds[index_name] = (
        spyndex.computeIndex(index=[index_name], params=params)
        .squeeze()
        .astype("float32")
    )

    # --- SANITY CHECK ---
    if "NDVI" in index_name:
        #  Check for outliers in valid pixels
        validate_index_against_masks(ds, index_name)

    return ds


def validate_index_against_masks(ds, index_var):
    for m_type in ["basic", "strict"]:
        mask_col = f"quality_mask_{m_type}"
        # Prüfe nur die als valide (1) markierten Pixel
        valid_data = ds[index_var].where(ds[mask_col] == 1)

        v_max = float(valid_data.max())
        v_min = float(valid_data.min())

        if v_max > 1.001 or v_min < -1.001:
            raise ValueError(
                f"Out-of-range {index_var} in {mask_col}: [{v_min:.4f}, {v_max:.4f}]"
            )
    print(f"✅ {index_var} is valid within both masks.")


def filter_static_vegetation_outliers(
    ds, index_name="NDVI", threshold_pct=0.75, time_dim="time_sentinel_2_l2a"
):
    """
    Identifies 'static bad pixels' that frequently show non-vegetation values
    and saves them in a permanent static mask.
    The static mask can then be used to refine vegetation masks.
    """
    da_val = ds[index_name]

    # 1. Use a baseline quality
    quality_mask = "quality_mask_strict"
    clear_veg_pixels = (ds.is_veg == 1) & (ds[quality_mask] == 1)

    # 2. Identify outliers (NDVI < 0 or > 0.95)
    is_invalid = ((da_val > 0.95) | (da_val < 0.0)) & clear_veg_pixels

    # 3. Calculate frequency of errors
    clear_veg_days = clear_veg_pixels.sum(dim=time_dim)
    error_count = is_invalid.sum(dim=time_dim)

    # Freq: percentage of 'clear' days that showed impossible NDVI
    freq = xr.where(clear_veg_days > 5, error_count / clear_veg_days, 0)

    # 4. Create the Static Bad Mask (2D)
    # True where the pixel is out of expected interval for more then 75%
    static_bad_mask = freq > threshold_pct

    # Create the named static mask: 1 = Keep, 0 = Static Outlier
    mask_name = f"{index_name}_valid_mask_static"
    ds[mask_name] = (~static_bad_mask).astype("uint8")

    # --- ASSERTION LOGIC ---
    # We check how many pixels in 'is_veg' are being "hit" by this new static mask
    # valid_before: All pixels currently marked as vegetation
    # pixels_to_remove: Pixels that are veg but now marked as static bad
    valid_before = int(ds.is_veg.sum())
    pixels_to_remove = int((ds.is_veg & static_bad_mask).sum())

    valid_after = int((ds.is_veg & ds[mask_name]).sum())

    # Check if the math adds up
    assert (
        valid_before - valid_after == pixels_to_remove
    ), f"Assertion failed! Before: {valid_before}, After: {valid_after}, Expected Removal: {pixels_to_remove}"

    # 5. Logging
    num_static_bad = int(static_bad_mask.sum())
    if num_static_bad > 0:
        print(
            f"📍 Created {mask_name}: {num_static_bad} pixels identified as permanent outliers."
        )

    return ds


def integrate_veg_and_wrongly_classified_mask(
    ds: xr.Dataset, index_name: str
) -> xr.Dataset:
    """
    Updates the vegetation mask by removing misclassified pixels
    and cleans up the temporary static mask.
    """
    mask_name = f"{index_name}_valid_mask_static"
    static_mask_2d = ds[mask_name]

    # Use bitwise multiplication to keep it uint8 and ensure correct broadcasting
    ds["is_veg"] = (ds["is_veg"] & static_mask_2d).astype("uint8")

    # Cleanup: The static mask's information is now baked into 'is_veg'
    ds = ds.drop_vars(mask_name, errors="ignore")

    return ds


# def final_data_integrity_check(ds, index_name="NDVI"):
#     """
#     Ultimativer Sanity Check vor dem Modell-Training.
#     Prüft auf NaNs, Wertebereiche, Datentypen und Masken-Logik.
#     """
#     print(f"🚀 Starte finalen Integritäts-Check für {index_name}...")

#     # 1. Check: Absolute NaN-Freiheit
#     # Ein einziges NaN kann den Gradienten im Training auf 'NaN' setzen (exploding gradients).
#     nan_count = int(ds[index_name].isnull().sum())
#     assert nan_count == 0, f"❌ CRITICAL: {index_name} enthält {nan_count} NaNs!"

#     # 2. Check: Wertebereich für Vegetation (Intervall [0, 1])
#     # Wir prüfen nur Pixel, die laut deiner Master-Maske als gültige Vegetation gelten.
#     # Wir nutzen ein kleines Delta (1e-6) für Float-Präzision.
#     veg_data = ds[index_name].where(ds.vegetation_mask_final == 1)

#     # Wir ignorieren hier die Nullen der optischen Maske (Wolken)
#     valid_veg_data = veg_data.where(ds.optical_mask_final_strict == 1)

#     max_val = float(valid_veg_data.max())
#     min_val = float(valid_veg_data.min())

#     assert max_val <= 1.000001, f"❌ VALUE ERROR: Max Wert {max_val} in {index_name} übersteigt 1.0!"
#     assert min_val >= -0.000001, f"❌ VALUE ERROR: Min Wert {min_val} in {index_name} ist unter 0.0!"

#     # 3. Check: Datentypen (Wichtig für GPU-Memory)
#     # Masken sollten uint8 sein, Messwerte float32.
#     assert ds[index_name].dtype == 'float32', f"Typ-Fehler: {index_name} sollte float32 sein, ist {ds[index_name].dtype}"
#     assert ds.vegetation_mask_final.dtype == 'uint8', "Typ-Fehler: Masken sollten uint8 sein!"

#     # 4. Check: Räumliche Konsistenz (Static Mask)
#     # Die Vegetation-Maske darf über die Zeit nicht variieren (muss statisch sein).
#     # Wir prüfen, ob die Standardabweichung über die Zeit Null ist.
#     flicker = ds.vegetation_mask_final.std(dim="time_sentinel_2_l2a").sum()
#     assert flicker == 0, "❌ LOGIC ERROR: 'vegetation_mask_final' variiert über die Zeit!"

#     # 5. Check: "Zero-Leckage"
#     # Wenn beide Masken 1 sind, darf der Wert im NDVI nicht 'NaN' gewesen sein (was wir zu 0 gefüllt haben).
#     # Das stellt sicher, dass wir keine 'Fake-Nullen' als echte Daten verkaufen.
#     # Wir nutzen hierfür deine valid_NDVI_mask von vorhin.
#     leakage = ((ds.optical_mask_final_strict == 1) & (ds.vegetation_mask_final == 1) & (ds.valid_NDVI_mask == 0)).sum()
#     assert int(leakage) == 0, f"❌ MASK ERROR: {int(leakage)} ungültige Berechnungen als 'Gültig' markiert!"

#     print("✅ FINALER CHECK BESTANDEN: Der Cube ist bereit für das ConvLSTM.")
#     return True
