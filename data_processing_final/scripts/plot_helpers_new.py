import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np


def find_cloud_free_indices(ds, threshold=0.99):
    """
    Finds indices of Sentinel 2 timesteps, that are almost cloud free.
    Based on the "mask_phys_strict" mask
    """
    # 1. Calculates mean over x and y per timestep
    quality_means = ds.mask_phys_strict.mean(dim=("x", "y"))

    # 2. Find indices which meet criteria
    indices = np.where(quality_means > threshold)[0]

    # 3. Calculate amount of indices
    count = len(indices)

    print(f"Found cloudfree timesteps (> {threshold*100}%): {count}")

    return indices


def plot_acquisition_timelines(ds):
    # 1. Zeitstempel extrahieren und in Pandas Datetime umwandeln
    t1 = pd.to_datetime(ds.time_sentinel_1_rtc.values)
    t2 = pd.to_datetime(ds.time_sentinel_2_l2a.values)

    fig, ax = plt.subplots(figsize=(15, 4))

    # 2. Sentinel-1 (Radar) plotten - Blaue Punkte/Linien
    ax.vlines(
        t1,
        0.8,
        1.2,
        color="royalblue",
        alpha=0.6,
        label="Sentinel-1 (Radar)",
        linewidth=1,
    )
    ax.scatter(t1, [1.0] * len(t1), color="royalblue", s=10, alpha=0.5)

    # 3. Sentinel-2 (Optisch) plotten - Grüne Punkte/Linien
    ax.vlines(
        t2,
        1.8,
        2.2,
        color="forestgreen",
        alpha=0.6,
        label="Sentinel-2 (Optisch)",
        linewidth=1,
    )
    ax.scatter(t2, [2.0] * len(t2), color="forestgreen", s=10, alpha=0.5)

    # 4. Optik & Achsen-Styling
    ax.set_ylim(0.5, 2.5)
    ax.set_yticks([1.0, 2.0])
    ax.set_yticklabels(["Sentinel-1", "Sentinel-2"], fontsize=12, fontweight="bold")

    # X-Achse formatieren
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Strich pro Monat
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))  # 'Jan 2024'
    plt.xticks(rotation=45)

    ax.set_title(
        "Vergleich der Aufnahme-Zeitpunkte (Acquisition Timelines)", fontsize=14, pad=15
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.legend(loc="upper right", frameon=True, shadow=True)
    plt.tight_layout()
    plt.show()


def plot_rgb(
    ds: xr.Dataset,
    timestep: int,
    time_dim: str = "time_sentinel_2_l2a",
    bands: Tuple[str, str, str] = ("B04", "B03", "B02"),
    stretch_pct: Tuple[int, int] = (2, 98),
    figsize: Tuple[int, int] = (10, 6),
    cloud_comp: bool = False,
    title_prefix: Optional[str] = "Timestep",
) -> Union[None, str]:
    """
    Plots a static true-color (RGB) composite for a specific timestep
    from a Sentinel-2 xarray.Dataset. Optionally plots a cloud mask comparison.

    Parameters
    ----------
    ds : xarray.Dataset
        Must have a time coordinate `time_dim` and DataArrays for the three bands.
    timestep : int
        The time index to plot (must be valid for ds[time_dim]).
    time_dim : str
        Name of the time dimension (default 'time_sentinel_2_l2a').
    bands : tuple of str
        The (red, green, blue) band names in ds (default ('B04','B03','B02')).
    stretch_pct : tuple of int
        Percentiles for contrast stretch (default 2nd to 98th).
    figsize : tuple
        Matplotlib figure size.
    cloud_comp : bool
        If True, plots the RGB image and the cloud mask comparison side-by-side.
    title_prefix : str or None
        Prefix for the figure title (used by interactive function).

    Returns
    -------
    None or str
        None on success, or an error message string if the timestep is invalid.
    """

    # Helper function to percentile‐stretch one (y,x) slice
    def stretch_arr(band):
        arr = band.compute().astype("float32")
        p2, p98 = np.percentile(arr, stretch_pct)
        return np.clip((arr - p2) / (p98 - p2), 0, 1)

    # 1. Input Validation
    if not 0 <= timestep < len(ds.coords[time_dim]):
        return f"Timeindex {timestep} out of range for dimension '{time_dim}' (size {len(ds.coords[time_dim])})."

    # 2. Subset data for the given timestep
    subset = ds.isel({time_dim: timestep})
    time_value_raw = subset[time_dim].item()

    time_formatted = pd.to_datetime(time_value_raw).strftime("%Y-%m-%d")

    try:
        r, g, b = subset[bands[0]], subset[bands[1]], subset[bands[2]]
    except KeyError as e:
        return f"One or more bands not found in dataset: {e}"

    # 3. Create RGB array (stretched)
    rgb = np.dstack(
        [stretch_arr(r).values, stretch_arr(g).values, stretch_arr(b).values]
    )

    # 4. Plotting Logic

    if cloud_comp:
        # --- Cloud Comparison Logic ---
        try:
            scl = subset["SCL"]
            cloud_mask = subset["cloud_mask"]
        except KeyError as e:
            return f"Required variable for cloud comparison missing: {e}. Set cloud_comp=False or check dataset."

        # Binary masks
        is_cloud_scl = scl.isin([3, 8, 9, 10])  # cloud classes SCL
        is_external_cloud = cloud_mask.isin([1, 2, 3])  # cloud classes cloud_mask

        # Calculate difference and agreement masks for the overlay
        scl_only_mask = (
            (is_cloud_scl & ~is_external_cloud).astype(float).values.squeeze()
        )
        external_only_mask = (
            (is_external_cloud & ~is_cloud_scl).astype(float).values.squeeze()
        )
        agreement_mask = (
            (is_external_cloud & is_cloud_scl).astype(float).values.squeeze()
        )

        # Create the RGB overlay (Red=SCL Only, Green=External Only, Blue=Agreement)
        overlay = np.dstack(
            [
                scl_only_mask,
                external_only_mask,
                agreement_mask,
            ]
        )

        # Plotting side by side (2 plots)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

        # Left Plot: RGB ONLY
        ax1.imshow(rgb)
        ax1.set_title("True Color (RGB) Image", fontsize=12)
        ax1.axis("off")

        # Right Plot: RGB + Mask Comparison Overlay
        ax2.imshow(rgb)
        ax2.imshow(overlay, alpha=0.5)

        ax2.set_title(
            "Mask Comparison\n" "Red: SCL Only | Green: External Only | Blue: Both",
            fontsize=12,
        )
        ax2.axis("off")

        fig.suptitle(f"{title_prefix}: {time_formatted}", fontsize=14, y=1.02)

    else:
        # Plotting RGB ONLY (1 plot)
        fig, ax1 = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

        ax1.imshow(rgb)
        ax1.set_title(
            f"True Color (RGB) Image\n{title_prefix}: {time_formatted}", fontsize=14
        )
        ax1.axis("off")

    plt.show()
    return None


def plot_statistical_outliers(ds, index_name, time_index):
    """
    Plots RGB and highlights only the pixels that are permanently
    masked due to frequent outliers.
    """

    # Identify the "Bad Seeds" (Static mask)
    static_bad_pixels = ds[f"{index_name}_valid_mask_static"] == 0

    # 2. Prepare RGB for a sample timestep (e.g., middle of the series)
    subset = ds.isel(time_sentinel_2_l2a=time_index)

    def stretch(band):
        arr = band.compute()
        p2, p98 = np.percentile(arr, (2, 98))
        return np.clip((arr - p2) / (p98 - p2), 0, 1)

    rgb = np.dstack([stretch(subset.B04), stretch(subset.B03), stretch(subset.B02)])

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # Left: RGB
    ax1.imshow(rgb)
    ax1.set_title(
        f"Reference RGB\n({pd.to_datetime(subset.time_sentinel_2_l2a.values).date()})"
    )
    ax1.axis("off")

    # Right: The Outliers
    # Show the mask in bright red on a black background for maximum contrast
    outlier_map = (
        static_bad_pixels.isel(time_sentinel_2_l2a=0).astype(float).values.squeeze()
    )
    ax2.imshow(outlier_map, cmap="Reds")
    ax2.set_title(f"Permanently Masked Pixels\n(Freq > {0.75*100}%)")
    ax2.axis("off")

    plt.show()

    print(f"Total static outliers found: {int(static_bad_pixels.sum())} pixels")
