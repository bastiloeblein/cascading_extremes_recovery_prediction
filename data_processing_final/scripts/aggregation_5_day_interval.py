import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
import pandas as pd
import numpy as np


def align_all_to_5d(ds, masking_type):

    # 1. Define which mask to use
    if masking_type == "basic":
        mask = ds.quality_mask_basic
    elif masking_type == "strict":
        mask = ds.quality_mask_strict

    # 2. Define common date as base for resampling
    start_date = ds.time_sentinel_2_l2a.min().values

    # 2. Identify all variables with S1 and S2temporal dimension
    s1_vars = [v for v in ds.data_vars if "time_sentinel_1_rtc" in ds[v].dims]
    s2_vars = [
        v
        for v in ds.data_vars
        if "time_sentinel_2_l2a" in ds[v].dims and "quality_mask" not in v
    ]  # Exclude masks

    # 3. Apply masking: Set all values to NaN which have 0 in mask
    assert (
        ds[f"NDVI_{masking_type}"].where(mask == 0).notnull().sum().values == 0
    ), "Masking failed: Non-NaN values found in clouded areas!"  # costly
    ds[s2_vars] = ds[s2_vars].where(mask == 1, np.nan)

    # 2. Resampling S2 data (optical):
    s2_res = (
        ds[s2_vars]
        .resample(time_sentinel_2_l2a="5D", label="right", origin=start_date)
        .median()
    )

    # 3. Resampling S1 data (radar):
    s1_res = (
        ds[s1_vars]
        .resample(time_sentinel_1_rtc="5D", label="right", origin=start_date)
        .median()
    )
    s1_res = s1_res.rename({"time_sentinel_1_rtc": "time_sentinel_2_l2a"})

    # 5. Get information on aggregation
    get_aggregation_information(
        ds, s2_vars, s1_vars, s1_res, s2_res, masking_type, True
    )

    # 6. Merge datasets
    # Optional: Think about it: This forces S1 to have the exact same time points as S2 (if there are S1 observations before or after S2 period, they get lost)
    s1_res = s1_res.reindex(time_sentinel_2_l2a=s2_res.time_sentinel_2_l2a, method=None)
    combined = xr.merge([s2_res, s1_res])

    # Create mask for S2 and S1
    combined["s2_final_mask"] = (
        combined[f"NDVI_{masking_type}"].notnull().astype("uint8")
    )
    combined["s1_final_mask"] = combined["vh"].notnull().astype("uint8")

    # Some assertions
    assert combined.s2_final_mask.max() <= 1 and combined.s2_final_mask.min() >= 0
    assert "time_sentinel_2_l2a" in combined.dims
    assert (
        combined.s2_final_mask.shape[0] == combined.s1_final_mask.shape[0]
    ), "Time dimension mismatch after merge!"

    return combined


def get_aggregation_information(
    ds, s2_vars, s1_vars, s1_res, s2_res, masking_type, plotting=False
):

    # Get start date
    start_date = ds.time_sentinel_2_l2a.min().values

    # Count number of images per 5-day bin
    s2_counts = (
        ds[s2_vars[0]]
        .resample(time_sentinel_2_l2a="5D", label="right", origin=start_date)
        .count(dim="time_sentinel_2_l2a")
    )
    s1_counts = (
        ds[s1_vars[0]]
        .resample(time_sentinel_1_rtc="5D", label="right", origin=start_date)
        .count(dim="time_sentinel_1_rtc")
    )

    # Get max number of images per bin over all pixels
    s2_images_per_bin = s2_counts.max(dim=["x", "y"])
    s1_images_per_bin = s1_counts.max(dim=["x", "y"])

    # Check if bin is nan
    is_s2_bin_empty = s2_images_per_bin.isnull()
    is_s1_bin_empty = s1_images_per_bin.isnull()

    # Count number of true values
    s2_empty_bins_count = int(is_s2_bin_empty.sum())
    s1_empty_bins_count = int(is_s1_bin_empty.sum())

    print("--- Binning Analysis (5-Day-Intervals) ---")
    print(
        f"Sentinel-2: {s2_empty_bins_count} empty bins of {len(s2_res.time_sentinel_2_l2a)} in total. Original Data: {len(ds.time_sentinel_2_l2a)} timesteps."
    )
    print(
        f"Sentinel-1: {s1_empty_bins_count} empty bins of {len(s1_res.time_sentinel_2_l2a)} in total. Original Data: {len(ds.time_sentinel_1_rtc)} timesteps."
    )

    occupied_bins_s2 = len(s2_res.time_sentinel_2_l2a) - s2_empty_bins_count
    assert occupied_bins_s2 <= len(
        ds.time_sentinel_2_l2a
    ), "Logic Error: More occupied bins than original images!"
    assert occupied_bins_s2 > 0, "Data Loss: All S2 bins are empty after masking!"

    if plotting:
        plot_acquisition_and_bins(ds, s1_counts, s2_counts, masking_type)

    return None


def plot_acquisition_and_bins(ds, s1_counts, s2_counts, masking_type):
    # 1. Extract raw timestamps
    t1_raw = pd.to_datetime(ds.time_sentinel_1_rtc.values)
    t2_raw = pd.to_datetime(ds.time_sentinel_2_l2a.values)

    # --- NEW: Identify empty original timestamps (All NaN across x and y) ---
    # Count only counts not-nan values across spatial dimension
    s2_valid_pixel_count = ds[f"NDVI_{masking_type}"].count(dim=["x", "y"]).values
    s1_valid_pixel_count = ds["vh"].count(dim=["x", "y"]).values

    # Colors: Green/Blue if valid, Red if all NaN
    s2_colors = ["forestgreen" if c > 0 else "red" for c in s2_valid_pixel_count]
    s1_colors = ["royalblue" if c > 0 else "red" for c in s1_valid_pixel_count]

    # Bin-Zeitachsen (resampled axes)
    t1_bins = pd.to_datetime(s1_counts.time_sentinel_1_rtc.values)
    t2_bins = pd.to_datetime(s2_counts.time_sentinel_2_l2a.values)

    # Counts for bins
    c1_vals = s1_counts.max(dim=["x", "y"]).fillna(0).values
    c2_vals = s2_counts.max(dim=["x", "y"]).fillna(0).values

    fig, ax = plt.subplots(figsize=(16, 8), sharex=True)

    # --- LEVEL 1: S2 Original Data (with Red for empty) ---
    ax.scatter(
        t2_raw,
        [4.0] * len(t2_raw),
        color=s2_colors,
        s=30,
        label="S2 Original Data",
        zorder=3,
    )
    ax.vlines(t2_raw, 3.8, 4.2, color="gray", alpha=0.2, linewidth=1)

    # --- LEVEL 2: S2 Bins ---
    ax.bar(
        t2_bins,
        c2_vals * 0.4,
        width=5,
        bottom=2.8,
        color="limegreen",
        alpha=0.4,
        label="S2 Bins (Obs count)",
        align="edge",
        edgecolor="green",
    )
    for t, val in zip(t2_bins, c2_vals):
        if val > 0:
            ax.text(
                t + pd.Timedelta(days=2.5),
                3.3,
                str(int(val)),
                color="darkgreen",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    # --- LEVEL 3: S1 Original Data (with Red for empty) ---
    ax.scatter(
        t1_raw,
        [1.5] * len(t1_raw),
        color=s1_colors,
        s=30,
        label="S1 Original Data",
        zorder=3,
    )
    ax.vlines(t1_raw, 1.3, 1.7, color="gray", alpha=0.2, linewidth=1)

    # --- LEVEL 4: S1 Bins ---
    ax.bar(
        t1_bins,
        c1_vals * 0.4,
        width=5,
        bottom=0.3,
        color="cornflowerblue",
        alpha=0.4,
        label="S1 Bins (Obs count)",
        align="edge",
        edgecolor="blue",
    )
    for t, va in zip(t1_bins, c1_vals):
        if va > 0:
            ax.text(
                t + pd.Timedelta(days=2.5),
                0.8,
                str(int(va)),
                color="darkblue",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    # --- STYLING ---
    ax.set_yticks([0.5, 1.5, 3.0, 4.0])
    ax.set_yticklabels(
        ["S1 Bins", "S1 Original Data", "S2 Bins", "S2 Original Data"],
        fontsize=11,
        fontweight="bold",
    )

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)

    # Add a custom legend entry for Red dots
    from matplotlib.lines import Line2D

    custom_lines = [
        Line2D([0], [0], color="forestgreen", marker="o", linestyle=""),
        Line2D([0], [0], color="royalblue", marker="o", linestyle=""),
        Line2D([0], [0], color="red", marker="o", linestyle=""),
    ]
    ax.legend(
        custom_lines,
        ["S2 Valid", "S1 Valid", "Empty (All NaN)"],
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
    )

    ax.set_title(
        "Data Coverage: Original Data (Valid vs. Empty) & 5-Day Bins",
        fontsize=15,
        pad=20,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.axhline(2.25, color="black", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    plt.show()
