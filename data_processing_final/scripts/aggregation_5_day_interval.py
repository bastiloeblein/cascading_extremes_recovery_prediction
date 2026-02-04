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
    start_date = pd.to_datetime(ds.precip_end_date)

    # 2. Identify all variables with S1 and S2 temporal dimension plus static variables
    s1_vars = [v for v in ds.data_vars if "time_sentinel_1_rtc" in ds[v].dims]
    s2_vars = [
        v
        for v in ds.data_vars
        if "time_sentinel_2_l2a" in ds[v].dims and "quality_mask" not in v
    ]  # Exclude masks
    static_features = ds[["ESA_LC", "COP_DEM", "is_veg"]]

    # 3. Apply masking: Set all values to NaN which have 0 in mask
    assert (
        ds[f"NDVI_{masking_type}"].where(mask == 0).notnull().sum().values == 0
    ), "Masking failed: Non-NaN values found in clouded areas!"  # costly
    ds[s2_vars] = ds[s2_vars].where(mask == 1, np.nan)

    # 2. Resampling S2 data (optical):
    ## Think about selecting the image with the lowest cloud cover within the bin!
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
    plot_full_acquisition_analysis(ds, masking_type="basic")

    # 6. Merge datasets
    # Optional: Think about it: This forces S1 to have the exact same time points as S2 (if there are S1 observations before or after S2 period, they get lost)
    s1_res = s1_res.reindex(time_sentinel_2_l2a=s2_res.time_sentinel_2_l2a, method=None)
    combined = xr.merge(
        [s2_res, s1_res, static_features]
    )  # Add static features (later on maybe think about broadcasting them to time_sentinel_2 as well)

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


def plot_full_acquisition_analysis(ds, masking_type="basic"):
    # 1. Timestamps & Validity
    t1_raw = pd.to_datetime(ds.time_sentinel_1_rtc.values)
    t2_raw = pd.to_datetime(ds.time_sentinel_2_l2a.values)
    ref_date = pd.to_datetime(ds.precip_end_date)

    # Check validity (pixel counts)
    s1_valid_counts = ds["vh"].count(dim=["x", "y"]).values
    s2_valid_counts = ds[f"NDVI_{masking_type}"].count(dim=["x", "y"]).values

    # 2. Dynamic Binning
    all_times = np.concatenate([t1_raw, t2_raw])
    start_obs, end_obs = all_times.min(), all_times.max()

    bins_before = pd.date_range(
        end=ref_date, start=start_obs - pd.Timedelta(days=5), freq="5D"
    )
    bins_after = pd.date_range(
        start=ref_date, end=end_obs + pd.Timedelta(days=5), freq="5D"
    )
    bins = sorted(list(set(bins_before.union(bins_after))))
    bin_centers = [bins[i] + (bins[i + 1] - bins[i]) / 2 for i in range(len(bins) - 1)]
    num_bins = len(bins) - 1

    # Stats counters
    s1_stats = {"no_obs": 0, "only_nan": 0}
    s2_stats = {"no_obs": 0, "only_nan": 0}

    fig, ax = plt.subplots(figsize=(22, 10))

    # --- Plotting Original Data ---
    ax.scatter(
        t2_raw,
        [2.0] * len(t2_raw),
        color=["forestgreen" if c > 0 else "red" for c in s2_valid_counts],
        s=45,
        zorder=4,
    )
    ax.scatter(
        t1_raw,
        [1.0] * len(t1_raw),
        color=["royalblue" if c > 0 else "red" for c in s1_valid_counts],
        s=45,
        zorder=4,
    )

    # Bin Dividers
    ax.vlines(bins, 2.7, 3.6, color="red", alpha=0.3, linewidth=1.2)

    s1_is_valid_any = s1_valid_counts > 0
    s2_is_valid_any = s2_valid_counts > 0

    for i in range(num_bins):
        m1_idx = np.where((t1_raw >= bins[i]) & (t1_raw < bins[i + 1]))[0]
        m2_idx = np.where((t2_raw >= bins[i]) & (t2_raw < bins[i + 1]))[0]

        c1, c2 = len(m1_idx), len(m2_idx)

        # Superschneller Check auf dem vorbereiteten Boolean-Array
        valid1 = s1_is_valid_any[m1_idx].any() if c1 > 0 else False
        valid2 = s2_is_valid_any[m2_idx].any() if c2 > 0 else False

        # Statistics
        if c1 == 0:
            s1_stats["no_obs"] += 1
        elif not valid1:
            s1_stats["only_nan"] += 1

        if c2 == 0:
            s2_stats["no_obs"] += 1
        elif not valid2:
            s2_stats["only_nan"] += 1

        center = bin_centers[i]
        if c2 > 0:
            color = "forestgreen" if valid2 else "red"
            ax.scatter(center, 3.3, marker="o", color=color, s=70, alpha=0.7)
            ax.text(
                center,
                3.45,
                str(c2),
                color=color,
                ha="center",
                fontweight="bold",
                fontsize=11,
            )

        if c1 > 0:
            color = "royalblue" if valid1 else "red"
            ax.scatter(center, 3.0, marker="x", color=color, s=70, alpha=0.7)
            ax.text(
                center,
                2.75,
                str(c1),
                color=color,
                ha="center",
                fontweight="bold",
                fontsize=11,
            )

    # --- Printing detailed Report ---
    total_empty_s1 = s1_stats["no_obs"] + s1_stats["only_nan"]
    total_empty_s2 = s2_stats["no_obs"] + s2_stats["only_nan"]

    print("\n📊 DETAILED BINNING REPORT (5-Day Intervals)")
    print(f"Total Intervals: {num_bins}")
    print(
        f"{'Platform':<15} | {'Missing Obs':<15} | {'Only NaN':<15} | {'Total Unusable'}"
    )
    print("-" * 70)
    print(
        f"{'Sentinel-1':<15} | {s1_stats['no_obs']:<15} | {s1_stats['only_nan']:<15} | {total_empty_s1} ({total_empty_s1/num_bins:.1%})"
    )
    print(
        f"{'Sentinel-2':<15} | {s2_stats['no_obs']:<15} | {s2_stats['only_nan']:<15} | {total_empty_s2} ({total_empty_s2/num_bins:.1%})"
    )
    print("-" * 70 + "\n")

    # Final Axis Styling
    ax.set_ylim(0.5, 4.2)
    ax.set_yticks([1.0, 2.0, 3.15])
    ax.set_yticklabels(
        [
            f"S1 Original\n(n={len(t1_raw)})",
            f"S2 Original\n(n={len(t2_raw)})",
            "5-Day Bins\n(Counts)",
        ],
        fontweight="bold",
        fontsize=12,
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45)
    ax.axvline(ref_date, color="black", linewidth=3, zorder=5)
    ax.text(
        ref_date,
        4.0,
        "EVENT",
        color="white",
        ha="center",
        fontweight="bold",
        bbox=dict(facecolor="black", boxstyle="round"),
    )

    plt.tight_layout()
    plt.show()
