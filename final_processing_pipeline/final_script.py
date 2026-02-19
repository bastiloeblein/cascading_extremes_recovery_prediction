import os
import sys
import gc
import base64
import warnings
import io
import pandas as pd
import numpy as np
import json
import xarray as xr
import matplotlib.pyplot as plt
from scripts.sentinel_1_processing import (
    find_global_veg_clipping_values,
    clip_s1_data,
    apply_lee_to_ds,
    normalize_s1_vars,
)  # , calculate_SAR_index #, aggregate_s1_causal_nearest
from scripts.sentinel_2_processing import (
    get_s2_quality_masks,
    get_vegetation_mask,
    apply_masking,
    clean_and_normalize_bands,
    report_permanent_nans_for_var,
    calculate_s2_index,
    filter_static_vegetation_outliers,
    integrate_veg_and_wrongly_classified_mask,
)
from scripts.plot_helpers_new import (
    find_cloud_free_indices,
    plot_statistical_outliers,
    plot_acquisition_timelines_filtered,
)
from scripts.era_5_processing import (
    subset_era5_spatial,
    subset_era5_time,
    aggregate_era5_metrics_new,
    check_time_alignment,
)  # , create_uniform_era5_features, verify_era5_alignment
from scripts.cube_processing import add_event_metadata
from scripts.aggregation_5_day_interval import align_all_to_5d
from scripts.normalize_and_clip import (
    normalize_dem,
    calculate_global_era5_stats,
    normalize_era5,
)


# Ignore warnings
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in divide"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in cast"
)
warnings.filterwarnings("ignore", category=xr.SerializationWarning)


# --- 2. HTML REPORT FUNCTION ---
def create_html_report(info_dir, cube_id, report_sequence):
    html_start = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Report {cube_id}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f0f2f5; }}
            .container {{ max-width: 1000px; margin: auto; background: white; padding: 20px; border-radius: 8px; }}
            pre {{ background: #222; color: #fff; padding: 10px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; }}
            .plot {{ text-align: center; margin: 20px 0; }}
            img {{ max-width: 100%; border: 1px solid #ccc; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Cube Report: {cube_id}</h1>
    """
    html_end = "</div></body></html>"

    content = ""
    for entry_type, value in report_sequence:
        if entry_type == "text":
            if value.strip():
                content += f"<pre>{value.strip()}</pre>"
        elif entry_type == "plot_b64":
            # Hier wird der Base64-String direkt eingebettet
            content += (
                f'<div class="plot"><img src="data:image/png;base64,{value}"></div>'
            )
        elif entry_type == "html_raw":
            # Fügt den IFrame direkt ein
            content += f'<div class="map-container">{value}</div>'

    with open(os.path.join(info_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_start + content + html_end)


def process_era5(cubes, era5_cubes, output_dir):

    era5_dir = os.path.join(output_dir, "era5")
    stats_path = os.path.join(output_dir, "global_era5_stats.json")

    # --- PHASE 1: Schneller Durchlauf für ERA5 Stats ---
    print("Pre-processing ERA5 to get global stats...")
    all_era5_series = {}
    os.makedirs(era5_dir, exist_ok=True)

    for key, ds_target in cubes.items():
        cube_features = []

        for i, era5_cube_raw in enumerate(era5_cubes):
            print(f"#### Retrieving ERA5 data for cube {key} ###")
            # 1. Temporal subset
            tmp = subset_era5_time(era5_cube_raw, ds_target)

            if tmp is None:
                print("❌ aggregate_era5_metrics_new hat None zurückgegeben!")

            # 2. Spatial subset
            tmp, _ = subset_era5_spatial(tmp, ds_target, plot_check=False)

            # 3. Aggregation to target temporal resolution
            vars_list = list(tmp.data_vars)
            tmp = aggregate_era5_metrics_new(tmp, ds_target, vars_list)

            # 3.1 Sanity check
            assert len(ds_target.time_sentinel_2_l2a) == len(
                tmp.time_sentinel_2_l2a
            ), f"Zeit-Längen-Mismatch! S2: {len(ds_target.time_sentinel_2_l2a)}, ERA5: {len(tmp.time_sentinel_2_l2a)}"
            assert ds_target.time_sentinel_2_l2a.equals(
                tmp.time_sentinel_2_l2a
            ), f"Zeit-Koordinaten-Mismatch in Cube {key}!"

            check_time_alignment(ds_target, tmp, "time_sentinel_2_l2a")

            # 4. Reduce dimensions to 1 (only time_sentinel_2_l2a)
            tmp = tmp.squeeze(dim=["latitude", "longitude"], drop=True)
            cube_features.append(tmp)

        # Merge all ERA5 vars for this cube
        era5_merged = xr.merge(cube_features)

        # Save ERA5 data separately for cube
        era5_path = f"{era5_dir}/{key}_era5.zarr"
        era5_merged.to_zarr(era5_path, mode="w", consolidated=True)

        # Add to dict for global statistic
        all_era5_series[key] = era5_merged

    if not all_era5_series:
        raise ValueError(
            "No ERA5 data was processed. Check input cubes and time ranges."
        )

    # 5. Calculate global vars
    first_key = list(all_era5_series.keys())[0]
    era5_stats_vars = list(all_era5_series[first_key].data_vars)
    global_era5_stats = calculate_global_era5_stats(all_era5_series, era5_stats_vars)

    # Save stats as json
    save_stats_to_json(global_era5_stats, stats_path)

    print("✅ Global ERA5 stats calculated and ERA5 cubes saved.")

    # Cleanup
    del all_era5_series
    gc.collect()

    return global_era5_stats


# --- 2. DIE HAUPTSCHLEIFE ---


def run_processing_pipeline(cubes, era5_cubes, output_dir, info_base="processing_info"):
    """
    Main Orchestrator for the Satellite Data Processing Pipeline.
    Processes each cube 1-by-1 to minimize memory footprint.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(info_base, exist_ok=True)

    print("=" * 60)
    print("PRE-PROCESSING GLOBAL Sentinel 1 PARAMETERS")
    print("=" * 60)

    # --- STAGE 0: S1 GLOBAL PARAMETERS ---
    s1_stats_path = os.path.join(output_dir, "global_s1_stats.json")

    # 0.1 Prepare Sentinel-1 Global Clipping Values
    if os.path.exists(s1_stats_path):
        print(f"-> Load existing S1 Stats from {s1_stats_path}")
        global_vv_max, global_vh_max = load_s1_stats(s1_stats_path)
    else:
        print("-> Calculate S1 percentiles (First run)...")
        global_vv_max, global_vh_max = find_global_veg_clipping_values(cubes)

        save_s1_stats(global_vv_max, global_vh_max, s1_stats_path)

    print("Global percentil values for all training cubes:\n")
    print("Global vv max: ", global_vv_max)
    print("Global vh max: ", global_vh_max)

    # 0.2 Prepare Global ERA5 Stats (Phase 1)
    # stats_path = "../training_cubes_test/global_era5_stats.json"
    stats_path = os.path.join(output_dir, "global_era5_stats.json")
    if os.path.exists(stats_path):
        print(f"-> Loading existing ERA5 global stats from {stats_path}")
        global_era5_stats = load_global_stats(stats_path)
    else:
        print("-> ERA5 stats not found. Starting Phase 1 (Global Extraction)...")
        global_era5_stats = process_era5(cubes, era5_cubes, output_dir)

    print("\n" + "=" * 60)
    print(f"STARTING CUBE PROCESSING LOOP ({len(cubes)} cubes found)")
    print("=" * 60)

    cube_keys = list(cubes.keys())
    for n, key in enumerate(cube_keys):
        save_path = os.path.join(output_dir, f"{key}.zarr")

        # --- CHECKPOINT: Skip if already done ---
        if os.path.exists(save_path):
            sys.__stdout__.write(f"⏭️  Skipping {key}: Already processed.\n")
            # Wir löschen ihn trotzdem aus dem Dict, um RAM zu sparen!
            cubes.pop(key)
            continue

        # --- MEMORY OPTIMIZATION: Extract and remove from dict ---
        ds = cubes.pop(key)
        cube_id = key

        info_dir = os.path.join(info_base, f"{n:03d}_{cube_id}")
        os.makedirs(info_dir, exist_ok=True)

        stdout_buffer = io.StringIO()
        sys.stdout = stdout_buffer
        report_sequence = []

        try:
            print(
                f"{'#'*10} PROCESSING CUBE: {cube_id} ({n+1}/{len(cubes)}) {'#'*10}\n"
            )

            # --- STAGE 1: SENTINEL-2 MASKING & QUALITY CONTROL ---
            print("Step 1: Applying Quality and Vegetation Masks...")
            ds = get_s2_quality_masks(ds)
            ds = get_vegetation_mask(ds)
            ds = apply_masking(ds)
            report_permanent_nans_for_var(ds, "B01", "time_sentinel_2_l2a")

            # --- STAGE 2: BAND NORMALIZATION ---
            print("Step 2: Cleaning and Normalizing Spectral Bands...")
            ds = clean_and_normalize_bands(ds)
            report_permanent_nans_for_var(ds, "B01_normalized", "time_sentinel_2_l2a")

            ## --- STAGE 3: INDEX CALCULATION ---
            print("Step 3: Calculating Vegetation and Water Indices...")
            for idx in ["NDVI", "kNDVI", "NIRv", "NDMI", "NDWI", "IRECI", "CIRE"]:
                ds = calculate_s2_index(ds, idx)
                if idx == "kNDVI":
                    report_permanent_nans_for_var(ds, idx, "time_sentinel_2_l2a")

            # --- STAGE 4: OUTLIER FILTERING & SPATIAL VALIDATION ---
            print("Step 4: Filtering Static Outliers...")
            ds = filter_static_vegetation_outliers(ds)
            indices = find_cloud_free_indices(ds)
            if len(indices) > 0:
                fig = plot_statistical_outliers(ds, indices[0], False)
                save_plot_to_report(fig, report_sequence, stdout_buffer)

            # --- STAGE 5: DATASET CLEANUP & INTEGRATION ---
            print("Step 5: Final Mask Integration and Dropping Raw Bands...")
            ds = integrate_veg_and_wrongly_classified_mask(ds)
            all_b_bands = [
                v for v in ds.data_vars if v.startswith("B") and v[1].isalnum()
            ]
            ds = ds.drop_vars(all_b_bands)

            # --- STAGE 6: SENTINEL-1 (SAR) PROCESSING ---
            print("Step 6: Processing Sentinel-1 (Radar) Data...")
            fig = plot_acquisition_timelines_filtered(
                ds, ds.attrs["precip_end_date"], 12, False
            )
            save_plot_to_report(fig, report_sequence, stdout_buffer)

            # Integrity check for NaNs
            nan_mask_vv_before = ds.vv.isnull()
            nan_mask_vh_before = ds.vh.isnull()

            # Process S1 vars
            ds = clip_s1_data(ds, global_vv_max, global_vh_max)
            ds = apply_lee_to_ds(ds, bands=["vv", "vh"], win_size=7, cu=0.25)
            ds = normalize_s1_vars(ds, global_vv_max, global_vh_max)

            # Integrity check for NaNs
            nan_mask_vv_after = ds.vv.isnull()
            nan_mask_vh_after = ds.vh.isnull()
            assert (
                nan_mask_vv_before == nan_mask_vv_after
            ).all(), "NaN mismatch in VV!"
            assert (
                nan_mask_vh_before == nan_mask_vh_after
            ).all(), "NaN mismatch in VH!"

            print("✅ Sentinel-1 processed (Clipping, Lee-Filter, Normalization).")

            # --- STAGE 7: TEMPORAL ALIGNMENT (5-DAY INTERVALS) ---
            print("Step 7: Resampling to regular 5-day intervals...")
            ds, fig_analysis = align_all_to_5d(ds, "strict", False)
            save_plot_to_report(fig_analysis, report_sequence, stdout_buffer)
            print("✅ Temporal alignment finished.")

            # --- STAGE 8: ERA5 CLIMATE DATA MERGING ---
            print("Step 8: Merging and Standardizing ERA5 Climate Data...")
            # era5_path = f"../training_cubes_test/era5/{cube_id}_era5.zarr"
            era5_path = os.path.join(output_dir, "era5", f"{cube_id}_era5.zarr")
            if os.path.exists(era5_path):
                era5_ds = xr.open_zarr(era5_path, consolidated=True)

                # Make sure that they have same temporal resolution
                era5_ds = era5_ds.reindex(
                    {"time_sentinel_2_l2a": ds.time_sentinel_2_l2a},
                    method="nearest",
                    tolerance=pd.Timedelta(days=1),
                )

                # Normalize ERA5
                era5_ds = normalize_era5(era5_ds, global_era5_stats)

                # Delete encodings to avoid conflicts
                for v in era5_ds.data_vars:
                    era5_ds[v].encoding = {}

                era5_valid_chunks = {
                    dim: chunks
                    for dim, chunks in ds.chunks.items()
                    if dim in era5_ds.dims
                }
                era5_ds = era5_ds.chunk(era5_valid_chunks)

                # Merge
                ds = xr.merge([ds, era5_ds], compat="override")
                ds = ds.unify_chunks()
                print("✅ ERA5 merged and standardized using global stats.")
            else:
                print(f"⚠️ Warning: No ERA5 cache found for {cube_id}")

            # --- STAGE 9: FINAL STANDARDIZATION & CLIPPING ---
            print("Step 9: Global Standardization and Outlier Clipping (-5, 5)...")
            ds = normalize_dem(ds)
            for var in ds.data_vars:
                if var != "ESA_LC":  # Keep Land Cover labels as they are
                    ds[var] = ds[var].clip(-5, 5)

            # --- STAGE 10: EXPORT ---
            print("Step 10: Final Export Preparation")
            # Remove encodings
            for var in ds.data_vars:
                ds[var].encoding = {}

            # Define Chunks
            master_chunks = {"x": 250, "y": 250}
            if "time_sentinel_2_l2a" in ds.dims:
                master_chunks["time_sentinel_2_l2a"] = 1

            # Selective Chunking: Only use dimensions that variable has
            for var in ds.data_vars:
                # Create filter dict
                var_dims = ds[var].dims
                safe_chunks = {
                    dim: master_chunks[dim] for dim in var_dims if dim in master_chunks
                }

                # Apply filtered chunks
                ds[var] = ds[var].chunk(safe_chunks)

            # Final check and export
            ds = ds.unify_chunks()
            save_path = os.path.join(output_dir, f"{cube_id}.zarr")
            ds.to_zarr(save_path, mode="w", consolidated=True)
            print(f"\n✅ SUCCESS: Cube {cube_id} saved to {save_path}")

        except Exception as e:
            # Report the error in the HTML log
            print(f"\n{'!'*20} FATAL ERROR {'!'*20}")
            print(f"Cube ID: {cube_id}\nError: {str(e)}")
            # Ensure the error is also visible in the terminal
            sys.__stdout__.write(f"❌ Error in Cube {cube_id}: {str(e)}\n")

        finally:
            # Finalize Report
            report_sequence.append(("text", stdout_buffer.getvalue()))
            create_html_report(info_dir, cube_id, report_sequence)

            # Reset System State
            sys.stdout = sys.__stdout__
            stdout_buffer.close()

            # Explicit Memory Cleanup
            ds.close()
            del ds
            gc.collect()
            sys.__stdout__.write(f"Progress: {n+1}/{len(cubes)} cubes completed.\n")


def save_plot_to_report(fig, report_sequence, stdout_buffer):
    """
    Saves the current stdout buffer and the plot as Base64 to the report sequence.
    Clears the buffer afterwards to prepare for the next step.
    """
    # 1. Retrieve current text logs from buffer and add to sequence
    text_content = stdout_buffer.getvalue()
    if text_content.strip():
        report_sequence.append(("text", text_content))

    # 2. Convert plot to Base64 string
    tmp_buffer = io.BytesIO()
    fig.savefig(tmp_buffer, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)  # Free RAM immediately

    b64_img = base64.b64encode(tmp_buffer.getvalue()).decode()
    report_sequence.append(("plot_b64", b64_img))

    # 3. Clear buffer for the next processing section
    stdout_buffer.truncate(0)
    stdout_buffer.seek(0)

    return report_sequence


def save_map_to_report(m, info_dir, cube_id, report_sequence):
    """
    Saves a Folium map as HTML and adds an IFrame reference to the report.
    """
    if m is None:
        return report_sequence

    # 1. Define path for the map file (within plots subfolder)
    map_filename = f"map_{cube_id}.html"
    plot_dir = os.path.join(info_dir, "plots")

    os.makedirs(plot_dir, exist_ok=True)
    map_path = os.path.join(plot_dir, map_filename)

    # 2. Save Folium map as standalone HTML file
    m.save(map_path)

    # 3. Create IFrame code for the report to keep the map interactive
    iframe_html = f'<iframe src="plots/{map_filename}" width="100%" height="500px" style="border:none;"></iframe>'

    report_sequence.append(("html_raw", iframe_html))
    return report_sequence


def save_stats_to_json(stats, file_path):
    """
    Saves the statistics dictionary as a JSON file, converting NumPy types to floats.
    """

    def convert_types(obj):
        # Convert NumPy floats to Python floats for JSON compatibility
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        return obj

    clean_stats = convert_types(stats)

    # Write cleaned dictionary to JSON file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(clean_stats, f, indent=4)

    print(f"📂 Global stats saved to: {file_path}")


def load_global_stats(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def get_diverse_sample(data_pool, n_size, seed):
    """Zieht ein Sample, das versucht alle Klimaklassen abzudecken."""
    sample_mandatory = []
    current_pool = data_pool.copy()
    relevant_classes = [
        c for c in pool["climate_class"].unique() if c != "E" and c != "O"
    ]
    # Sicherstellen, dass jede Klasse einmal vorkommt (sofern n_size reicht)
    classes_in_pool = [
        c for c in relevant_classes if c in current_pool["climate_class"].unique()
    ]

    for c_class in classes_in_pool:
        if len(sample_mandatory) < n_size:
            match = current_pool[current_pool["climate_class"] == c_class].sample(
                n=1, random_state=seed
            )
            sample_mandatory.append(match)
            current_pool = current_pool.drop(match.index)

    # Rest auffüllen
    n_remaining = n_size - len(sample_mandatory)
    if n_remaining > 0:
        remainder = current_pool.sample(n=n_remaining, random_state=seed)
        sample_mandatory.append(remainder)

    return pd.concat(sample_mandatory)


def save_s1_stats(vv_max, vh_max, file_path):
    stats = {"global_vv_max": float(vv_max), "global_vh_max": float(vh_max)}
    with open(file_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"💾 Sentinel-1 Stats gespeichert in {file_path}")


def load_s1_stats(file_path):
    with open(file_path, "r") as f:
        stats = json.load(f)
    return stats["global_vv_max"], stats["global_vh_max"]


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Define data directories
    S3_BASE_URL = "https://s3.waw3-2.cloudferro.com/swift/v1/"
    ERA5_PATH = "/net/data/arceme/era5_land/"
    BUCKET_NAME = "ARCEME-DC-1"
    # OUTPUT_DIR = "../training_cubes"
    OUTPUT_DIR = "/scratch/sloeblein_new"
    pei_cube_name = "PEICube_era5land.zarr"
    t2_cube_name = "t2_ERA5land.zarr"
    tp_cube_name = "tp_ERA5land.zarr"

    # Setup S3 Filesystem
    import s3fs
    import fsspec

    fs = s3fs.S3FileSystem(
        anon=True, client_kwargs={"endpoint_url": "https://s3.waw3-2.cloudferro.com"}
    )

    # 2. LOAD & FILTER METADATA (CSV)
    print("Loading and filtering metadata...")
    df_data = pd.read_csv("data/train_test_split.csv", sep=",")
    pool = df_data.loc[df_data["year"] != 2015].copy()

    # 3. SAMPLING (Train/Test Split)
    # Using your diverse sampling logic to get the IDs
    test_df = get_diverse_sample(pool, n_size=6, seed=420)
    remaining_after_test = pool.drop(test_df.index)
    train_df = get_diverse_sample(remaining_after_test, n_size=22, seed=780)

    train_cubes_ids = train_df["DisNo."].to_list()

    # 4. REMOTE DATA FETCHING (S3 to xarray)
    print(f"Fetching {len(train_cubes_ids)} cubes from S3...")
    cubes = {}

    for cube_id in train_cubes_ids:

        try:
            url = f"{S3_BASE_URL}{BUCKET_NAME}/DC__{cube_id}.zarr"
            mapper = fsspec.get_mapper(url)
            ds = xr.open_zarr(mapper, consolidated=True)

            # Ensure correct chunking
            ds = ds.chunk(
                {
                    "x": 500,
                    "y": 500,
                    "time_sentinel_2_l2a": -1,
                    "time_sentinel_1_rtc": -1,
                }
            )

            # Metadata & DEM Pre-processing
            ds.attrs["cube_id"] = cube_id
            ds = add_event_metadata(ds, df_data, cube_id)

            # Collapse DEM time dimension
            ds["COP_DEM"] = ds.COP_DEM.mean(dim="time_cop_dem_glo_30_dged_cog")
            ds = ds.drop_dims("time_cop_dem_glo_30_dged_cog")

            cubes[cube_id] = ds
            print(f"Successfully connected to Cube: {cube_id}")

        except Exception as e:
            print(f"❌ Failed to load Cube {cube_id}: {e}")

    # 5. LOADINF ERA5 RAW DATA
    pei_cube = xr.open_zarr(os.path.join(ERA5_PATH, pei_cube_name), consolidated=False)
    t2_cube = xr.open_zarr(os.path.join(ERA5_PATH, t2_cube_name), consolidated=False)
    tp_cube = xr.open_zarr(os.path.join(ERA5_PATH, tp_cube_name), consolidated=False)
    era5_cubes = [pei_cube, t2_cube, tp_cube]

    # 6. RUN THE PIPELINE
    if cubes:
        try:
            run_processing_pipeline(
                cubes=cubes,
                era5_cubes=era5_cubes,  # Ensure this is loaded
                output_dir=OUTPUT_DIR,
                info_base="processing_info",
            )
            print("\n" + "=" * 60)
            print("🚀 PIPELINE EXECUTION FINISHED")
            print("=" * 60)
        except Exception as e:
            sys.stdout = sys.__stdout__
            print(f"Pipeline crashed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            sys.stdout = sys.__stdout__
    else:
        print("No cubes loaded. Pipeline skipped.")
