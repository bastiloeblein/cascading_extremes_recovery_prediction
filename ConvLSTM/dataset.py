import os
import torch
import pandas as pd
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold


class ARCEME_Dataset(Dataset):
    def __init__(
        self,
        cube_paths,
        context_length,
        target_length,
        patch_size,
        train,
        fixed_tiles=None,
    ):
        """
        Args:
            cube_paths: List of paths to the .zarr files.
            context_length: Number of timesteps for input (X).
            target_length: Number of timesteps for prediction (Y).
            patch_size: Size of the spatial crop
            train: If True, uses random cropping. If False, uses center crop or full.
            fixed_tiles: Containing the list with offsets for tiling validation strategy
        """
        self.cube_paths = cube_paths
        self.context_len = context_length
        self.target_len = target_length
        self.patch_size = patch_size
        self.train = train
        self.fixed_tiles = fixed_tiles

        # Definition of the channels (Order matters for the model!)
        self.s2_vars = ["kNDVI", "CIRE", "IRECI", "NDWI", "NDMI", "NIRv"]  #  UPDATE!!
        self.s1_vars = ["vv", "vh"]
        self.era5_vars = [
            "pei_30_mean",
            "pei_90_mean",
            "pei_180_mean",
            "pei_360_mean",  # UPDATE!
            "t2m_mean",
            "t2mmax_mean",
            "t2mmin_mean",
            "tp_dailymax_mean",
            "tp_dailymean_mean",
            "tp_rollingmax_mean",
        ]
        self.static_vars = ["ESA_LC", "COP_DEM", "is_veg"]

    def __len__(self):
        if self.fixed_tiles is not None:
            return len(self.fixed_tiles)

        # So the model does not see so few patches per epoch
        if len(self.cube_paths) < 50:
            return len(self.cube_paths) * 10

        return len(self.cube_paths)

    def __getitem__(self, idx):

        # 1. Patching strategy
        if not self.train and self.fixed_tiles is not None:
            # VALIDATION STRATEGY: Use the calculated points
            tile_info = self.fixed_tiles[idx]
            path = tile_info["path"]
            top = tile_info["top"]
            left = tile_info["left"]
        else:
            # TRAINING STRATEGY: Random Patching
            # Use modulo to prevent IndexErrors if you fake __len__
            real_idx = idx % len(self.cube_paths)
            path = self.cube_paths[real_idx]

            try:
                ds_temp = xr.open_zarr(path, consolidated=True)
            except Exception:
                ds_temp = xr.open_zarr(path, consolidated=False)

            h, w = ds_temp.x.size, ds_temp.y.size
            top = np.random.randint(0, h - self.patch_size)
            left = np.random.randint(0, w - self.patch_size)
            ds_temp.close()

        # 2. Now load the actual patch
        try:
            ds = xr.open_zarr(path, consolidated=True)
        except Exception:
            ds = xr.open_zarr(path, consolidated=False)

        # Select patch
        ds = ds.isel(
            x=slice(top, top + self.patch_size), y=slice(left, left + self.patch_size)
        )

        # Check if ds is expected size
        if ds.x.size < self.patch_size or ds.y.size < self.patch_size:
            raise ValueError(
                f"Patch slicing failed at {path}. Received: {ds.x.size}x{ds.y.size}"
            )

        # 3. Time Window slicing based on metadata
        cutoff_date = pd.to_datetime(ds.attrs["precip_end_date"])
        ds_ctx = ds.sel(time_sentinel_2_l2a=slice(None, cutoff_date)).tail(
            time_sentinel_2_l2a=self.context_len
        )
        ds_target = ds.sel(
            time_sentinel_2_l2a=slice(cutoff_date + pd.Timedelta(days=1), None)
        ).head(time_sentinel_2_l2a=self.target_len)

        # Asserts for Time Length
        assert (
            len(ds_ctx.time_sentinel_2_l2a) == self.context_len
        ), f"Context time mismatch: expected {self.context_len}, got {len(ds_ctx.time_sentinel_2_l2a)}"
        assert (
            len(ds_target.time_sentinel_2_l2a) == self.target_len
        ), f"Target time mismatch: expected {self.target_len}, got {len(ds_target.time_sentinel_2_l2a)}"

        # 4. Fill NaNs with 0.0
        ds_ctx = ds_ctx.fillna(0.0)
        ds_target = ds_target.fillna(0.0)

        # 5. Input Construction (Context Window)
        # Convert xarray to numpy, then to torch tensor (float)
        x_s2 = torch.from_numpy(
            ds_ctx[self.s2_vars].to_array().values
        ).float()  # (C_s2, T, H, W)
        x_s1 = torch.from_numpy(
            ds_ctx[self.s1_vars].to_array().values
        ).float()  # (C_s1, T, H, W)
        x_era5_1d = torch.from_numpy(
            ds_ctx[self.era5_vars].to_array().values
        ).float()  # (C_era5, T, H, W) # doenst have H and W yet

        # ERA5 Broadcasting: (C, T) -> (C, T, H, W)
        x_era5 = broadcast_era5(x_era5_1d, self.patch_size, self.patch_size)

        # 6. Masking & Vegetation Logic (Mirroring EarthNet example)
        # Get vegetation mask
        is_veg = torch.from_numpy(ds_ctx["is_veg"].values).float()  # (T, H, W)

        # Mask S2 and S1 by vegetation
        m_s2 = (torch.from_numpy(ds_ctx["mask_s2"].values).float() * is_veg).unsqueeze(
            0
        )  # (1, T, H, W)
        m_s1 = (torch.from_numpy(ds_ctx["mask_s1"].values).float() * is_veg).unsqueeze(
            0
        )  # (1, T, H, W)

        # 7. ESA LC One-Hot-Encoding
        ## 1. Extract ESA_LC and convert to torch
        lc = torch.from_numpy(ds_ctx["ESA_LC"].values).long()  # (T, H, W)
        lc_onehot = encode_landcover(lc)

        # 6. Static features
        x_stat_raw = torch.from_numpy(
            ds_ctx[self.static_vars].to_array().values
        ).float()  # (C_stat, T, H, W)
        x_static = torch.cat(
            [lc_onehot, x_stat_raw.permute(1, 0, 2, 3)], dim=1
        ).permute(
            1, 0, 2, 3
        )  # (13, T, H, W)

        # 7. Final Input Tensor: (T, C, H, W)
        x_context = torch.cat(
            [x_s2, x_s1, x_era5, m_s2, m_s1, x_static], dim=0
        ).permute(
            1, 0, 2, 3
        )  # (T_ctx, C_all, H, W)

        # Validation checks
        # Check if Time dimension is consistent across all inputs
        time_dims = [
            x_s2.shape[1],
            x_s1.shape[1],
            x_era5.shape[1],
            m_s2.shape[1],
            m_s1.shape[1],
            x_static.shape[1],
        ]
        assert len(set(time_dims)) == 1, f"Time dimension mismatch: {time_dims}"

        # Check if Channel count is correct
        expected_channels = (
            x_s2.shape[0]
            + x_s1.shape[0]
            + x_era5.shape[0]
            + m_s2.shape[0]
            + m_s1.shape[0]
            + x_static.shape[0]
        )
        assert (
            x_context.shape[1] == expected_channels
        ), f"Channel mismatch! Expected {expected_channels}, got {x_context.shape[1]}"

        # 8. Future Climate (used to guide the model prediction)
        x_fut_era5_1d = torch.from_numpy(
            ds_target[self.era5_vars].to_array().values
        ).float()
        x_future_era5 = broadcast_era5(
            x_fut_era5_1d, self.patch_size, self.patch_size
        )  # (T_target, C_era5, H, W)

        # 9. Target & Loss Mask
        y_target = (
            torch.from_numpy(ds_target["kNDVI"].values).unsqueeze(1).float()
        )  # (T_target, H, W)

        # Loss Mask: Only calculate loss where kNDVI is real AND it is vegetation
        is_veg_target = torch.from_numpy(ds_target["is_veg"].values).float()
        target_mask = (
            torch.from_numpy(ds_target["target_mask"].values).float() * is_veg_target
        ).unsqueeze(
            1
        )  # (T_target, 1, H, W)

        # Final Tensor Shape Checks
        assert (
            x_context.ndim == 4
        ), f"Context must be (T, C, H, W), got {x_context.shape}"
        assert (
            x_future_era5.ndim == 4
        ), f"Future ERA5 must be (T, C, H, W), got {x_future_era5.shape}"
        assert y_target.shape[0] == self.target_len, "Target time dimension mismatch"

        # Range Checks
        assert not torch.isnan(x_context).any(), "NaN values detected in x_context"
        assert not torch.isnan(
            x_future_era5
        ).any(), "NaN values detected in x_future_era5"
        assert not torch.isnan(y_target).any(), "NaN values detected in y_target"
        assert (
            x_context.size(2) == self.patch_size
            and x_context.size(3) == self.patch_size
        ), "Spatial dimensions mismatch"

        meta = {"top": top, "left": left, "path": path}

        ds.close()

        return x_context, x_future_era5, y_target, target_mask, meta


# --- Implementation of Leave-Time-and-Region-Out CV ---


def get_llto_splits(root_dir, csv_path="train_test_split.csv", k=5, show=False):
    """
    Implements k-fold Leave-Location-and-Time-Out (LLTO) splitting according to
    https://doi.org/10.1016/j.envsoft.2017.12.001.
    Groups data by Koppen-Geiger region (Location) and Phenological Season (Time)
    and ensures that Region+Season group are never split between train and val.

    Args:
        root_dir: Directory containing the .zarr cubes.
        csv_path: Path to the metadata CSV.
        k: Number of folds (e.g., 3, 4, or 5).

    Returns:
        List of tuples: [(train_paths, val_paths), ...] for k folds.
    """
    # 1. Load and filter only the training split
    df = pd.read_csv(csv_path)
    df = df[df["split"] == "train"].copy()

    # 2. Map file paths
    processed_files = {
        f.replace("_postprocessed.zarr", ""): os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.endswith(".zarr")
    }
    df["full_path"] = df["DisNo."].map(processed_files)

    # Drop entries not found on disk
    initial_count = len(df)
    df = df.dropna(subset=["full_path"])
    print(f"Matched {len(df)}/{initial_count} cubes from CSV to disk.")

    # 3. Create the unique 'Location-Time' Group ID
    df["llto_group"] = (
        df["koppen_geiger"].astype(str) + "_" + df["pheno_season_name"].astype(str)
    )

    # 4. Initialize GroupKFold
    gkf = GroupKFold(n_splits=k)

    cv_splits = []

    # GroupKFold requires X, y, and groups.
    indices = np.arange(len(df))
    groups = df["llto_group"].values

    for train_idx, val_idx in gkf.split(indices, groups=groups):
        train_paths = df.iloc[train_idx]["full_path"].tolist()
        val_paths = df.iloc[val_idx]["full_path"].tolist()

        cv_splits.append((train_paths, val_paths))

    print(f"Successfully created {k} folds using LLTO GroupKFold.")

    # Print fold statistics to verify independence
    if show:
        for i, (t, v) in enumerate(cv_splits):
            print(f"Fold {i}: Train={len(t)} cubes, Val={len(v)} cubes")

    return cv_splits


def broadcast_era5(era5_tensor, target_h, target_w):
    """
    Broadcasts a (C, T) era5 tensor to (C, T, H, W).
    """
    # era5_tensor: (Channels, Time)
    # Result: (Channels, Time, H, W)
    return era5_tensor.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, target_h, target_w)


def get_val_tiles_auto(cube_paths, patch_size=256, dim_max=1000):
    """
    Automatically creates a grid of tiles to perfectly cover the cube.
    Calculates the required number of tiles and optimal stride to
    ensure no gaps and uniform minimal overlap.
    """
    tiled_list = []

    # 1. Calculate how many patches we need to cover the dimension
    num_tiles = int(np.ceil(dim_max / patch_size))

    # 2. Calculate the optimal stride
    if num_tiles > 1:
        stride = (dim_max - patch_size) // (num_tiles - 1)
    else:
        stride = 0

    # 3. Generate starting coordinates (offsets)
    offsets = [i * stride for i in range(num_tiles)]

    # 4. Final safety check: ensure the last patch ends exactly at dim_max
    if offsets[-1] + patch_size != dim_max:
        offsets[-1] = dim_max - patch_size

    for path in cube_paths:
        for top in offsets:
            for left in offsets:
                tiled_list.append({"path": path, "top": top, "left": left})

    print(f"Grid Strategy: {num_tiles}x{num_tiles} tiles ({len(offsets)**2} per cube).")
    print(f"Patch Size: {patch_size}, Resulting Stride: {stride}")

    return tiled_list


def encode_landcover(lc_tensor):
    """
    Transforms ESA Landcover into One-Hot Encoding.
    Input: (T, H, W) or (H, W)
    Output: (T, 12, H, W)
    """
    labels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    mapping = torch.zeros(101, dtype=torch.long)
    for i, val in enumerate(labels):
        mapping[val] = i

    lc_mapped = mapping[lc_tensor]
    # One-Hot and permute to (T, C, H, W)
    lc_onehot = F.one_hot(lc_mapped, num_classes=len(labels))

    if lc_onehot.ndim == 4:  # (T, H, W, C)
        return lc_onehot.permute(0, 3, 1, 2).float()
    else:  # (H, W, C) -> (C, H, W)
        return lc_onehot.permute(2, 0, 1).float()
