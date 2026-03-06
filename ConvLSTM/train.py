import yaml
import torch
import os
import random
import numpy as np
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from ConvLSTM_model import ConvLSTM_Model
from dataset import ARCEME_Dataset, get_llto_splits, get_val_tiles_auto
from utils import print_channel_info

# Set seed (only for reproducibility between patch approaches - can be delted later)
L.seed_everything(42, workers=True)
torch.set_float32_matmul_precision("medium")

# --- Load Config ---
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# --- Constants & Paths ---
PROCESSED_DIR = "/scratch/sloeblein/postprocessed"
CSV_PATH = "../final_processing_pipeline/data/train_test_split.csv"
K_FOLDS = 4  # for CV (test for 4 and 5)

# --- Dynamic Channel Calculation ---
v_cfg = cfg["data"]["variables"]
num_s2 = len(v_cfg["s2"])
num_s1 = len(v_cfg["s1"])
num_era5 = len(v_cfg["era5"])
num_masks = 2  # mask_s1 and mask_s2
num_lc = 12  # ESA Landcover One-Hot encoded
num_stat = len(v_cfg["static"])

# Input channels (Context): All features combined
# S2 + S1 + ERA5 + Masks + LC_OneHot + Statics
total_in_channels = num_s2 + num_s1 + num_era5 + num_masks + num_lc + num_stat

# Future channels (Decoder): Features known in the future
# ERA5 + LC_OneHot + Statics
total_fut_channels = num_era5  # + num_lc + num_stat

print("--- Channel Configuration ---")
print(f"Input Channels (Context): {total_in_channels}")
print(f"Future Channels (Known):  {total_fut_channels}")
print("-----------------------------")


# Get CV Splits
all_folds = get_llto_splits(PROCESSED_DIR, CSV_PATH, k=K_FOLDS)

# --- Training Loop ---
for fold_idx, (train_files, val_files) in enumerate(all_folds):
    print("\n" + "=" * 50)
    print(f"🚀 STARTING FOLD {fold_idx} (LLTO-CV)")
    print("=" * 50)

    # Get list of patches for validation cubes
    val_tiles = get_val_tiles_auto(val_files, patch_size=cfg["data"]["patch_size"])

    # Create Datasets
    train_ds = ARCEME_Dataset(
        train_files,
        context_length=cfg["data"]["context_length"],
        target_length=cfg["data"]["target_length"],
        patch_size=cfg["data"]["patch_size"],
        train=True,
        s2_vars=v_cfg["s2"],
        s1_vars=v_cfg["s1"],
        era5_vars=v_cfg["era5"],
        static_vars=v_cfg["static"],
    )
    val_ds = ARCEME_Dataset(
        val_files,
        context_length=cfg["data"]["context_length"],
        target_length=cfg["data"]["target_length"],
        patch_size=cfg["data"]["patch_size"],
        train=False,
        s2_vars=v_cfg["s2"],
        s1_vars=v_cfg["s1"],
        era5_vars=v_cfg["era5"],
        static_vars=v_cfg["static"],
        fixed_tiles=val_tiles,
    )

    ######################################################  Kann raus #######################################################################################
    # 2. Worker-Handling für echte Reproduzierbarkeit
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)
    ##########################################################################################################################################################

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
    )  # seed worker und so kann gelöscht werden
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=4
    )

    # Model Initialization
    # Pass calculated channels into the model config
    cfg["model"]["input_channels"] = total_in_channels
    cfg["model"]["future_channels"] = total_fut_channels

    model = ConvLSTM_Model(cfg)

    # Logger & Callbacks (Saves best model of a fold)
    logger = TensorBoardLogger("tb_logs", name=f"fold_{fold_idx}")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="best-model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    # Trainer
    trainer = Trainer(
        check_val_every_n_epoch=1,
        max_epochs=cfg["training"]["max_epochs"],
        accelerator=cfg["training"]["accelerator"],
        devices=cfg["training"]["devices"],
        precision=cfg["training"]["precision"],
        logger=logger,
        callbacks=[checkpoint_callback, early_stop],
    )

    # Overview
    print_channel_info(v_cfg["s2"], v_cfg["s1"], v_cfg["era5"], v_cfg["static"])

    # Start training
    print(f"Starting training for Fold {fold_idx}...")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    trainer.fit(model, train_loader, val_loader)
