import pytorch_lightning as pl
from ConvLSTM import ConvLSTM
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class ConvLSTM_Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # 1. Dynamische Kanäle aus der Config ziehen
        # Vergangenheit (z.B. S1 + S2 + Wetter + Statics = 34)
        input_channels = cfg["model"]["input_channels"]

        # Zukunft (z.B. Wetter + Statics = 24)
        # Wir addieren +1, weil die Vorhersage (kNDVI) im Decoder dazugeklebt wird
        decoder_input_channels = cfg["model"]["future_channels"] + 1

        # Hier rufen wir dein ConvLSTM aus Schritt 2 auf
        # Wir ziehen alle Werte direkt aus der cfg-Datei
        self.model = ConvLSTM(
            input_dim=input_channels,  # 34
            decoder_input_dim=decoder_input_channels,
            output_dim=cfg["model"]["output_channels"],  # 1
            hidden_dims=cfg["model"]["hidden_channels"],  # z.B. [64, 64]
            num_layers=cfg["model"]["n_layers"],  # z.B. 2
            kernel_size=cfg["model"]["kernel"],  # 3
            dilation=cfg["model"]["dilation_rate"],  # 1
            baseline=cfg["model"]["baseline"],  # "last_frame"
        )

        # 3. Loss & Metriken
        # Wir nutzen MSE, aber 'none', damit wir die Maske (Vegetation) anwenden können
        self.criterion = nn.MSELoss(reduction="none")
        self.lr = cfg["training"]["start_learn_rate"]

    def forward(self, x_ctx, prediction_count, non_pred_feat):

        preds, pred_deltas, baselines = self.model(
            x_ctx, non_pred_feat=non_pred_feat, prediction_count=prediction_count
        )

        return preds, pred_deltas, baselines

    def training_step(self, batch, batch_idx):
        """
        Ein Batch kommt aus deinem Dataset.py und hat (B, T, C, H, W)
        """
        x_ctx, x_fut, y_true, mask, _ = batch
        # x_ctx: Context           (B, T_ctx, C_in, 256, 256)
        # x_fut: Climate in target (B, T_target, C_fut, 256, 256)
        # y_true: GT kNDVI         (B, T_target, 1 (kNDVI), 256, 256)
        # mask: Vegetation mask    (B, T_target, 1, 256, 256)

        # Forward pass
        y_pred, _, _ = self(x_ctx, prediction_count=y_true.size(1), non_pred_feat=x_fut)

        # Loss calculation
        loss = self.criterion(y_pred, y_true)  # (B, 5, 1, 256, 256)

        # Apply mask: Only mind vegetation pixels and filled pixels
        masked_loss = (loss * mask).sum() / (
            mask.sum() + 1e-8
        )  # but what if there are timesteps which are completely nan in GT

        self.log("train_loss", masked_loss, prog_bar=True, on_step=True, on_epoch=True)
        return masked_loss

    def validation_step(self, batch, batch_idx):
        x_ctx, x_fut, y_true, mask, _ = batch

        y_pred, _, _ = self(x_ctx, prediction_count=y_true.size(1), non_pred_feat=x_fut)

        loss = self.criterion(y_pred, y_true)
        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)

        self.log("val_loss", masked_loss, prog_bar=True, on_epoch=True)
        return masked_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Scheduler: Verringert die Lernrate, wenn der val_loss nicht mehr sinkt
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
