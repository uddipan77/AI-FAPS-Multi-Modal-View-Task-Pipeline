"""
Module for MultiTaskModule with classification fusion.

This module defines the MultiTaskModule class, which outputs:
  - Classification logits (for a 2-class problem) using fused image and force features.
  - Regression scalar (for DC resistance) using force-only features.

The classification head fuses features from 4 image views and force features via one of three methods:
  - "concat": Concatenates image features (from 4 images) and force features.
  - "bilinear": Applies a bilinear transformation to the concatenated features.
  - "attention": Uses a single-head attention mechanism on the fused features.

Example dimensions:
  - 4 images: each with shape [B, image_feature_dim]
  - Force: [B, force_feature_dim]
  - Classification fusion: [B, fused_dim]
  - Regression: [B, force_feature_dim]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning import LightningModule
from torchmetrics import Accuracy, MeanMetric, F1Score, R2Score, MaxMetric


class MultiTaskModule(LightningModule):
    """
    Multi-task Lightning module combining fused image+force features for classification and
    force-only features for regression.

    The classification head uses one of three fusion methods ("concat", "bilinear", or "attention")
    to combine the features. The regression head uses force-only features.
    """

    def __init__(
        self,
        modal_nets: dict,
        fusion_type: str = "attention",  # "concat", "bilinear", or "attention"
        num_classes: int = 2,
        optimizer_cfg: dict = None,
        scheduler_cfg: dict = None,
        alpha_classification: float = 1.0,
        alpha_regression: float = 1.0,
        # Feature dimensions
        image_feature_dim: int = 1000,   # per single image
        force_feature_dim: int = 64,     # from time-series net
        fusion_output_dim: int = 512,    # for bilinear/attention
        # Classification head MLP
        hidden_dim_class: int = 512,  # 256,
        dropout_class: float = 0.2,
        # Regression head MLP (force-only)
        hidden_dim_reg: int = 128,
        dropout_reg: float = 0.2,
        # Weighted metric scalars
        weight_f1: float = 0.5,
        weight_r2: float = 0.5,
        **kwargs
    ):
        """
        Initialize the MultiTaskModule.

        Args:
            modal_nets (dict): Dictionary of sub-networks for each modality.
            fusion_type (str, optional): Fusion type ("concat", "bilinear", or "attention"). Defaults to "attention".
            num_classes (int, optional): Number of classes for classification. Defaults to 2.
            optimizer_cfg (dict, optional): Optimizer configuration. Defaults to None.
            scheduler_cfg (dict, optional): Scheduler configuration. Defaults to None.
            alpha_classification (float, optional): Weight for classification loss. Defaults to 1.0.
            alpha_regression (float, optional): Weight for regression loss. Defaults to 1.0.
            image_feature_dim (int, optional): Feature dimension per image. Defaults to 1000.
            force_feature_dim (int, optional): Force feature dimension. Defaults to 64.
            fusion_output_dim (int, optional): Output dimension for fusion (for bilinear/attention). Defaults to 512.
            hidden_dim_class (int, optional): Hidden dimension for classification head MLP. Defaults to 512.
            dropout_class (float, optional): Dropout probability for classification head. Defaults to 0.2.
            hidden_dim_reg (int, optional): Hidden dimension for regression head (if used). Defaults to 128.
            dropout_reg (float, optional): Dropout probability for regression head. Defaults to 0.2.
            weight_f1 (float, optional): Weight factor for F1 metric. Defaults to 0.5.
            weight_r2 (float, optional): Weight factor for R2 metric. Defaults to 0.5.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Store sub-networks and fusion type.
        self.net = nn.ModuleDict({k: v for k, v in modal_nets.items()})
        self.fusion_type = fusion_type.lower()

        # Basic dimensions.
        self.image_feature_dim = image_feature_dim
        self.force_feature_dim = force_feature_dim

        # ----------- Classification Fusion -----------
        if self.fusion_type == "concat":
            # For 4 images, concatenate all image features along with force features.
            self.class_fused_dim = 4 * self.image_feature_dim + self.force_feature_dim

        elif self.fusion_type == "bilinear":
            self.class_fused_dim = fusion_output_dim
            self.class_fusion_layer = nn.Bilinear(
                4 * self.image_feature_dim,
                self.force_feature_dim,
                self.class_fused_dim
            )

        elif self.fusion_type == "attention":
            self.class_fused_dim = fusion_output_dim
            # Linear layers to compute Q (from image features) and K, V (from force features).
            self.Wq_class = nn.Linear(4 * self.image_feature_dim, self.class_fused_dim)
            self.Wk_class = nn.Linear(self.force_feature_dim, self.class_fused_dim)
            self.Wv_class = nn.Linear(self.force_feature_dim, self.class_fused_dim)

        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")

        # ----------- Classification MLP Head -----------
        self.classifier_head = nn.Sequential(
            nn.Linear(self.class_fused_dim, hidden_dim_class),
            nn.BatchNorm1d(hidden_dim_class),
            nn.ReLU(),
            nn.Dropout(dropout_class),
            nn.Linear(hidden_dim_class, num_classes)
        )

        # ----------- Regression is Force-Only -----------
        # Regression head uses only force features.
        self.regressor_head = nn.Linear(self.force_feature_dim, 1)
        # Alternative regression head (uncomment if needed):
        # self.regressor_head = nn.Sequential(
        #     nn.Linear(self.force_feature_dim, hidden_dim_reg),
        #     nn.BatchNorm1d(hidden_dim_reg),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_reg),
        #     nn.Linear(hidden_dim_reg, 1)
        # )

        # Normalization layers.
        self.image_norm = nn.LayerNorm(4 * self.image_feature_dim)
        self.force_norm = nn.LayerNorm(self.force_feature_dim)

        # Loss functions.
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        # Classification metrics.
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        task_type = "binary" if num_classes == 2 else "multiclass"
        self.train_f1 = F1Score(task=task_type, num_classes=num_classes)
        self.val_f1 = F1Score(task=task_type, num_classes=num_classes)
        self.test_f1 = F1Score(task=task_type, num_classes=num_classes)

        # Regression metrics.
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

        # Track best metrics.
        self.val_f1_best = MaxMetric()
        self.val_r2_best = MaxMetric()

        # Mean metrics for logging losses.
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Weighted multi-task loss.
        self.alpha_class = alpha_classification
        self.alpha_reg = alpha_regression

        # Weighted metric (F1 + R2).
        self.weight_f1 = weight_f1
        self.weight_r2 = weight_r2

    # ---------------------------
    # Fusion for Classification
    # ---------------------------
    def forward_class_fusion(self, img_feats, force_feats):
        """
        Fuse image and force features for classification.

        Args:
            img_feats (Tensor): Image features tensor with shape [B, 4 * image_feature_dim].
            force_feats (Tensor): Force features tensor with shape [B, force_feature_dim].

        Returns:
            Tensor: Fused feature tensor for classification.
        """
        if self.fusion_type == "concat":
            fused = torch.cat([img_feats, force_feats], dim=-1)

        elif self.fusion_type == "bilinear":
            fused = self.class_fusion_layer(img_feats, force_feats)

        elif self.fusion_type == "attention":
            Q = self.Wq_class(img_feats)
            K = self.Wk_class(force_feats)
            V = self.Wv_class(force_feats)
            d_k = float(self.class_fused_dim)
            attn_scores = (Q * K).sum(dim=-1, keepdim=True) / math.sqrt(d_k)
            alpha = torch.sigmoid(attn_scores)  # shape: [B, 1]
            fused = alpha * V + (1.0 - alpha) * Q

        return fused

    def forward(self, batch_dict):
        """
        Forward pass through the network.

        Args:
            batch_dict (dict): Dictionary with the following keys:
                "images": List of 4 tensors, each of shape [B, 3, H, W].
                "forces": Tensor of shape [B, 5, seq_len] containing force time series.
                "class_label": Tensor of shape [B] with classification labels.
                "reg_label": Tensor of shape [B] with regression labels.

        Returns:
            tuple: A tuple (logits, reg_out) where:
                logits (Tensor): Classification outputs with shape [B, num_classes].
                reg_out (Tensor): Regression outputs with shape [B, 1].
        """
        img_feats = None
        force_feats = None

        # 1) Gather features from sub-networks.
        for modality_name, net_module in self.net.items():
            if modality_name == "images":
                views = batch_dict["images"]  # list of 4 images, each of shape [B, 3, H, W]
                view_feats = []
                for img_tensor in views:
                    feat = net_module(img_tensor)  # => [B, image_feature_dim]
                    view_feats.append(feat)
                # Concatenate features from 4 images to get shape [B, 4 * image_feature_dim].
                img_feats = torch.cat(view_feats, dim=-1)
                img_feats = self.image_norm(img_feats)
            elif modality_name == "forces":
                force_feats = net_module(batch_dict["forces"])  # => [B, force_feature_dim]
                force_feats = self.force_norm(force_feats)

        # 2) Classification branch: Fuse features and pass through MLP.
        fused_class = self.forward_class_fusion(img_feats, force_feats)
        logits = self.classifier_head(fused_class)

        # 3) Regression branch: Force-only MLP.
        reg_out = self.regressor_head(force_feats)

        return logits, reg_out

    # ------------------------------
    # Training Step
    # ------------------------------
    def training_step(self, batch, batch_idx):
        """
        Execute a training step.

        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Combined loss for the training step.
        """
        logits, reg_out = self(batch)

        class_loss = self.ce_loss(logits, batch["class_label"])
        mse = self.mse_loss(reg_out, batch["reg_label"].unsqueeze(-1))

        preds = torch.argmax(logits, dim=-1)
        self.train_acc(preds, batch["class_label"])
        self.train_f1(preds, batch["class_label"])
        self.train_r2.update(reg_out.view(-1), batch["reg_label"].view(-1))

        loss = self.alpha_class * class_loss + self.alpha_reg * mse
        self.train_loss.update(loss)
        self.log("train/loss", self.train_loss, on_epoch=True, prog_bar=True)
        self.log("train/class_loss", class_loss, on_epoch=True)
        self.log("train/mse_loss", mse, on_epoch=True)
        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        """
        Called at the end of a training epoch. Logs and resets training metrics.
        """
        r2_val = self.train_r2.compute()
        self.log("train/r2", r2_val, on_epoch=True)
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_r2.reset()
        self.train_loss.reset()

    # ------------------------------
    # Validation Step
    # ------------------------------
    def validation_step(self, batch, batch_idx):
        """
        Execute a validation step.

        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Combined loss for the validation step.
        """
        logits, reg_out = self(batch)
        class_loss = self.ce_loss(logits, batch["class_label"])
        mse = self.mse_loss(reg_out, batch["reg_label"].unsqueeze(-1))

        preds = torch.argmax(logits, dim=-1)
        self.val_acc(preds, batch["class_label"])
        self.val_f1(preds, batch["class_label"])
        self.val_r2.update(reg_out.view(-1), batch["reg_label"].view(-1))

        loss = self.alpha_class * class_loss + self.alpha_reg * mse
        self.val_loss.update(loss)
        self.log("val/loss", self.val_loss, on_epoch=True, prog_bar=True)
        self.log("val/class_loss", class_loss, on_epoch=True)
        self.log("val/mse_loss", mse, on_epoch=True)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """
        Called at the end of a validation epoch. Logs best metrics, weighted metric,
        and resets validation metrics.
        """
        f1_score = self.val_f1.compute()
        self.val_f1_best.update(f1_score)
        self.log("val/f1_best", self.val_f1_best.compute(), prog_bar=True)

        r2_val = self.val_r2.compute()
        self.val_r2_best.update(r2_val)
        self.log("val/r2_best", self.val_r2_best.compute(), prog_bar=True)

        weighted_metric = self.weight_f1 * f1_score + self.weight_r2 * r2_val
        self.log("val/weighted_f1_r2", weighted_metric, on_epoch=True, prog_bar=True)

        val_loss_epoch = self.val_loss.compute()
        self.log("val/loss_epoch", val_loss_epoch, prog_bar=True)

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_r2.reset()
        self.val_loss.reset()

    # ------------------------------
    # Test Step
    # ------------------------------
    def test_step(self, batch, batch_idx):
        """
        Execute a test step.

        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Combined loss for the test step.
        """
        logits, reg_out = self(batch)
        class_loss = self.ce_loss(logits, batch["class_label"])
        mse = self.mse_loss(reg_out, batch["reg_label"].unsqueeze(-1))

        preds = torch.argmax(logits, dim=-1)
        self.test_acc(preds, batch["class_label"])
        self.test_f1(preds, batch["class_label"])
        self.test_r2.update(reg_out.view(-1), batch["reg_label"].view(-1))

        loss = self.alpha_class * class_loss + self.alpha_reg * mse
        self.test_loss.update(loss)
        self.log("test/loss", self.test_loss, on_epoch=True, prog_bar=True)
        self.log("test/class_loss", class_loss, on_epoch=True)
        self.log("test/mse_loss", mse, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        """
        Called at the end of a test epoch. Logs test metrics and resets test metrics.
        """
        self.log("test/acc", self.test_acc.compute(), prog_bar=True)
        self.log("test/f1", self.test_f1.compute(), prog_bar=True)

        r2_val = self.test_r2.compute()
        self.log("test/r2", r2_val, prog_bar=True)

        self.test_acc.reset()
        self.test_f1.reset()
        self.test_r2.reset()
        self.test_loss.reset()

    # ------------------------------
    # Optimizer Configuration
    # ------------------------------
    def configure_optimizers(self):
        """
        Configure and return the optimizer and learning rate scheduler.

        Returns:
            dict: Dictionary containing the optimizer and, if specified, the scheduler configuration.
        """
        opt_cfg = self.hparams.optimizer_cfg
        sch_cfg = self.hparams.scheduler_cfg
        optimizer = opt_cfg(params=self.parameters())

        if sch_cfg is not None:
            scheduler = sch_cfg(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_epoch",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
