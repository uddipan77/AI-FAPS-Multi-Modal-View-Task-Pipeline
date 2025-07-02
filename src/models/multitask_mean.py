"""
Module for MultiTaskModule with advanced fusion options.

This module defines the MultiTaskModule class, which is a multi-task Lightning
module that outputs classification logits (for a 2-class problem) and a regression
scalar (for DC resistance). The classification branch uses fusion of image and force
features via one of three methods:
  - "concat": Concatenate image and force features.
  - "bilinear": Apply a bilinear transformation to fuse image and force features.
  - "attention": Use a single-head gating-based attention mechanism.
By default, the regression branch is force-only.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning import LightningModule
from torchmetrics import Accuracy, MeanMetric, F1Score, R2Score, MaxMetric


class MultiTaskModule(LightningModule):
    """
    Multi-task Lightning module with classification and regression outputs.

    The module outputs:
      - Classification logits (for a 2-class problem) by fusing image and force features.
      - Regression scalar (for DC resistance) using only force features.

    Fusion methods available for classification:
      - "concat": Concatenates image features and force features.
      - "bilinear": Uses a bilinear transform between image features and force features.
      - "attention": Uses a single-head gating-based attention mechanism.
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
        image_feature_dim: int = 1000,  # e.g. DenseNet output dim
        force_feature_dim: int = 64,    # e.g. InceptionTime output dim
        fusion_output_dim: int = 512,     # dimension of fused features (for bilinear/attention)
        hidden_dim_class: int = 256,
        dropout_class: float = 0.2,
        hidden_dim_reg: int = 128,
        dropout_reg: float = 0.2,
        weight_f1: float = 0.5,
        weight_r2: float = 0.5,
        **kwargs
    ):
        """
        Initialize the MultiTaskModule.

        Args:
            modal_nets (dict): Dictionary of sub-networks for each modality.
            fusion_type (str, optional): Fusion method ("concat", "bilinear", or 
                "attention"). Defaults to "attention".
            num_classes (int, optional): Number of classes for classification.
                Defaults to 2.
            optimizer_cfg (dict, optional): Optimizer configuration.
            scheduler_cfg (dict, optional): Scheduler configuration.
            alpha_classification (float, optional): Weight for classification loss.
                Defaults to 1.0.
            alpha_regression (float, optional): Weight for regression loss.
                Defaults to 1.0.
            image_feature_dim (int, optional): Dimensionality of image features.
                Defaults to 1000.
            force_feature_dim (int, optional): Dimensionality of force features.
                Defaults to 64.
            fusion_output_dim (int, optional): Dimensionality of fused features (for
                bilinear or attention). Defaults to 512.
            hidden_dim_class (int, optional): Hidden dimension for the classification
                MLP head. Defaults to 256.
            dropout_class (float, optional): Dropout probability for the classification
                head. Defaults to 0.2.
            hidden_dim_reg (int, optional): Hidden dimension for the regression MLP head.
                Defaults to 128.
            dropout_reg (float, optional): Dropout probability for the regression head.
                Defaults to 0.2.
            weight_f1 (float, optional): Weight factor for the F1 metric.
                Defaults to 0.5.
            weight_r2 (float, optional): Weight factor for the R2 metric.
                Defaults to 0.5.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Sub-networks.
        self.net = nn.ModuleDict({k: v for k, v in modal_nets.items()})
        self.fusion_type = fusion_type.lower()

        # Store feature dimensions.
        self.force_feature_dim = force_feature_dim
        self.image_feature_dim = image_feature_dim

        # Determine classification fused dimension.
        if self.fusion_type == "concat":
            # Fused feature will be the concatenation of image and force features.
            self.class_fused_dim = self.image_feature_dim + self.force_feature_dim
        elif self.fusion_type == "bilinear":
            # Fused feature dimension is provided by fusion_output_dim.
            self.class_fused_dim = fusion_output_dim
            self.fusion_layer = nn.Bilinear(
                self.image_feature_dim,
                self.force_feature_dim,
                self.class_fused_dim
            )
        elif self.fusion_type == "attention":
            self.class_fused_dim = fusion_output_dim
            # For attention fusion, define linear layers to compute Q, K, and V.
            self.Wq = nn.Linear(self.image_feature_dim, self.class_fused_dim)
            self.Wk = nn.Linear(self.force_feature_dim, self.class_fused_dim)
            self.Wv = nn.Linear(self.force_feature_dim, self.class_fused_dim)
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")

        # Classification Head (MLP).
        self.classifier_head = nn.Sequential(
            nn.Linear(self.class_fused_dim, hidden_dim_class),
            nn.BatchNorm1d(hidden_dim_class),
            nn.ReLU(),
            nn.Dropout(dropout_class),
            nn.Linear(hidden_dim_class, num_classes)
        )

        # Regression Head (Force-only MLP).
        self.regressor_head = nn.Linear(self.force_feature_dim, 1)
        # Alternative regression head (uncomment below if needed).
        # self.regressor_head = nn.Sequential(
        #     nn.Linear(self.force_feature_dim, hidden_dim_reg),
        #     nn.BatchNorm1d(hidden_dim_reg),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_reg),
        #     nn.Linear(hidden_dim_reg, 1)
        # )

        # Normalization layers.
        self.image_norm = nn.LayerNorm(self.image_feature_dim)
        self.force_norm = nn.LayerNorm(self.force_feature_dim)

        # Loss functions.
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        # Classification Metrics.
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        task_type = "binary" if num_classes == 2 else "multiclass"
        self.train_f1 = F1Score(task=task_type, num_classes=num_classes)
        self.val_f1 = F1Score(task=task_type, num_classes=num_classes)
        self.test_f1 = F1Score(task=task_type, num_classes=num_classes)

        # Regression Metrics.
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

        # Best metric trackers.
        self.val_f1_best = MaxMetric()
        self.val_r2_best = MaxMetric()

        # Mean metrics for logging losses.
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Loss weighting for multi-task optimization.
        self.alpha_class = alpha_classification
        self.alpha_reg = alpha_regression

        # Weights for combining metrics.
        self.weight_f1 = weight_f1
        self.weight_r2 = weight_r2

    def fuse_for_classification(self, img_feats, force_feats):
        """
        Fuse image and force features for classification.

        Args:
            img_feats (Tensor): Image features of shape [B, image_feature_dim].
            force_feats (Tensor): Force features of shape [B, force_feature_dim].

        Returns:
            Tensor: Fused feature tensor.
        """
        if self.fusion_type == "concat":
            if img_feats is None:
                return force_feats
            if force_feats is None:
                return img_feats
            fused_feat = torch.cat([img_feats, force_feats], dim=-1)

        elif self.fusion_type == "bilinear":
            if (img_feats is None) or (force_feats is None):
                raise ValueError("Bilinear fusion requires both image and force features.")
            fused_feat = self.fusion_layer(img_feats, force_feats)

        elif self.fusion_type == "attention":
            if (img_feats is None) or (force_feats is None):
                raise ValueError("Attention fusion requires both image and force features.")
            Q = self.Wq(img_feats)
            K = self.Wk(force_feats)
            V = self.Wv(force_feats)

            d_k = float(self.class_fused_dim)
            attn_scores = (Q * K).sum(dim=-1, keepdim=True) / math.sqrt(d_k)
            alpha = torch.sigmoid(attn_scores)
            fused_feat = alpha * V + (1.0 - alpha) * Q

        return fused_feat

    def forward(self, batch_dict):
        """
        Forward pass through the network.

        Args:
            batch_dict (dict): Dictionary containing:
                "images": [4 x Tensor(B, 3, H, W)] - list of image tensors,
                "forces": Tensor(B, 5, seq_len) - force time series,
                "class_label": Tensor(B) - classification labels,
                "reg_label": Tensor(B) - regression labels.

        Returns:
            tuple: A tuple (logits, reg_out) where:
                - logits (Tensor): Classification output with shape [B, num_classes].
                - reg_out (Tensor): Regression output with shape [B, 1].
        """
        img_feats = None
        force_feats = None

        # 1) Gather features from sub-networks.
        for modality_name, net_module in self.net.items():
            if modality_name == "images":
                # Process image inputs: 4 images per sample.
                # Compute embeddings for each view.
                views = batch_dict["images"]
                view_feats = []
                for img_tensor in views:
                    feat = net_module(img_tensor)
                    view_feats.append(feat)
                # Average the image embeddings.
                img_feats_stacked = torch.stack(view_feats, dim=1)  # [B, 4, image_feature_dim]
                img_feats = img_feats_stacked.mean(dim=1)
                img_feats = self.image_norm(img_feats)

            elif modality_name == "forces":
                # Process force inputs.
                force_feats = net_module(batch_dict["forces"])
                force_feats = self.force_norm(force_feats)

        # 2) Classification branch: fuse image and force features.
        fused_feat = self.fuse_for_classification(img_feats, force_feats)
        logits = self.classifier_head(fused_feat)

        # 3) Regression branch: use force-only features.
        reg_out = self.regressor_head(force_feats)

        return logits, reg_out

    def training_step(self, batch, batch_idx):
        """
        Execute a training step.

        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Combined loss for the training step.
        """
        logits, reg_out = self(batch)
        class_loss = self.ce_loss(logits, batch["class_label"])
        mse = self.mse_loss(reg_out, batch["reg_label"].unsqueeze(-1))

        preds = torch.argmax(logits, dim=-1)
        self.train_acc(preds, batch["class_label"])
        self.train_f1(preds, batch["class_label"])
        self.train_r2.update(reg_out.view(-1), batch["reg_label"].view(-1))

        loss = self.alpha_class * class_loss + self.alpha_reg * mse
        self.train_loss(loss)

        self.log("train/loss", self.train_loss, on_epoch=True, prog_bar=True)
        self.log("train/class_loss", class_loss, on_epoch=True)
        self.log("train/mse_loss", mse, on_epoch=True)
        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        """
        Called at the end of a training epoch.

        Logs and resets training metrics.
        """
        r2_val = self.train_r2.compute()
        self.log("train/r2", r2_val, on_epoch=True)

        self.train_acc.reset()
        self.train_f1.reset()
        self.train_r2.reset()
        self.train_loss.reset()

    def validation_step(self, batch, batch_idx):
        """
        Execute a validation step.

        Args:
            batch (dict): Input validation batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Combined loss for the validation step.
        """
        logits, reg_out = self(batch)
        class_loss = self.ce_loss(logits, batch["class_label"])
        mse = self.mse_loss(reg_out, batch["reg_label"].unsqueeze(-1))

        preds = torch.argmax(logits, dim=-1)
        self.val_acc(preds, batch["class_label"])
        self.val_f1(preds, batch["class_label"])
        self.val_r2.update(reg_out.view(-1), batch["reg_label"].view(-1))

        loss = self.alpha_class * class_loss + self.alpha_reg * mse
        self.val_loss(loss)

        self.log("val/loss", self.val_loss, on_epoch=True, prog_bar=True)
        self.log("val/class_loss", class_loss, on_epoch=True)
        self.log("val/mse_loss", mse, on_epoch=True)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        """
        Called at the end of a validation epoch.

        Logs best metrics and resets validation metrics.
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

    def test_step(self, batch, batch_idx):
        """
        Execute a test step.

        Args:
            batch (dict): Input test batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Combined loss for the test step.
        """
        logits, reg_out = self(batch)
        class_loss = self.ce_loss(logits, batch["class_label"])
        mse = self.mse_loss(reg_out, batch["reg_label"].unsqueeze(-1))

        preds = torch.argmax(logits, dim=-1)
        self.test_acc(preds, batch["class_label"])
        self.test_f1(preds, batch["class_label"])
        self.test_r2.update(reg_out.view(-1), batch["reg_label"].view(-1))

        loss = self.alpha_class * class_loss + self.alpha_reg * mse
        self.test_loss(loss)

        self.log("test/loss", self.test_loss, on_epoch=True, prog_bar=True)
        self.log("test/class_loss", class_loss, on_epoch=True)
        self.log("test/mse_loss", mse, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        """
        Called at the end of a test epoch.

        Logs final test metrics and resets test metrics.
        """
        self.log("test/acc", self.test_acc.compute(), prog_bar=True)
        self.log("test/f1", self.test_f1.compute(), prog_bar=True)

        r2_val = self.test_r2.compute()
        self.log("test/r2", r2_val, prog_bar=True)

        self.test_acc.reset()
        self.test_f1.reset()
        self.test_r2.reset()
        self.test_loss.reset()

    def configure_optimizers(self):
        """
        Configure and return the optimizer and learning rate scheduler.

        Returns:
            dict: Dictionary containing the optimizer (and scheduler if provided).
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
