"""
Module for MultiTaskModule.

This module defines the MultiTaskModule class, which is a multi-task Lightning
module that outputs classification logits (for a 2-class or multi-class problem)
and a regression scalar (for DC resistance). It uses force and image features for
classification, and force-only for regression. Additionally, LayerNorm is applied
to the image and force features before fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning import LightningModule
from torchmetrics import Accuracy, MeanMetric, F1Score, R2Score, MaxMetric


class MultiTaskModule(LightningModule):
    """
    Multi-task Lightning module with both classification and regression outputs.

    This module uses force and image features for classification and force-only features
    for regression. It supports two fusion types for the image features ('concat' or 'mean')
    and applies LayerNorm normalization before fusing the features.
    """

    def __init__(
        self,
        modal_nets: dict,
        fusion_type: str = "concat",  # "concat" or "mean"
        num_classes: int = 2,
        optimizer_cfg: dict = None,
        scheduler_cfg: dict = None,
        alpha_classification: float = 1.0,
        alpha_regression: float = 1.0,
        image_feature_dim: int = 1000,  # Depends on your CNN architecture (e.g., DenseNet)
        force_feature_dim: int = 64,    # From InceptionTime or other force feature extractor
        weight_f1: float = 0.5,
        weight_r2: float = 0.5,
        **kwargs
    ):
        """
        Initialize the MultiTaskModule.

        Args:
            modal_nets (dict): Dictionary of sub-networks for each modality.
            fusion_type (str, optional): Fusion method to combine image features,
                either "concat" or "mean". Defaults to "concat".
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
            weight_f1 (float, optional): Weight factor for F1 metric.
                Defaults to 0.5.
            weight_r2 (float, optional): Weight factor for R2 metric.
                Defaults to 0.5.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Sub-networks for each modality.
        self.net = nn.ModuleDict({k: v for k, v in modal_nets.items()})
        self.fusion_type = fusion_type.lower()

        # Validate fusion_type.
        if self.fusion_type not in ["concat", "mean"]:
            raise ValueError(
                f"Unsupported fusion_type: {fusion_type}. "
                "Choose 'concat' or 'mean'."
            )

        # Store feature dimensions.
        self.force_feature_dim = force_feature_dim
        self.image_feature_dim = image_feature_dim

        # Determine fused dimension and define image LayerNorm.
        if self.fusion_type == "concat":
            # 4 image views => final image features shape: [B, 4 * image_feature_dim]
            self.fused_feature_dim = 4 * self.image_feature_dim + self.force_feature_dim
            self.image_norm = nn.LayerNorm(self.image_feature_dim * 4)
        else:
            # "mean" => final image features shape: [B, image_feature_dim]
            self.fused_feature_dim = self.image_feature_dim + self.force_feature_dim
            self.image_norm = nn.LayerNorm(self.image_feature_dim)

        # Force normalization.
        self.force_norm = nn.LayerNorm(self.force_feature_dim)

        # Define classification head.
        if self.fusion_type == "concat":
            self.classifier_head = nn.Sequential(
                nn.Linear(self.fused_feature_dim, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1024, num_classes)
            )
        elif self.fusion_type == "mean":
            self.classifier_head = nn.Linear(self.fused_feature_dim, num_classes)

        # Define regression head.
        self.regressor_head = nn.Linear(self.force_feature_dim, 1)

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

        # Best metric trackers.
        self.val_f1_best = MaxMetric()
        self.val_r2_best = MaxMetric()

        # Mean metrics for logging losses.
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Weights for multi-task loss.
        self.alpha_class = alpha_classification
        self.alpha_reg = alpha_regression

        # Weights for combined metrics.
        self.weight_f1 = weight_f1
        self.weight_r2 = weight_r2

    def forward(self, batch_dict):
        """
        Forward pass through the network.

        Args:
            batch_dict (dict): Dictionary containing:
                "images": [4 x Tensor(B, 3, H, W)] - Multi-view images,
                "forces": Tensor(B, 5, seq_len) - Force time series,
                "class_label": Tensor(B) - Classification labels,
                "reg_label": Tensor(B) - Regression labels, etc.

        Returns:
            tuple: A tuple containing:
                - logits (Tensor): [B, num_classes] classification output.
                - reg_out (Tensor): [B, 1] regression output.
        """
        img_feats = None
        force_feats = None

        # Gather features for each modality.
        for modality_name, net_module in self.net.items():
            if modality_name.lower() == "images":
                views = batch_dict["images"]
                view_feats = []
                for img_tensor in views:
                    # Each feature has shape: [B, image_feature_dim]
                    feat = net_module(img_tensor)
                    view_feats.append(feat)

                if self.fusion_type == "concat":
                    # Concatenate image features [B, 4 * image_feature_dim].
                    img_feats = torch.cat(view_feats, dim=-1)
                    img_feats = self.image_norm(img_feats)
                else:
                    # Average image features [B, image_feature_dim].
                    img_feats_stacked = torch.stack(view_feats, dim=1)  # [B, 4, image_feature_dim]
                    img_feats = img_feats_stacked.mean(dim=1)
                    img_feats = self.image_norm(img_feats)

            elif modality_name.lower() == "forces":
                # Process force features.
                force_feats = net_module(batch_dict["forces"])
                force_feats = self.force_norm(force_feats)

        # Fuse features.
        if img_feats is not None and force_feats is not None:
            fused_feat = torch.cat([img_feats, force_feats], dim=-1)
        elif img_feats is not None:
            fused_feat = img_feats
        else:
            fused_feat = force_feats

        # Classification prediction.
        logits = self.classifier_head(fused_feat)
        # Regression prediction (using force features only).
        reg_out = self.regressor_head(force_feats)

        return logits, reg_out

    def training_step(self, batch, batch_idx):
        """
        Execute one training step for a batch.

        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Combined loss for the batch.
        """
        logits, reg_out = self(batch)

        # Compute losses.
        class_loss = self.ce_loss(logits, batch["class_label"])
        mse = self.mse_loss(reg_out, batch["reg_label"].unsqueeze(-1))

        # Compute predictions for metrics.
        preds = torch.argmax(logits, dim=-1)

        # Update training metrics.
        self.train_acc(preds, batch["class_label"])
        self.train_f1(preds, batch["class_label"])
        self.train_r2.update(reg_out.view(-1), batch["reg_label"].view(-1))

        # Combine losses.
        loss = self.alpha_class * class_loss + self.alpha_reg * mse
        self.train_loss(loss)

        # Logging.
        self.log("train/loss", self.train_loss, on_epoch=True, prog_bar=True)
        self.log("train/class_loss", class_loss, on_epoch=True)
        self.log("train/mse_loss", mse, on_epoch=True)
        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        """
        Called at the end of the training epoch.

        Computes R^2 metric and resets training metrics.
        """
        r2_val = self.train_r2.compute()
        self.log("train/r2", r2_val, on_epoch=True)
        self.train_r2.reset()
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_loss.reset()

    def validation_step(self, batch, batch_idx):
        """
        Execute one validation step for a batch.

        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Combined loss for the validation batch.
        """
        logits, reg_out = self(batch)

        class_loss = self.ce_loss(logits, batch["class_label"])
        mse = self.mse_loss(reg_out, batch["reg_label"].unsqueeze(-1))

        preds = torch.argmax(logits, dim=-1)

        # Update validation metrics.
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
        Called at the end of the validation epoch.

        Computes best metrics, logs weighted metric, and resets validation metrics.
        """
        f1_score = self.val_f1.compute()
        self.val_f1_best.update(f1_score)
        self.log("val/f1_best", self.val_f1_best.compute(), prog_bar=True)

        r2_val = self.val_r2.compute()
        self.val_r2_best.update(r2_val)
        self.log("val/r2_best", self.val_r2_best.compute(), prog_bar=True)

        # Compute weighted metric.
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
        Execute one test step for a batch.

        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Combined loss for the test batch.
        """
        logits, reg_out = self(batch)

        class_loss = self.ce_loss(logits, batch["class_label"])
        mse = self.mse_loss(reg_out, batch["reg_label"].unsqueeze(-1))

        preds = torch.argmax(logits, dim=-1)

        # Update test metrics.
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
        Called at the end of the test epoch.

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
        Configure optimizers and (optionally) learning rate schedulers.

        Returns:
            dict: Dictionary containing the optimizer configuration and, if provided,
                  the scheduler configuration.
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
                    "monitor": "val/loss_epoch",  # e.g., "val/f1_best" or "val/r2_best"
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
