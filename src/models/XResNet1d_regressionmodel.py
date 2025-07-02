from lightning import LightningModule
import torch
import torch.nn.functional as F
from torchmetrics import MaxMetric, R2Score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
from tsai.models.XResNet1d import (
    xresnet1d18, xresnet1d34, xresnet1d50, xresnet1d101, xresnet1d152,
    xresnet1d18_deep, xresnet1d34_deep, xresnet1d50_deep,
    xresnet1d18_deeper, xresnet1d34_deeper, xresnet1d50_deeper
)


class XResNet1dRegressionModel(LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        model_depth: str = "xresnet1d50_deep",
        optimizer=None,
        scheduler=None,
    ):
        """
        Args:
            input_dim (int): Number of input channels (e.g., 5).
            output_dim (int): Number of output features (1 for regression).
            model_depth (str): Which XResNet1d architecture to use.
            optimizer: Optimizer configuration.
            scheduler: Scheduler configuration.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Map from string -> tsai xresnet1d function
        model_map = {
            "xresnet1d18": xresnet1d18,
            "xresnet1d34": xresnet1d34,
            "xresnet1d50": xresnet1d50,
            "xresnet1d101": xresnet1d101,
            "xresnet1d152": xresnet1d152,
            "xresnet1d18_deep": xresnet1d18_deep,
            "xresnet1d34_deep": xresnet1d34_deep,
            "xresnet1d50_deep": xresnet1d50_deep,
            "xresnet1d18_deeper": xresnet1d18_deeper,
            "xresnet1d34_deeper": xresnet1d34_deeper,
            "xresnet1d50_deeper": xresnet1d50_deeper,
        }

        if model_depth not in model_map:
            raise ValueError(
                f"Unsupported model_depth={model_depth}. "
                f"Choose from {list(model_map.keys())}"
            )

        # Build the XResNet1d model
        self.model = model_map[model_depth](c_in=input_dim, c_out=output_dim)

        # Metrics
        self.best_val_r2 = MaxMetric()
        self.val_r2_metric = R2Score()
        self.test_r2_metric = R2Score()

    def forward(self, x):
        """Forward pass: x shape = (batch_size, input_dim, seq_len)"""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).squeeze(-1)
        loss = F.mse_loss(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).squeeze(-1)
        loss = F.mse_loss(y_pred, y)
        self.val_r2_metric.update(y_pred, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        epoch_val_r2 = self.val_r2_metric.compute()
        self.best_val_r2.update(epoch_val_r2)
        self.log("epoch_val_r2", epoch_val_r2, on_epoch=True, prog_bar=True)
        self.log("best_val_r2", self.best_val_r2.compute(), on_epoch=True, prog_bar=True)
        self.val_r2_metric.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).squeeze(-1)
        loss = F.mse_loss(y_pred, y)
        self.test_r2_metric.update(y_pred, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        epoch_test_r2 = self.test_r2_metric.compute()
        self.log("test_r2", epoch_test_r2, on_epoch=True, prog_bar=True)
        self.test_r2_metric.reset()

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
