from lightning import LightningModule
import torch
import torch.nn.functional as F
from torchmetrics import MaxMetric, R2Score
from tsai.models.InceptionTime import InceptionTime


class InceptionTimeRegressionModel(LightningModule):
    def __init__(
        self,
        input_dim,
        output_dim=1,
        nf=32,
        seq_len=2881,
        depth=6,
        bottleneck=True,
        optimizer=None,
        scheduler=None,
    ):
        """
        InceptionTime model for regression tasks, adapted to the merged pipeline.

        Args:
            input_dim (int): Number of input channels (e.g., 5 for 5 subsegments).
            output_dim (int): Number of output units (1 for regression).
            nf (int): Base number of filters.
            seq_len (int): Input sequence length.
            depth (int): Depth of InceptionTime network.
            bottleneck (bool): Whether to use bottleneck layers.
            optimizer: Optimizer configuration.
            scheduler: Scheduler configuration.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Build the InceptionTime model
        self.model = InceptionTime(
            c_in=input_dim,
            c_out=output_dim,
            nf=nf,
            depth=depth,
            bottleneck=bottleneck,
            seq_len=seq_len,
        )

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
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
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
        """
        Configures optimizers and learning-rate schedulers.
        """
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
