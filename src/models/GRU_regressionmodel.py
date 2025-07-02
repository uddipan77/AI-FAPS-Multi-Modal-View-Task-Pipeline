from lightning import LightningModule
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MaxMetric, R2Score
from tsai.models.RNN import GRU  # Import GRU from tsai


class GRURegressionModel(LightningModule):
    def __init__(
        self, 
        input_dim, 
        output_dim=1, 
        hidden_size=100, 
        n_layers=1, 
        bias=True, 
        rnn_dropout=0.0, 
        bidirectional=False, 
        fc_dropout=0.0, 
        optimizer=None,
        scheduler=None,
        **kwargs
    ):
        """
        GRURegressionModel for regression tasks.

        Args:
            input_dim (int): Number of input channels.
            output_dim (int): Number of output features (1 for regression).
            hidden_size (int): Hidden size of GRU layers.
            n_layers (int): Number of GRU layers.
            bias (bool): Whether to use bias in GRU layers.
            rnn_dropout (float): Dropout applied between GRU layers.
            bidirectional (bool): Whether GRU is bidirectional.
            fc_dropout (float): Dropout applied before the fully connected layer.
            optimizer (dict): Optimizer configuration.
            scheduler (dict): Scheduler configuration.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        # Define the GRU model
        self.model = GRU(
            c_in=input_dim, 
            c_out=output_dim, 
            hidden_size=hidden_size, 
            n_layers=n_layers, 
            bias=bias, 
            rnn_dropout=rnn_dropout, 
            bidirectional=bidirectional, 
            fc_dropout=fc_dropout,
            **kwargs
        )

        # Metrics
        self.best_val_r2 = MaxMetric()
        self.val_r2_metric = R2Score()
        self.test_r2_metric = R2Score()

    def forward(self, x):
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
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
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
