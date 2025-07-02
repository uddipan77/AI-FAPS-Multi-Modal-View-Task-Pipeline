from pathlib import Path
import os

import pytest
import torch

from src.data.mnist_datamodule import MNISTDataModule
from src.data.linear_winding_datamodule import LinearWindingDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size):
    data_dir = os.path.join(os.getcwd(), "data", "MNIST")

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [32, 128])
def test_linear_winding_datamodule(batch_size):
    data_dir = os.path.join(os.getcwd(), "data", "LinearWinding")

    dm = LinearWindingDataModule(data_dir=data_dir, batch_size=batch_size)

    assert not dm.train_dataset and not dm.validation_dataset and not dm.test_dataset
    assert Path(data_dir, "train").exists()
    assert Path(data_dir, "test").exists()

    dm.setup()
    assert dm.train_dataset and dm.validation_dataset and dm.test_dataset
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    
    assert len(batch['image']) == batch_size
    assert len(batch['force']) == batch_size
    assert len(batch['label']) == batch_size
