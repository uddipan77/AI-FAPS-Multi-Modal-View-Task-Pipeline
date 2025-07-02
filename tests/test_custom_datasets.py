from pathlib import Path
import os

import torchvision.transforms as T
import numpy as np
import pytest
import torch

from src.data.components.linear_winding_dataset import LinearWindingDataset
from src.data.components.linear_winding_transforms import ToTensor


@pytest.mark.parametrize("train", [True, False])
def test_linear_winding_dataset(train):
    
    data_dir = os.path.join(os.getcwd(), "data", "LinearWinding")

    ds = LinearWindingDataset(root_dir=data_dir, train=train, transform=ToTensor())    
    
    assert len(ds) > 0
    
    sample = ds[0]
    
    assert type(sample['image']) == torch.Tensor
    assert type(sample['force']) == torch.Tensor
    assert type(sample['label']) == str
    
