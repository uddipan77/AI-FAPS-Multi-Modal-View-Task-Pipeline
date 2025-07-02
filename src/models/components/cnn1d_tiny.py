from typing import Any, List

import torch
import torch.nn as nn


class CNN1DTiny(nn.Module):
    """Simple CNN1D model using convolutional layers with dynamic hidden layers.
    
    :param input_size: Number of input features.
    :param in_channels: Number of input channels.
    :param hidden_channels: List of hidden channels.
    :param output_size: Number of output features.
    """

    def __init__(
        self,
        input_size: int,
        in_channels: int,
        hidden_channels: List[int],
        output_size: int,
    ):
        super().__init__()

        self.input_size = input_size

        blocks = []
        
        # Create the blocks based on number of hidden channels
        for hidden_channel in hidden_channels:
            
            block = self.create_block(in_channels, hidden_channel)
            blocks.append(block)
            
            # Update in_channels for next block
            in_channels = hidden_channel

        # Add the classifier output layer with a Flatten layer
        self.net = torch.nn.Sequential(
            *blocks,
            nn.Flatten(),
            nn.Linear(in_channels * input_size, output_size)
        )
    
    def create_block(self, in_channels, out_channels):
        """Create a block with a convolutional layer, batch normalization and ReLU activation function."""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out
