from typing import Any, List

import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    """Simple classifier model using fully connected layers with dynamic hidden layers.

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
            
            print(in_channels, hidden_channel)
            block = self.create_block(in_channels, hidden_channel)
            blocks.append(block)
            
            # Update in_channels for next block
            in_channels = hidden_channel

        # Add the classifier output layer
        self.net = torch.nn.Sequential(
            *blocks,
            nn.Linear(in_channels, output_size)
        )
        
    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out

    def create_block(self, in_channels, out_channels):
        """Create a block with a linear layer and sigmoid activation function."""
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid()
        )