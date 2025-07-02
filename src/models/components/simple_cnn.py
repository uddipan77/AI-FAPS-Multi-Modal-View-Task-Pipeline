import torch
from torch import nn


class NeuralNetBlock(nn.Module):
    """A simple neural network block with convolutional layer, activation function, max pooling, batch normalization, and dropout.
    
    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param kernel_size: Kernel size for the convolutional layer. Default is 3.
    :param stride: Stride for the convolutional layer. Default is 1.
    :param padding: Padding for the convolutional layer. Default is 1.
    :param pool_kernel_size: Kernel size for the max pooling layer. Default is 2.
    :param pool_stride: Stride for the max pooling layer. Default is 2.
    :param dropout_p: Dropout probability. Default is 0.3.
    :param use_batchnorm: Whether to use batch normalization. Default is True.
    :param activation_func: Activation function to use. Default is nn.ReLU.
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1, 
                 pool_kernel_size=2, 
                 pool_stride=2, 
                 dropout_p=0.3, 
                 use_batchnorm=True, 
                 activation_func=nn.ReLU):
        super().__init__()

        # Initialize the layers list
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            activation_func(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        ]

        # Optionally include Batch Normalization
        if use_batchnorm:
            layers.insert(2, nn.BatchNorm2d(out_channels))

        # Add dropout
        layers.append(nn.Dropout2d(p=dropout_p))

        # Create the sequential block
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SimpleConvNeuralNet(nn.Module):
    """Simple convolutional neural network model using convolutional layers with dynamic hidden layers.
    
    :param input_size: Size of the input image. Default is 224.
    :param in_channels: Number of input channels. Default is 3.
    :param hidden_channels: List of hidden channels. Default is [16, 32, 64].
    :param output_size: Number of output features. Default is 2.
    :param kernel_size: Kernel size for the convolutional layer. Default is 3.
    :param stride: Stride for the convolutional layer. Default is 1.
    :param padding: Padding for the convolutional layer. Default is 1.
    :param pool_kernel_size: Kernel size for the max pooling layer. Default is 2.
    :param pool_stride: Stride for the max pooling layer. Default is 2.
    :param dropout_p: Dropout probability. Default is 0.3.
    :param use_batchnorm: Whether to use batch normalization. Default is True.
    """
    
    def __init__(
        self,
        input_size: int = 224,
        in_channels: int = 3,
        hidden_channels: list = [16, 32, 64],
        output_size: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pool_kernel_size: int = 2,
        pool_stride: int = 2,
        dropout_p: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        layers = []
        in_channels_current = in_channels
        image_size = input_size

        # Create the blocks based on number of hidden channels
        for hidden_channel in hidden_channels:
            layers.append(
                NeuralNetBlock(
                    in_channels_current,
                    hidden_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    pool_kernel_size=pool_kernel_size,
                    pool_stride=pool_stride,
                    dropout_p=dropout_p,
                    use_batchnorm=use_batchnorm,
                    activation_func=nn.ReLU,
                )
            )
            
            # Update in_channels_current for next block
            in_channels_current = hidden_channel
            
            # For final features size calculation
            image_size = image_size // 2  # Account for max-pooling reducing image size

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_channels_current * image_size * image_size, output_size)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    # Create an instance of the SimpleConvNeuralNet
    model = SimpleConvNeuralNet()
    print(model)
