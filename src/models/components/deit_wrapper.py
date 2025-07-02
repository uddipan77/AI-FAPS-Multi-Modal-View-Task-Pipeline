from torch import nn
import torch.hub

import timm  # Import timm for version check

# Ensure the correct version of timm is installed
#assert timm.__version__ == "0.3.2" #, "Please install timm version 0.3.2 to use DeiT models"


class DeitWrapper(nn.Module):
    """
    Wrapper class for DeiT models loaded from torch.hub, with the ability to freeze and unfreeze layers.
    
    :param model_name: Model name to load from torch.hub. Defaults to 'deit_base_patch16_224'.
    :param unfreeze_layer_count: Number of transformer layers to unfreeze. Default is 0.
    """
    
    def __init__(self, model_name='deit_base_patch16_224', unfreeze_layer_count=0):
        super(DeitWrapper, self).__init__()
        # Load the model from torch.hub
        self.model = torch.hub.load('facebookresearch/deit:main', model_name, pretrained=True)

        # Freeze all parameters initially
        self.freeze_parameters()

        # Unfreeze the specified transformer layers
        if unfreeze_layer_count > 0:
            self._unfreeze_layers(unfreeze_layer_count)

    def freeze_parameters(self):
        """Freezes all parameters in the model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def _unfreeze_layers(self, unfreeze_layer_count):
        """
        Unfreezes the specified number of transformer layers from the top of the model.
        Assumes model is a nn.Sequential for simplification. Modify based on actual model structure.
        """
        if hasattr(self.model, 'blocks'):
            for layer in self.model.blocks[-unfreeze_layer_count:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            print("Model does not have 'blocks' attribute. Unfreezing might require specific adjustments.")

    def forward(self, x):
        return self.model(x)
