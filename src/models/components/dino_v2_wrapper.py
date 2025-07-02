from torch import nn
import torch.hub

class DinoV2Wrapper(nn.Module):
    """
    Wrapper class for DINO V2 models loaded from torch.hub, with the ability to freeze and unfreeze layers.
    
    :param model_name: Model name to load from torch.hub. Defaults to 'dinov2_vits14'.
    :param unfreeze_layer_count: Number of transformer layers to unfreeze. Default is 0.
    """
    
    def __init__(self, model_name='dinov2_vits14', unfreeze_layer_count=0):
        super(DinoV2Wrapper, self).__init__()
        # Load the model from torch.hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)

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
        Unfreeze the specified number of transformer layers from the top of the model.
        Assumes model is a nn.Sequential for simplification. Adjust based on actual model structure.
        """
        # Since DINO models from torch.hub may not have a straightforward 'encoder.layer' attribute,
        # modify this method based on actual loaded model structure.
        # Here we assume the model is simple sequential for demonstration.
        if hasattr(self.model, 'blocks'):
            for layer in self.model.blocks[-unfreeze_layer_count:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            print("Model does not have 'blocks' attribute. Unfreezing might require specific adjustments.")

    def forward(self, x):
        return self.model(x)
