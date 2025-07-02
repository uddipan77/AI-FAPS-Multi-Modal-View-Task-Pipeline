from torch import nn
import timm  # Import timm to load CaiT models

class CaiTWrapper(nn.Module):
    """
    Wrapper class for CaiT models loaded from timm, with the ability to freeze and unfreeze layers.
    
    :param model_name: Model name to load from timm. Defaults to 'cait_m48_448'.
    :param unfreeze_layer_count: Number of transformer layers to unfreeze. Default is 0.
    """
    
    def __init__(self, model_name='cait_m48_448', unfreeze_layer_count=0):
        super(CaiTWrapper, self).__init__()
        # Load the model from timm
        self.model = timm.create_model(model_name, pretrained=True)

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
        Assumes model has a transformer-like structure with blocks, modify based on actual model structure.
        """
        if hasattr(self.model, 'blocks'):
            for layer in self.model.blocks[-unfreeze_layer_count:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            print("Model does not have 'blocks' attribute. Unfreezing might require specific adjustments.")

    def forward(self, x):
        return self.model(x)
