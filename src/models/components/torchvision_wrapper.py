from torch import nn
from torchvision import models


class TorchVisionWrapper(nn.Module):
    """Wrapper class for torchvision models with the ability to freeze and unfreeze layers.
    
    :param model_name: Name of the torchvision model to use. Default is 'resnet18'.
    :param weights: Weights to use for the model. Default is 'DEFAULT'.
    :param unfreeze_layer_count: Number of layers to unfreeze. Default is 0.
    :param model_kwargs: Additional keyword arguments for the model.
    """
    
    def __init__(self, model_name='resnet18', weights='DEFAULT', unfreeze_layer_count=0, **model_kwargs):
        
        super(TorchVisionWrapper, self).__init__()
        
        # Load the model from torchvision with the specified weights and additional keyword arguments
        self.model = models.get_model(model_name, weights=weights, **model_kwargs)

        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last 'unfreeze_layer_count' layers
        if unfreeze_layer_count > 0:
            self._unfreeze_layers(unfreeze_layer_count)

    def _unfreeze_layers(self, unfreeze_layer_count):
        # Flatten the model into a list of layers
        layers = [module for module in self.model.modules() if len(list(module.children())) == 0]

        # Unfreeze the specified number of layers from the end
        for layer in layers[-unfreeze_layer_count:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)


#The models.get_model function is a utility introduced in more recent versions of the torchvision library. 
#It is designed to provide a more flexible and standardized way to load pretrained models from the torchvision.models module.

#1. Transfer Learning
#One of the most common uses of pretrained models is transfer learning. In this approach, you take a model that has been pretrained on a large dataset (like ImageNet) and adapt it to your specific task. Typically, you do this by:

#Freezing most of the pretrained layers to retain the knowledge the model has already learned.
#Unfreezing the final layers to fine-tune them on your specific dataset.