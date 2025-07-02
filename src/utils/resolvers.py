import torchvision.models as models


def torchvision_weights(model_name, weight):
    return models.__dict__[model_name][weight]

def torchvision_transforms(model_name, weight):
    return models.__dict__[model_name][weight].transforms()


all_custom_resolvers = {
    'torchvision_weights': torchvision_weights,
    'torchvision_transforms': torchvision_transforms,
}