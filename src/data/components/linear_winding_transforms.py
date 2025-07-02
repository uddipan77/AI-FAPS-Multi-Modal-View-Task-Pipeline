import torchvision.transforms as T
import torch


class MinMaxNormalize(object):

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        min_val = tensor.min(dim=0, keepdim=True)[0]
        max_val = tensor.max(dim=0, keepdim=True)[0]
        return (tensor - min_val) / (max_val - min_val)
