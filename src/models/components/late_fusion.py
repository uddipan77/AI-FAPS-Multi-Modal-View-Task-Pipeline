import torch


class LateFusion(torch.nn.Module):
    """Late fusion model layer for combining multiple modalities.
    
    :param input_features: Number of input features.
    :param output_size: Number of output features.
    :param fusion_model: Model to use for fusion. Default is None, which uses a linear layer.
    :param fusion_type: Type of fusion to use. Default is 'concat'. Other options are not implemented.
    :param fusion_dim: Dimension to concatenate the modalities. Default is 1.
    """
    
    def __init__(self, input_features, output_size, fusion_model=None, fusion_type='concat', fusion_dim=1):
        super().__init__()
        self.fusion_model = torch.nn.Linear(input_features, output_size) if fusion_model is None else fusion_model
        self.fusion_type = fusion_type
        self.fusion_dim = fusion_dim

    def forward(self, *modalities):
        
        if self.fusion_type == 'concat':
            out = torch.cat(modalities, dim=self.fusion_dim)
        else:
            raise NotImplementedError(f"Fusion type '{self.fusion_type}' is not implemented.")
        
        out = self.fusion_model(out)
        
        return out