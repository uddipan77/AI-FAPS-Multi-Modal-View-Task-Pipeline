import torch
import torch.nn as nn
from tsai.models.ROCKET import RocketClassifier

import torch
import torch.nn as nn
from tsai.models.ROCKET import RocketClassifier  # Make sure this is correct

class RocketWrapper(nn.Module):
    def __init__(self, num_kernels=10000, normalize_input=True, normalize_features=False, random_state=42, alphas=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        super(RocketWrapper, self).__init__()
        self.rocket = RocketClassifier(num_kernels=num_kernels, random_state=random_state, normalize_input=normalize_input, normalize_features=normalize_features)
        self.alphas = alphas
        self.fitted = False  # Track whether the model is fitted

    def forward(self, x):
        if not self.fitted:
            raise RuntimeError("RocketWrapper has not been fitted yet. Call 'fit' before 'forward'.")
        return self.rocket.predict(x)

    def fit(self, x_train):
        self.rocket.fit(x_train)
        self.fitted = True

