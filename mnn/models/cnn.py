import torch
from torch import Tensor

class GeneralCnnPool(torch.nn.Module):
    def __init__(self, feature_extractor: torch.nn.Module, pool: torch.nn.Module, classifier: torch.nn.Module):
        super(GeneralCnnPool, self).__init__()
        self.feature_extractor = feature_extractor
        self.pool = pool
        self.classifier = classifier

    def forward(self, x: Tensor):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x