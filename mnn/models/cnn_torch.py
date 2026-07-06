# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
from torch import Tensor


class CnnWithPoolingClassifier(torch.nn.Module):
    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        pooling: torch.nn.Module,
        classifier: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.pooling = pooling
        self.classifier = classifier

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)
        x = self.pooling(x)
        return self.classifier(x)
