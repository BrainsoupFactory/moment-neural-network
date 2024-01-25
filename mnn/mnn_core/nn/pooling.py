# -*- coding: utf-8 -*-
from typing import Tuple

import torch
from torch import Tensor
from . import functional

class MnnPooling(torch.nn.Module):
    def __init__(self, input_dim: int = 256, mask_cov: bool = False, biological_pooling=True):
        super(MnnPooling, self).__init__()
        self.mask_cov = mask_cov
        self.input_dim = input_dim
        self.biological_pooling = biological_pooling
        if mask_cov:
            self.register_buffer('unit_matrix', torch.eye(input_dim))
        else:
            self.register_buffer('unit_matrix', None)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.flatten(x, start_dim=-2)
        mean, cov = functional.mean_cov_pooling(x, biological_pooling=self.biological_pooling)
        if self.mask_cov:
            cov = cov * self.unit_matrix
        return mean, cov