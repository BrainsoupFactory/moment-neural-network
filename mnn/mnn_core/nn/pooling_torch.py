# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from . import functional_torch as functional


class MomentPooling(torch.nn.Module):
    def __init__(self, input_dim: int = 256, *, mask_covariance: bool = False, biological_pooling: bool = True) -> None:
        super().__init__()
        self.mask_covariance = mask_covariance
        self.input_dim = input_dim
        self.biological_pooling = biological_pooling
        if mask_covariance:
            self.register_buffer("unit_matrix", torch.eye(input_dim))
        else:
            self.register_buffer("unit_matrix", None)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.flatten(x, start_dim=-2)
        mean, covariance = functional.mean_covariance_pooling(x, biological_pooling=self.biological_pooling)
        if self.mask_covariance:
            covariance = covariance * self.unit_matrix.to(device=covariance.device, dtype=covariance.dtype) #type: ignore
        return mean, covariance
