# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from .activation_torch import MomentActivation
from .batch_norm_torch import MomentBatchNorm1d, MomentBatchNorm1dNoCorrelation
from .custom_batch_norm_torch import CustomMomentBatchNorm1d
from .linear_torch import MomentLinear, MomentLinearNoCorrelation


class MomentBlock(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        linear_bias: bool = False,
        bias_variance: bool = False,
        dropout: Optional[float] = None,
        sparse_degree: Optional[int] = None,
        norm_type: str = "custom",
        special_init: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.norm_type = norm_type
        self.dropout = dropout
        self.special_init = special_init
        self.linear = MomentLinear(
            in_features,
            out_features,
            bias=linear_bias,
            dropout=dropout,
            sparse_degree=sparse_degree,
        )
        if norm_type == "custom":
            self.norm = CustomMomentBatchNorm1d(
                out_features,
                bias_variance=bias_variance,
                special_init=special_init,
                **kwargs,
            )
        else:
            self.norm = MomentBatchNorm1d(
                out_features,
                bias_variance=bias_variance,
                special_init=special_init,
                **kwargs,
            )
        self.activation = MomentActivation()

    def forward(self, mean: Tensor, covariance: Tensor) -> Tuple[Tensor, Tensor]:
        mean, covariance = self.linear(mean, covariance)
        mean, covariance = self.norm(mean, covariance)
        return self.activation(mean, covariance)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"dropout={self.dropout}, norm_type={self.norm_type}, special_init={self.special_init}"
        )


class MomentBlockNoCorrelation(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias_std: bool = True,
        special_init: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.special_init = special_init
        self.linear = MomentLinearNoCorrelation(in_features, out_features)
        self.norm = MomentBatchNorm1dNoCorrelation(out_features, bias_std=bias_std, special_init=special_init)
        self.activation = MomentActivation()

    def forward(self, mean: Tensor, std: Tensor) -> Tuple[Tensor, Tensor]:
        mean, std = self.linear(mean, std)
        mean, std = self.norm(mean, std)
        return self.activation(mean, std)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"special_init={self.special_init}"
        )
