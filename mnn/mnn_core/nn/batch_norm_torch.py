# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter

from . import functional_torch as functional


class MomentBatchNorm1d(torch.nn.Module):
    __constants__ = ["num_features"]

    def __init__(
        self,
        num_features: int,
        *,
        bias_variance: bool = False,
        special_init: bool = True,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.batch_norm_mean = torch.nn.BatchNorm1d(num_features)
        self.scale = scale
        self.special_init = special_init
        if special_init:
            with torch.no_grad():
                self.batch_norm_mean.weight.fill_(2.5)
                self.batch_norm_mean.bias.fill_(2.5)
        if bias_variance:
            self.bias_variance = Parameter(torch.ones(num_features))
        else:
            self.register_parameter("bias_variance", None)

    def forward(self, *args) -> Tuple[Tensor, Tensor]:
        mean, covariance = functional.parse_moment_input(args)
        return functional.batch_norm_1d_moments(
            self.batch_norm_mean,
            mean,
            covariance,
            self.batch_norm_mean.weight,
            self.bias_variance,
            self.scale,
        )

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, bias_variance={self.bias_variance is not None}, "
            f"special_init={self.special_init}, scale={self.scale}"
        )


class MomentBatchNorm1dNoCorrelation(torch.nn.Module):
    __constants__ = ["num_features"]

    def __init__(
        self,
        num_features: int,
        *,
        bias_std: bool = False,
        special_init: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.special_init = special_init
        self.batch_norm_mean = torch.nn.BatchNorm1d(num_features)
        if special_init:
            with torch.no_grad():
                self.batch_norm_mean.weight.fill_(2.5)
                self.batch_norm_mean.bias.fill_(2.5)
        if bias_std:
            self.bias_std = Parameter(torch.empty(num_features))
        else:
            self.register_parameter("bias_std", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.bias_std is not None:
            if self.special_init:
                init.uniform_(self.bias_std, 2, 10)
            else:
                init.zeros_(self.bias_std)

    def forward(self, *args) -> Tuple[Tensor, Tensor]:
        mean, std = functional.parse_moment_input(args)
        output_mean = self.batch_norm_mean(mean)
        if self.batch_norm_mean.training or not self.batch_norm_mean.track_running_stats:
            variance = std.pow(2) * self.batch_norm_mean.weight.pow(2) / (
                torch.var(mean, dim=0, keepdim=True) + self.batch_norm_mean.eps
            )
        else:
            variance = std.pow(2) * self.batch_norm_mean.weight.pow(2) / (
                self.batch_norm_mean.running_var + self.batch_norm_mean.eps
            )
        if self.bias_std is not None:
            variance = variance + self.bias_std.pow(2)
        return output_mean, torch.sqrt(variance)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, bias_std={self.bias_std is not None}, "
            f"special_init={self.special_init}"
        )
