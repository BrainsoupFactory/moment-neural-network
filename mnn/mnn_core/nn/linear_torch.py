# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter

from . import functional_torch as functional


class MomentLinear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        bias_variance: bool = False,
        scale: Optional[float] = None,
        dropout: Optional[float] = None,
        sparse_degree: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self.weight = Parameter(torch.empty(out_features, in_features))
        self.sparse_degree = sparse_degree

        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        if bias_variance:
            self.bias_variance = Parameter(torch.ones(out_features))
        else:
            self.register_parameter("bias_variance", None)

        self.reset_parameters()
        self._register_sparse_mask()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        if self.bias_variance is not None:
            with torch.no_grad():
                self.bias_variance.fill_(1.0)

    def _register_sparse_mask(self) -> None:
        if self.sparse_degree is None:
            self.register_buffer("sparse_mask", None)
            return
        if not 0 < int(self.sparse_degree) <= self.in_features:
            raise ValueError("sparse_degree must be in (0, in_features]")
        mask = torch.zeros(self.out_features, self.in_features)
        for row in range(self.out_features):
            indices = torch.randperm(self.in_features)[: int(self.sparse_degree)]
            mask[row, indices] = 1.0
        self.register_buffer("sparse_mask", mask)

    def forward(self, *args) -> Tuple[Tensor, Tensor]:
        mean, covariance = functional.parse_moment_input(args)
        weight = self.weight if self.sparse_mask is None else self.weight * self.sparse_mask
        return functional.moment_linear(
            mean,
            covariance,
            weight,
            self.bias,
            self.bias_variance,
            self.dropout,
            self.scale,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bias_variance={self.bias_variance is not None}, "
            f"dropout={self.dropout is not None}, scale={self.scale}, sparse_degree={self.sparse_degree}"
        )


class MomentLinearNoCorrelation(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        bias_variance: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        if bias_variance:
            self.bias_variance = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_variance", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            if self.bias_variance is not None:
                init.uniform_(self.bias_variance, 0, bound)

    def forward(self, *args) -> Tuple[Tensor, Tensor]:
        mean, std = functional.parse_moment_input(args)
        output_mean = F.linear(mean, self.weight, self.bias)
        bias_variance = None if self.bias_variance is None else F.softplus(self.bias_variance)
        output_std = F.linear(std, self.weight.pow(2), bias_variance)
        return output_mean, output_std

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bias_variance={self.bias_variance is not None}"
        )


class Identity(torch.nn.Module):
    def forward(self, *args):
        return args
