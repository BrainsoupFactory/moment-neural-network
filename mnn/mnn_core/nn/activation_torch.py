# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from ..torch_activation import constant_current_activation
from ..torch_activation import mnn_activation_with_correlation
from ..torch_activation import mnn_activation_without_correlation
from . import functional_torch as functional


class MomentActivation(torch.nn.Module):
    """Torch-first moment activation layer for mean/covariance inputs."""

    def forward(self, *args) -> Tuple[Tensor, Tensor]:
        mean, covariance = functional.parse_moment_input(args)
        if mean.size(-1) != 1 and covariance.dim() > mean.dim():
            std, correlation = functional.compute_correlation(covariance)
            mean, std, correlation = mnn_activation_with_correlation(mean, std, correlation)
            covariance = functional.compute_covariance(std, correlation)
        else:
            std = torch.sqrt(covariance)
            mean, std = mnn_activation_without_correlation(mean, std)
            covariance = std.pow(2)
        return mean, covariance


class ConstantCurrentActivationTorch(torch.nn.Module):
    """Torch-first constant-current activation."""

    def __init__(
        self,
        threshold_voltage: float = 20.0,
        membrane_conductance: float = 0.05,
        refractory_time: float = 5.0,
    ) -> None:
        super().__init__()
        self.threshold_voltage = threshold_voltage
        self.membrane_conductance = membrane_conductance
        self.refractory_time = refractory_time

    def forward(self, current: Tensor) -> Tensor:
        return constant_current_activation(
            current,
            self.threshold_voltage,
            self.membrane_conductance,
            self.refractory_time,
        )

    def extra_repr(self) -> str:
        return (
            f"threshold_voltage={self.threshold_voltage}, "
            f"membrane_conductance={self.membrane_conductance}, "
            f"refractory_time={self.refractory_time}"
        )
