# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def sample_shape(num_neurons, num_steps=None):
    if num_steps is None:
        return [1, num_neurons]
    if isinstance(num_neurons, int):
        if isinstance(num_steps, int):
            return (num_steps, num_neurons)
        return list(num_steps) + [num_neurons]
    if isinstance(num_steps, int):
        return [num_steps] + list(num_neurons)
    return list(num_steps) + list(num_neurons)


def _prefix_shape(num_neurons, num_steps: int):
    if isinstance(num_neurons, int):
        return [num_steps]
    return [num_steps] + list(num_neurons)[:-1]


def _sample_correlated_gaussian(mean: Tensor, covariance: Tensor, prefix: list[int]) -> Tensor:
    jitter = torch.finfo(mean.dtype).eps if mean.is_floating_point() else 1e-6
    eye = torch.eye(covariance.size(-1), dtype=covariance.dtype, device=covariance.device)
    factor = torch.linalg.cholesky(covariance + eye * jitter)
    noise = torch.randn(*prefix, covariance.size(-1), dtype=mean.dtype, device=mean.device)
    if factor.dim() == 2:
        return torch.matmul(noise, factor.transpose(-1, -2)) + mean
    return torch.einsum("...n,...mn->...m", noise, factor) + mean


def pregenerate_gaussian_current(
    num_neurons,
    num_steps: int,
    mean: Tensor,
    std: Tensor,
    rho: Optional[Tensor] = None,
) -> Tensor:
    if rho is None:
        shape = sample_shape(num_neurons, num_steps)
        return torch.randn(shape, dtype=mean.dtype, device=mean.device) * std + mean
    covariance = torch.matmul(std.unsqueeze(-1), std.unsqueeze(-2)) * rho
    return _sample_correlated_gaussian(mean, covariance, _prefix_shape(num_neurons, num_steps))
