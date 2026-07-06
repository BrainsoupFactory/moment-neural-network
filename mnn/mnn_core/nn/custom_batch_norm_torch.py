# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from . import functional_torch as functional


def _compute_weight(gamma: Optional[Tensor], variance: Tensor, eps: float = 1e-5) -> Tensor:
    if gamma is None:
        return 1 / torch.sqrt(variance + eps)
    return gamma / torch.sqrt(variance + eps)


@torch.no_grad()
def _update_running_state(
    mean: Tensor,
    variance: Tensor,
    running_mean: Tensor,
    running_variance: Tensor,
    momentum: float = 0.9,
) -> tuple[Tensor, Tensor]:
    running_mean.mul_(momentum).add_(mean, alpha=1 - momentum)
    running_variance.mul_(momentum).add_(variance, alpha=1 - momentum)
    return running_mean, running_variance


@torch.no_grad()
def _update_mean_variance(variance: Tensor, mean_variance: Optional[Tensor], momentum: float = 0.9) -> Optional[Tensor]:
    if mean_variance is not None:
        mean_variance.mul_(momentum).add_(variance.reshape_as(mean_variance), alpha=1 - momentum)
    return mean_variance


def _batch_norm_train(
    mean: Tensor,
    covariance: Tensor,
    gamma: Optional[Tensor],
    beta: Optional[Tensor],
    running_mean: Tensor,
    running_variance: Tensor,
    mean_variance: Optional[Tensor] = None,
    beta_variance: Optional[Tensor] = None,
    momentum: float = 0.9,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    mean_variance_batch, mean_batch = torch.var_mean(mean, dim=0, keepdim=True)
    if covariance.dim() > mean.dim():
        variance = torch.diagonal(covariance, dim1=-1, dim2=-2)
    else:
        variance = covariance
    variance = torch.mean(variance, dim=0, keepdim=True)
    mean_variance = _update_mean_variance(variance, mean_variance, momentum)
    total_variance = mean_variance_batch + variance
    _update_running_state(
        mean_batch.reshape_as(running_mean),
        total_variance.reshape_as(running_variance),
        running_mean,
        running_variance,
        momentum,
    )

    norm_weight = _compute_weight(gamma, total_variance, eps)
    normalized_mean = (mean - mean_batch) * norm_weight
    if beta is not None:
        normalized_mean = normalized_mean + beta
    if covariance.dim() > mean.dim():
        normalized_covariance = covariance * torch.matmul(norm_weight.unsqueeze(-1), norm_weight.unsqueeze(-2))
        if beta_variance is not None:
            normalized_covariance = normalized_covariance + functional.variance_to_covariance(beta_variance)
    else:
        normalized_covariance = covariance * norm_weight * norm_weight
        if beta_variance is not None:
            normalized_covariance = normalized_covariance + torch.nn.functional.softplus(beta_variance)
    return normalized_mean, normalized_covariance, running_mean, running_variance, mean_variance


def _batch_norm_eval(
    mean: Tensor,
    covariance: Tensor,
    gamma: Optional[Tensor],
    beta: Optional[Tensor],
    running_mean: Tensor,
    running_variance: Tensor,
    beta_variance: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    norm_weight = _compute_weight(gamma, running_variance, eps)
    normalized_mean = (mean - running_mean) * norm_weight
    if beta is not None:
        normalized_mean = normalized_mean + beta
    if covariance.dim() > mean.dim():
        normalized_covariance = covariance * torch.matmul(norm_weight.unsqueeze(-1), norm_weight.unsqueeze(-2))
        if beta_variance is not None:
            normalized_covariance = normalized_covariance + functional.variance_to_covariance(beta_variance)
    else:
        normalized_covariance = covariance * norm_weight * norm_weight
        if beta_variance is not None:
            normalized_covariance = normalized_covariance + torch.nn.functional.softplus(beta_variance)
    return normalized_mean, normalized_covariance


class CustomMomentBatchNorm1d(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.9,
        bias_variance: bool = False,
        special_init: bool = True,
        affine: bool = True,
        record_mean_variance: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.special_init = special_init

        if affine:
            self.weight = Parameter(torch.ones(num_features))
            self.bias = Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_variance", torch.ones(num_features))
        if record_mean_variance:
            self.register_buffer("mean_variance", torch.ones(num_features))
        else:
            self.register_buffer("mean_variance", None)

        if bias_variance and affine:
            self.bias_variance = Parameter(torch.ones(num_features))
        else:
            self.register_parameter("bias_variance", None)

        if special_init and affine:
            with torch.no_grad():
                self.weight.fill_(2.5)
                self.bias.fill_(2.5)
                if self.bias_variance is not None:
                    self.bias_variance.fill_(2.5)

    def forward(self, *args) -> Tuple[Tensor, Tensor]:
        mean, covariance = functional.parse_moment_input(args)
        if self.training:
            mean, covariance, self.running_mean, self.running_variance, self.mean_variance = _batch_norm_train(
                mean,
                covariance,
                self.weight,
                self.bias,
                self.running_mean,
                self.running_variance,
                self.mean_variance,
                self.bias_variance,
                self.momentum,
                self.eps,
            )
        else:
            mean, covariance = _batch_norm_eval(
                mean,
                covariance,
                self.weight,
                self.bias,
                self.running_mean,
                self.running_variance,
                self.bias_variance,
                self.eps,
            )
        return mean, covariance

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, bias_variance={self.bias_variance is not None}, "
            f"special_init={self.special_init}, momentum={self.momentum}, eps={self.eps}, affine={self.affine}"
        )
