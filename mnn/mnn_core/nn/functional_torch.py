# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def set_diagonal(tensor: Tensor, value: float) -> Tensor:
    result = tensor.clone()
    torch.diagonal(result, dim1=-1, dim2=-2).fill_(value)
    return result


def compute_covariance(std: Tensor, correlation: Tensor, *, enforce_unit_diagonal: bool = False) -> Tensor:
    if enforce_unit_diagonal:
        correlation = set_diagonal(correlation, 1.0)
    return torch.matmul(std.unsqueeze(-1), std.unsqueeze(-2)) * correlation


def ensure_spd(covariance: Tensor, variance: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
    if variance is None:
        variance = torch.diagonal(covariance, dim1=-2, dim2=-1)
    mask = (variance > 0).to(dtype=covariance.dtype)
    covariance = covariance * torch.matmul(mask.unsqueeze(-1), mask.unsqueeze(-2))
    return covariance, torch.relu(variance)


def compute_correlation(
    covariance: Tensor,
    *,
    eps: float = 1e-5,
    safe_check: bool = False,
) -> Tuple[Tensor, Tensor]:
    variance = torch.diagonal(covariance, dim1=-2, dim2=-1)
    if safe_check:
        covariance, variance = ensure_spd(covariance, variance)
    std = torch.sqrt(variance) + eps
    correlation = covariance / torch.matmul(std.unsqueeze(-1), std.unsqueeze(-2))
    return std, set_diagonal(correlation, 1.0)


def compute_signal_correlation(mean: Tensor, covariance: Tensor) -> Tensor:
    return torch.matmul(mean.unsqueeze(-1), mean.unsqueeze(-2)) + covariance


def covariance_from_factor(factor: Tensor, scale: Optional[float] = None) -> Tensor:
    covariance = torch.matmul(factor, factor.transpose(-2, -1))
    if scale is None:
        return covariance / math.sqrt(covariance.size(-1))
    return covariance * scale


def variance_to_covariance(variance: Tensor, scale: Optional[float] = None) -> Tensor:
    if variance.dim() != 1:
        raise ValueError("variance_to_covariance expects a 1D tensor")
    covariance = torch.diag(F.softplus(variance))
    if scale is not None:
        covariance = covariance / scale
    return covariance


def parse_moment_input(args: Sequence[object]) -> tuple[Tensor, Tensor]:
    if len(args) == 1 and not isinstance(args[0], Tensor):
        mean, covariance = args[0]
    else:
        mean, covariance = args
    return mean, covariance


parse_input = parse_moment_input


def moment_linear(
    mean: Tensor,
    covariance: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    bias_variance: Optional[Tensor] = None,
    dropout: Optional[torch.nn.Dropout] = None,
    scale: Optional[float] = None,
) -> tuple[Tensor, Tensor]:
    if dropout is not None:
        batch = mean.size(0)
        weight = dropout(weight.unsqueeze(0).expand(batch, -1, -1))

    output_mean = torch.matmul(weight, mean.unsqueeze(-1)).squeeze(-1)
    if bias is not None:
        output_mean = output_mean + bias

    output_covariance = torch.matmul(weight, torch.matmul(covariance, weight.transpose(-1, -2)))
    if bias_variance is not None:
        output_covariance = output_covariance + variance_to_covariance(bias_variance, scale)
    return output_mean, output_covariance


def batch_norm_1d_moments(
    batch_norm: torch.nn.BatchNorm1d,
    mean: Tensor,
    covariance: Tensor,
    weight: Tensor,
    bias_variance: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> tuple[Tensor, Tensor]:
    output_mean = batch_norm(mean)
    if batch_norm.training or not batch_norm.track_running_stats:
        mean_std = torch.std(mean, dim=0, keepdim=True) + batch_norm.eps
    else:
        mean_std = torch.sqrt(batch_norm.running_var) + batch_norm.eps
    output_covariance = (
        torch.mm(weight.unsqueeze(-1), weight.unsqueeze(-2))
        * covariance
        / torch.matmul(mean_std.unsqueeze(-1), mean_std.unsqueeze(-2))
    )
    if bias_variance is not None:
        output_covariance = output_covariance + variance_to_covariance(bias_variance, scale)
    return output_mean, output_covariance


def normalize_mean_covariance(mean: Tensor, covariance: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor]:
    norm = torch.sum(mean.pow(2), dim=-1, keepdim=True) + eps
    return mean / torch.sqrt(norm), covariance / norm.unsqueeze(-1)


def gaussian_sampling_transform(
    sample: Tensor,
    mean: Tensor,
    covariance: Tensor,
    eps: Tensor,
    decoding_time: float = 1.0,
    expand_sample: bool = False,
) -> Tensor:
    if eps.dim() < 2:
        raise ValueError("eps must have at least 2 dimensions")
    batch, _ = mean.size()
    if expand_sample:
        sample = sample.unsqueeze(0).expand(batch, -1, -1)
    factor = torch.linalg.cholesky_ex(covariance + eps)[0].unsqueeze(1)
    return torch.matmul(factor, sample.unsqueeze(-1)).squeeze(-1) / math.sqrt(decoding_time) + mean.unsqueeze(1)


@torch.no_grad()
def gaussian_sampling_predict(
    sample_points: Tensor,
    mean: Tensor,
    covariance: Tensor,
    eps: Tensor,
    decoding_time: float = 1.0,
    transformed_samples: Optional[Tensor] = None,
    num_classes: int = 10,
    num_samples: int = 1000,
    normalize: bool = False,
    return_prediction: bool = True,
    expand_sample: bool = False,
) -> Tensor:
    if transformed_samples is None:
        transformed_samples = gaussian_sampling_transform(
            sample_points,
            mean,
            covariance,
            eps,
            decoding_time,
            expand_sample,
        )
    prediction = torch.max(transformed_samples, dim=-1)[-1]
    prediction = F.one_hot(prediction, num_classes)
    prediction = torch.sum(prediction, dim=1)
    if normalize:
        prediction = prediction / num_samples
    if return_prediction:
        prediction = torch.max(prediction, dim=-1)[-1]
    return prediction


def upper_triangular_vector(covariance: Tensor, diagonal: int = 0) -> Tensor:
    size = covariance.size(-1)
    indices = torch.triu_indices(size, size, offset=diagonal, device=covariance.device)
    return covariance[..., indices[0], indices[1]]


triu_vec = upper_triangular_vector


def mean_covariance_pooling(
    x: Tensor,
    *,
    unbiased: bool = True,
    biological_pooling: bool = False,
) -> tuple[Tensor, Tensor]:
    sample_count = x.size(-1) - int(unbiased)
    mean = torch.mean(x, dim=-1, keepdim=True)
    centered = x - mean
    if x.dim() == 2:
        covariance = torch.einsum("mi,ni->mn", centered, centered) / sample_count
    else:
        covariance = torch.einsum("bmi,bni->bmn", centered, centered) / sample_count
    mean = mean.squeeze(-1)
    if biological_pooling:
        covariance = covariance + torch.diag_embed(mean)
    return mean, covariance


mean_cov_pooling = mean_covariance_pooling


def weight_fusion(linear: torch.nn.Module, norm: torch.nn.Module) -> tuple[Tensor, Optional[Tensor]]:
    """Fuse a moment linear layer with a following moment batch norm layer for SNN conversion."""
    if hasattr(norm, "batch_norm_mean"):
        batch_norm = norm.batch_norm_mean
        running_mean = batch_norm.running_mean
        running_variance = batch_norm.running_var
        gamma = batch_norm.weight
        beta = batch_norm.bias
        eps = batch_norm.eps
    else:
        running_mean = norm.running_mean
        running_variance = norm.running_variance
        gamma = norm.weight
        beta = norm.bias
        eps = norm.eps

    scale = 1 / torch.sqrt(running_variance + eps) if gamma is None else gamma / torch.sqrt(running_variance + eps)
    weight = linear.weight * scale.unsqueeze(-1)
    if getattr(linear, "bias", None) is None:
        bias = -running_mean * scale
    else:
        bias = (linear.bias - running_mean) * scale
    if beta is not None:
        bias = bias + beta
    return weight, bias
