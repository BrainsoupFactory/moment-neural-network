# -*- coding: utf-8 -*-
import torch
from torch.nn.parameter import Parameter
from torch import Tensor
from typing import Optional, Tuple
from . import functional


def _compute_weight(gamma: Optional[Tensor], var, eps=1e-5):
    if gamma is None:
        return 1 / torch.sqrt(var + eps)
    else:
        return gamma / torch.sqrt(var + eps)


@torch.no_grad()
def _track_running_state(mean, var, running_mean, running_var, momentum=0.9):
    running_mean = momentum * running_mean + (1 - momentum) * mean
    running_var = momentum * running_var + (1 - momentum) * var
    return running_mean, running_var


@torch.no_grad()
def _track_mean_variance(var, mean_variance, momentum=0.9):
    mean_variance = momentum * mean_variance + (1 - momentum) * var
    return mean_variance


def _batch_norm_for_train(u: Tensor, cov: Tensor, gamma: Tensor, beta: Optional[Tensor],
                          running_mean: Tensor, running_var: Tensor, mean_variance: Optional[Tensor]=None, beta_var=None,
                          momentum: float = 0.9, eps: float = 1e-5):

    u_var, u_mean = torch.var_mean(u, dim=0, keepdim=True)
    if cov.dim() > u.dim():
        var = torch.diagonal(cov, dim1=-1, dim2=-2)
    else:
        var = cov
    var = torch.mean(var, dim=0, keepdim=True)
    if mean_variance is not None:
        mean_variance = _track_mean_variance(var.reshape(mean_variance.shape), mean_variance, momentum)
    u_var = u_var + var

    running_mean, running_var = _track_running_state(u_mean.reshape(running_mean.shape), u_var.reshape(running_var.shape), running_mean, running_var, momentum)

    norm_weight = _compute_weight(gamma, u_var, eps)
    u_norm = (u - u_mean) * norm_weight
    if beta is not None:
        u_norm = u_norm + beta
    if cov.dim() > u.dim():
        cov_norm = cov * torch.matmul(norm_weight.unsqueeze(-1), norm_weight.unsqueeze(-2))
        if beta_var is not None:
            cov_norm = cov_norm + functional.var2cov(beta_var)
    else:
        cov_norm = cov * norm_weight * norm_weight
        if beta_var is not None:
            cov_norm = cov_norm + functional.F.softplus(beta_var)
    return u_norm, cov_norm, running_mean, running_var, mean_variance


def _batch_norm_for_test(u, cov, gamma, beta, running_mean, running_var, beta_var=None, eps=1e-5):
    norm_weight = _compute_weight(gamma, running_var, eps)
    u_norm = (u - running_mean) * norm_weight
    if beta is not None:
        u_norm = u_norm + beta
    if cov.dim() > u.dim():
        cov_norm = cov * torch.matmul(norm_weight.unsqueeze(-1), norm_weight.unsqueeze(-2))
        if beta_var is not None:
            cov_norm = cov_norm + functional.var2cov(beta_var)
    else:
        cov_norm = cov * norm_weight * norm_weight
        if beta_var is not None:
            cov_norm = cov_norm + functional.F.softplus(beta_var)
    return u_norm, cov_norm


class CustomBatchNorm1D(torch.nn.Module):
    def __init__(self, num_features, eps: float = 1e-5, momentum: float = 0.9,
                 bias_var: bool = False, special_init: bool = True, affine: bool = True, record_mean_var: bool = False):
        super(CustomBatchNorm1D, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        if affine:
            self.weight = Parameter(torch.ones(num_features))
            self.bias = Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        if record_mean_var:
            self.register_buffer('mean_variance', torch.ones(num_features))
        else:
            self.register_buffer('mean_variance', None)

        if bias_var and affine:
            self.bias_var = Parameter(torch.ones(num_features))
        else:
            self.register_parameter('bias_var', None)

        self.affine = affine
        self.special_init = special_init
        if special_init and affine:
            self.weight.data.fill_(2.5)
            self.bias.data.fill_(2.5)
            if self.bias_var is not None:
                self.bias_var.data.fill_(2.5)

    def forward(self, *args) -> Tuple[Tensor, Tensor]:
        u, cov = functional.parse_input(args)
        if self.training:
            u, cov, self.running_mean, self.running_var, self.mean_variance = _batch_norm_for_train(u, cov, self.weight, self.bias,
            self.running_mean, self.running_var, self.mean_variance,
            self.bias_var, self.momentum, self.eps,)
        else:
            u, cov = _batch_norm_for_test(u, cov, self.weight, self.bias, self.running_mean, self.running_var, self.bias_var, self.eps)
        return u, cov

    def extra_repr(self) -> str:
        return 'num_features: {}, bias_std={}, special_init={}, momentum={}, eps={}, affine={}'.format(
            self.num_features, self.bias_var is not None, self.special_init, self.momentum, self.eps, self.affine
        )



