# -*- coding: utf-8 -*-
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from . import functional

class BatchNorm1dDuo(torch.nn.Module):
    __constant__ = ['num_features']
    num_features: int

    def __init__(self, num_features: int, bias_var: bool = False, special_init: bool = True, scale=None) -> None:
        super(BatchNorm1dDuo, self).__init__()
        self.num_features = num_features
        self.bn_mean = torch.nn.BatchNorm1d(num_features)
        if special_init:
            self.bn_mean.weight.data.fill_(2.5)
            self.bn_mean.bias.data.fill_(2.5)
        self.scale = scale
        self.special_init = special_init
        if bias_var:
            self.bias_var = Parameter(torch.ones(num_features))
        else:
            self.register_parameter("bias_var", None)

    def forward(self, *args) -> Tuple[Tensor, Tensor]:
        u, cov = functional.parse_input(args)
        return functional.mnn_bn1d_forward(self.bn_mean, u, cov, self.bn_mean.weight, self.bias_var, self.scale)

    def extra_repr(self) -> str:
        return 'num_features: {}, bias_std: {}, special_init: {}, scale: {}'.format(
            self.num_features, self.bias_var is not None, self.special_init, self.scale)


class BatchNorm1dNoRho(torch.nn.Module):
    __constant__ = ['num_features']
    num_features: int

    def __init__(self, num_features: int, bias_std: bool = False, special_init: bool = True) -> None:
        super(BatchNorm1dNoRho, self).__init__()
        self.num_features = num_features
        self.special_init = special_init
        self.bn_mean = torch.nn.BatchNorm1d(num_features)
        if special_init:
            self.bn_mean.weight.data.fill_(2.5)
            self.bn_mean.bias.data.fill_(2.5)

        if bias_std:
            self.bias_std = Parameter(torch.Tensor(num_features))
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
        u, s = functional.parse_input(args)
        uhat = self.bn_mean(u)
        if self.bn_mean.training:
            var = torch.pow(s, 2) * torch.pow(self.bn_mean.weight, 2) / \
                  (torch.var(u, dim=0, keepdim=True) + self.bn_mean.eps)

        else:
            if self.bn_mean.track_running_stats:
                var = torch.pow(s, 2) * torch.pow(self.bn_mean.weight, 2) / \
                      (self.bn_mean.running_var + self.bn_mean.eps)
            else:
                var = torch.pow(s, 2) * torch.pow(self.bn_mean.weight, 2) / \
                      (torch.var(u, dim=0, keepdim=True) + self.bn_mean.eps)

        if self.bias_std is not None:
            var += torch.pow(self.bias_std, 2)

        return uhat, torch.sqrt(var)

    def extra_repr(self) -> str:
        return 'num_features: {}, bias_std: {}, special_init: {}, scale: {}'.format(
            self.num_features, self.bias_std is not None, self.special_init, self.scale)

