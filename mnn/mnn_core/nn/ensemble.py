# -*- coding: utf-8 -*-
from typing import Tuple, Optional

import torch
from torch import Tensor

from .activation import OriginMnnActivation
from . import linear, batch_norm, custom_batch_norm


class EnsembleLinearDuo(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, ln_bias_mean: bool = False, bn_bias_var: bool = False,
                 dropout: Optional[float] = None, sparse_degree=None, norm_type: str = 'bn_custom', activation: str = 'raw',
                 special_init: bool = True, **kwargs) -> None:
        super(EnsembleLinearDuo, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_type = activation
        self.norm_type = norm_type
        self.dropout = dropout
        self.ln_bias_mean = ln_bias_mean
        self.bn_bias_std = ln_bias_mean
        self.special_init = special_init
        self.linear = linear.LinearDuo(in_features, out_features, bias=ln_bias_mean, dropout=dropout, sparse_degree=sparse_degree)
        if norm_type == 'bn_custom':
            self.bn = custom_batch_norm.CustomBatchNorm1D(out_features, bias_var=bn_bias_var, special_init=special_init, **kwargs)
        else:
            self.bn = batch_norm.BatchNorm1dDuo(out_features, bias_var=bn_bias_var, special_init=special_init, **kwargs)
        self.activate = OriginMnnActivation()

    def forward(self, u: Tensor, cov: Tensor) -> Tuple[Tensor, Tensor]:
        u, cov = self.linear(u, cov)
        u, cov = self.bn(u, cov)
        u, cov = self.activate(u, cov)
        return u, cov

    def extra_repr(self) -> str:
        return 'in_features: {}, out_features: {}, ln_bias_mean={}, bn_bias_std={}, dropout={}, norm_type={}, ' \
               'activation={}, special_init={}'.format(
            self.in_features, self.out_features, self.ln_bias_mean, self.bn_bias_std, self.dropout,
            self.norm_type, self.activation_type, self.special_init
        )

class EnsembleLinearNoRho(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bn_bias_std: bool = True,
                 special_init: bool = True) -> None:
        super(EnsembleLinearNoRho, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.special_init = special_init
        self.linear = linear.LinearNoRho(in_features, out_features)
        self.bn = batch_norm.BatchNorm1dNoRho(out_features, bias_std=bn_bias_std, special_init=special_init)
        self.act = OriginMnnActivation()
        
    def forward(self, u: Tensor, s: Tensor) -> Tuple[Tensor, Tensor]:
        u, s = self.linear(u, s)
        u, s = self.bn(u, s)
        u, s = self.act(u, s)
        return u, s