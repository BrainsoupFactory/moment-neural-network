# -*- coding: utf-8 -*-
from typing import Tuple, Optional

import torch
from torch import Tensor

from .. import mnn_core

def _general_forward(inputs, feature_extractor: torch.nn.ModuleList, decoder: Optional[torch.nn.Module] = None):
    u, cov = inputs
    for module in feature_extractor:
        u, cov = module(u, cov)
    if decoder is not None:
        u, cov = decoder(u, cov)
    return u, cov

def decoding_protocol_layer(num_neurons, num_class, bias, use_mean=True, use_cov=False, signal_correlation=False):

    if use_cov and use_mean:
        if signal_correlation:
            in_dims = int(num_neurons * (num_neurons + 1) / 2)
        else:
            in_dims = int(num_neurons * (num_neurons + 1) / 2) + num_neurons

        if num_class is None:
            predict = mnn_core.nn.Identity()
        else:
            predict = torch.nn.Linear(in_dims, num_class, bias=bias)
    elif use_cov and not use_mean:
        in_dims = int(num_neurons * (num_neurons + 1) / 2)
        if num_class is None:
            predict = mnn_core.nn.Identity()
        else:
            predict = torch.nn.Linear(in_dims, num_class, bias=bias)
    elif not use_cov and not use_mean:
        raise Exception('use_cov and use_mean cannot be False at same time!')
    else:
        predict = None
    return predict

def decoding_policy(u: Tensor, cov: Tensor, use_mean=True, use_cov=False, signal_correlation=False):
    if not use_cov and use_mean:
        return u, cov
    elif use_cov and not use_mean:
        cov = mnn_core.nn.functional.triu_vec(cov).unsqueeze(-1)
        return cov
    elif not use_cov and not use_mean:
        raise Exception('use_cov and use_mean cannot be False at same time!')
    else:
        if signal_correlation:
            x = torch.matmul(u.unsqueeze(-1), u.unsqueeze(-2)) + cov
            x = mnn_core.nn.functional.triu_vec(x).squeeze(-1)
        else:
            cov = mnn_core.nn.functional.triu_vec(cov).unsqueeze(-1)
            x = torch.cat([u, cov], dim=-1)
            return x


class SnnMlp(torch.nn.Module):
    def __init__(self, structure, num_class: int = 10, use_mean=True, use_cov=False, signal_correlation=False,  sparse_degree=None, **kwargs):
        super(SnnMlp, self).__init__()
        self.mlp = torch.nn.ModuleList()

        if len(structure) >= 2:
            for i in range(len(structure) - 1):
                if sparse_degree is None:
                    in_deg = None
                elif isinstance(sparse_degree, int):
                    in_deg = sparse_degree
                else:
                    in_deg = sparse_degree[i]
                self.mlp.append(mnn_core.nn.EnsembleLinearDuo(structure[i], structure[i + 1], sparse_degree=in_deg, **kwargs))
        else:
            self.mlp.append(mnn_core.nn.Identity())
        if num_class is None:
            self.predict = mnn_core.nn.Identity()
        else:
            self.predict = mnn_core.nn.EnsembleLinearDuo(structure[-1], num_class, **kwargs)
        self.use_mean = use_mean
        self.use_cov = use_cov
        self.signal_correlation = signal_correlation
        self.num_class = num_class
        self.structure = structure

    def forward(self, inputs: Tuple):
        u, cov = _general_forward(inputs, self.mlp, self.predict)
        output = decoding_policy(u, cov, self.use_mean, self.use_cov, self.signal_correlation)
        return output


class MnnMlp(torch.nn.Module):
    def __init__(self, structure, num_class: int = 10, bn_bias_var: bool = False, predict_bias: bool = False,
                 predict_bias_var: bool = False, use_mean: bool = True, use_cov: bool = False,
                 special_init: bool = True, dropout: Optional[float] = None, momentum: float = 0.9, eps: float = 1e-5,
                 record_bn_mean_var=False, signal_correlation=False, ln_bias=False, ln_bias_var=False, sparse_degree=None):
        super(MnnMlp, self).__init__()
        self.mlp = torch.nn.ModuleList()
        for i in range(len(structure) - 1):
            if sparse_degree is None:
                in_deg = None
            elif isinstance(sparse_degree, int):
                in_deg = sparse_degree
            else:
                in_deg = sparse_degree[i]
            self.mlp.append(mnn_core.nn.LinearDuo(structure[i], structure[i + 1], dropout=dropout, bias=ln_bias, bias_var=ln_bias_var,sparse_degree=in_deg))
            self.mlp.append(
                mnn_core.nn.CustomBatchNorm1D(structure[i + 1], bias_var=bn_bias_var, special_init=special_init,
                                              momentum=momentum, eps=eps, record_mean_var=record_bn_mean_var))
            self.mlp.append(mnn_core.nn.OriginMnnActivation())

        self.use_cov = use_cov
        self.use_mean = use_mean
        self.structure = structure
        self.num_class = num_class
        self.predict_bias = predict_bias
        self.signal_correlation = signal_correlation

        predict = decoding_protocol_layer(structure[-1], num_class, predict_bias, use_mean, use_cov, signal_correlation)
        if predict is not None:
            self.predict = predict
        else:
            in_dims = structure[-1]
            if num_class is None:
                self.predict = mnn_core.nn.Identity()
            else:
                self.predict = mnn_core.nn.LinearDuo(in_dims, num_class, bias=predict_bias, bias_var=predict_bias_var)

    def forward(self, inputs: Tuple):
        u, cov = _general_forward(inputs, self.mlp)
        output = decoding_policy(u, cov, self.use_mean, self.use_cov, self.signal_correlation)
        return self.predict(output)

class AnnMlp(torch.nn.Module):
    def __init__(self, structure, num_class: int = 10, need_bn: bool = True, predict_bias: bool = True,
                 activation_func='relu'):
        super(AnnMlp, self).__init__()
        self.mlp = torch.nn.ModuleList()
        if len(structure) >= 2:
            for i in range(len(structure) - 1):
                if need_bn:
                    self.mlp.append(torch.nn.Linear(structure[i], structure[i + 1], bias=False))
                    self.mlp.append(torch.nn.BatchNorm1d(structure[i + 1]))
                else:
                    self.mlp.append(torch.nn.Linear(structure[i], structure[i + 1], bias=True))
                if activation_func == 'gelu':
                    self.mlp.append(torch.nn.GELU())
                elif activation_func == 'sigmoid':
                    self.mlp.append(torch.nn.Sigmoid())
                else:
                    self.mlp.append(torch.nn.ReLU())
        else:
            self.mlp.append(torch.nn.Identity())
        self.predict = torch.nn.Linear(structure[-1], num_class, bias=predict_bias)

    def forward(self, x):
        for module in self.mlp:
            x = module(x)
        return self.predict(x)