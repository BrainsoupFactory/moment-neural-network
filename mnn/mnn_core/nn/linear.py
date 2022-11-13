# -*- coding: utf-8 -*-
import torch
import math
from typing import Tuple, Optional
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from . import functional


class LinearDuo(torch.nn.Module):
    __constant__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False, bias_var: bool = False, scale=None, dropout: Optional[float] = None, sparse_degree=None) -> None:
        super(LinearDuo, self).__init__()
        if dropout is None:
            self.dropout = dropout
        else:
            self.dropout = torch.nn.Dropout(dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.scale = scale

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))

        else:
            self.register_parameter('bias', None)

        if bias_var:
            self.bias_var = Parameter(torch.ones(out_features))
        else:
            self.register_parameter('bias_var', None)
        self.reset_parameters()
        
        self.sparse_degree = sparse_degree
        self.generate_sparse_mask()
    
    def generate_sparse_mask(self):
        if self.sparse_degree is not None:
            assert 0 < self.sparse_degree <= self.in_features
            in_deg = int(self.sparse_degree)
            sparse_mask = []
            for i in range(self.out_features):
                temp = torch.from_numpy(np.random.choice(self.in_features, in_deg, False))
                temp = torch.cat([torch.zeros_like(temp).unsqueeze(0), temp.unsqueeze(0)]).to(torch.int)
                temp = torch.sparse_coo_tensor(temp, torch.ones(in_deg), (1, self.in_features))
                sparse_mask.append(temp.to_dense())
            sparse_mask = torch.cat(sparse_mask)
            assert sparse_mask.size() == self.weight.size()
            self.register_buffer('sparse_mask', sparse_mask)
        else:
            self.register_buffer('sparse_mask', None)

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        if self.bias_var is not None:
            self.bias_var.data.fill_(1.)

    def forward(self, *args) -> Tuple[Tensor, Tensor]:
        u, cov = functional.parse_input(args)
        if self.sparse_mask is not None:
            weight = self.weight * self.sparse_mask
        else:
            weight = self.weight
        u, cov = functional.mnn_linear(u, cov, weight, self.bias, self.bias_var, self.dropout, self.scale)
        return u, cov

    def extra_repr(self) -> str:
        return 'in_features: {}, out_features: {}, bias_mean: {}, bias_var: {}, dropout: {}, scale: {}'.format(
            self.in_features, self.out_features, self.bias is not None, self.bias_var is not None,
                                                 self.dropout is not None, self.scale)


class LinearNoRho(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias_mean: bool = False, bias_std: bool = False) -> None:
        super(LinearNoRho, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))

        if bias_mean:
            self.bias_mean = Parameter(torch.Tensor(out_features))

        else:
            self.register_parameter('bias_mean', None)

        if bias_std:
            self.bias_std = Parameter(torch.Tensor(out_features))

        else:
            self.register_parameter('bias_std', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias_mean is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias_mean, -bound, bound)

            if self.bias_std is not None:
                init.uniform_(self.bias_std, 0, bound)

    def forward(self, *args) -> Tuple[Tensor, Tensor]:
        u, s = functional.parse_input(args)
        out_u = F.linear(u, self.weight, self.bias_mean)
        if self.bias_std is None:
            out_s = F.linear(torch.pow(s, 2), torch.pow(self.weight, 2), self.bias_std)
        else:
            out_s = F.linear(torch.pow(s, 2), torch.pow(self.weight, 2), torch.pow(self.bias_std, 2))

        out_s = torch.sqrt(out_s)
        return out_u, out_s

    def extra_repr(self) -> str:
        return 'in_features: {}, out_features: {}, bias_mean: {}, bias_std: {}'.format(
            self.in_features, self.out_features, self.bias_mean is not None, self.bias_std is not None)


class Identity(torch.nn.Module):
    def forward(self, *args):
        return args
