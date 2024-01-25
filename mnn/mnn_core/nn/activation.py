# -*- coding: utf-8 -*-
import torch
from torch import Tensor
from typing import Tuple
from . import functional
from ..mnn_pytorch import mnn_activate_trio, mnn_activate_no_rho, constant_current_activate_func

class OriginMnnActivation(torch.nn.Module):

    def forward(self, *args) -> Tuple[Tensor, Tensor]:
        u, cov = functional.parse_input(args)
        if u.size(-1) != 1:
            s, r = functional.compute_correlation(cov)
            u, s, r = mnn_activate_trio(u, s, r)
            cov = functional.compute_covariance(s, r)
        else:
            # 1d case
            cov = torch.sqrt(cov)
            u, cov = mnn_activate_no_rho(u, cov)
            cov = torch.pow(cov, 2)
        return u, cov

class ConstantCurrentActivation(torch.nn.Module):
    def __init__(self, V_th: float = 20., L: float = 0.05, T_ref: float = 5.0) -> None:
        super().__init__()
        self.V_th = V_th
        self.L = L
        self.T_ref = T_ref
    
    def forward(self, x):
        x = constant_current_activate_func(x, self.V_th, self.L, self.T_ref)
        return x