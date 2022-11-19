# -*- coding: utf-8 -*-
import torch
from torch import Tensor
from typing import Tuple
from . import functional
from ..mnn_pytorch import mnn_activate_trio, mnn_activate_no_rho

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