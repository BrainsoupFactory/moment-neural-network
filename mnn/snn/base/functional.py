# -*- coding: utf-8 -*-
import torch
from torch import Tensor
import numpy as np
from typing import Optional


def sample_size(num_neurons, num_steps=None):
    if num_steps is None:
        num = [1, num_neurons]
    else:
        if isinstance(num_neurons, int):
            if isinstance(num_steps, int):
                num = (num_steps, num_neurons)
            else:
                num = list(num_steps) + [num_neurons]
        else:
            if isinstance(num_steps, int):
                num = [num_steps] + list(num_neurons)
            else:
                num = list(num_steps) + list(num_neurons)
    return num

def pregenerate_gaussian_current(num_neurons, num_steps: int, mean: Tensor, std: Tensor, rho: Optional[Tensor] = None, generator=None):
    if rho is None:
        num = sample_size(num_neurons, num_steps)
        pregenerated = torch.randn(num, device=mean.device) * std + mean
    else:
        cov = torch.matmul(std.unsqueeze(-1), std.unsqueeze(-2)) * rho
        if generator is None:
            generator = np.random.default_rng()
        if isinstance(num_neurons, int):
            size = num_steps
        else:
            size = [num_steps] + list(num_neurons)[:-1]
        pregenerated = generator.multivariate_normal(mean.cpu().numpy(), cov.cpu().numpy(), size=size)
        pregenerated = torch.from_numpy(pregenerated).to(dtype=mean.dtype, device=mean.device)
    return pregenerated