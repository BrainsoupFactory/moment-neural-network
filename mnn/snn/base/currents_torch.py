# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from .functional_torch import pregenerate_gaussian_current, sample_shape


class BaseCurrentSource(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        raise NotImplementedError


class PregeneratedCurrent(BaseCurrentSource):
    def __init__(self, current: Tensor, dt: float = 1e-2) -> None:
        super().__init__()
        self.register_buffer("current", current, persistent=False)
        self.dt = dt
        self.step_count = 0

    def forward(self) -> Tensor:
        current = self.current[self.step_count]
        self.step_count += 1
        return current

    def reset(self) -> None:
        self.step_count = 0


class ConstantCurrentSource(BaseCurrentSource):
    def __init__(self, current: Tensor, dt: float = 1e-2) -> None:
        super().__init__()
        self.register_buffer("current", current * dt, persistent=False)
        self.dt = dt
        self.step_count = 0

    def forward(self) -> Tensor:
        self.step_count += 1
        return self.current

    def reset(self) -> None:
        self.step_count = 0


class IndependentGaussianCurrentSource(BaseCurrentSource):
    def __init__(self, num_neurons, mean: Tensor, std: Tensor) -> None:
        super().__init__()
        self.num_neurons = num_neurons
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def sample(self, batch_size=torch.Size()) -> Tensor:
        shape = (self.num_neurons,) if isinstance(self.num_neurons, int) else tuple(self.num_neurons)
        return torch.randn(shape, dtype=self.mean.dtype, device=self.mean.device) * self.std + self.mean

    def forward(self) -> Tensor:
        return self.sample()

    def reset(self) -> None:
        pass


class MultivariateGaussianCurrentSource(BaseCurrentSource):
    def __init__(self, num_neurons, mean: Tensor, covariance: Tensor) -> None:
        super().__init__()
        self.num_neurons = num_neurons
        self.register_buffer("mean", mean)
        self.register_buffer("covariance", covariance)
        jitter = torch.finfo(mean.dtype).eps if mean.is_floating_point() else 1e-6
        eye = torch.eye(covariance.size(-1), dtype=covariance.dtype, device=covariance.device)
        self.register_buffer("factor", torch.linalg.cholesky(covariance + eye * jitter))

    def sample(self, batch_size=torch.Size()) -> Tensor:
        if isinstance(self.num_neurons, int):
            prefix = []
        else:
            prefix = list(self.num_neurons)[:-1]
        noise = torch.randn(*prefix, self.covariance.size(-1), dtype=self.mean.dtype, device=self.mean.device)
        if self.factor.dim() == 2:
            return torch.matmul(noise, self.factor.transpose(-1, -2)) + self.mean
        return torch.einsum("...n,...mn->...m", noise, self.factor) + self.mean

    def forward(self) -> Tensor:
        return self.sample()

    def reset(self) -> None:
        pass


class GaussianCurrentSource(BaseCurrentSource):
    def __init__(
        self,
        num_neurons,
        mean: Tensor,
        std: Tensor,
        rho: Optional[Tensor] = None,
        *,
        dt: float = 1e-1,
        pregenerate: bool = False,
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_neurons = num_neurons
        self.num_steps = num_steps
        self.dt = dt
        self.is_pregenerated = pregenerate and num_steps is not None
        self.step_count = 0
        self.register_buffer("mean", mean * dt)
        self.register_buffer("std", std * math.sqrt(dt))
        self.register_buffer("rho", rho)
        self.generator: Optional[BaseCurrentSource] = None
        self.register_buffer("pregenerated", None, persistent=False)
        self.reset()

    def forward(self) -> Tensor:
        if self.is_pregenerated:
            current = self.pregenerated[self.step_count]
        elif self.generator is None:
            current = None
        else:
            current = self.generator()
        self.step_count += 1
        return current

    def reset(self) -> None:
        self.step_count = 0
        if self.is_pregenerated:
            self.pregenerated = pregenerate_gaussian_current(
                self.num_neurons,
                int(self.num_steps),
                self.mean,
                self.std,
                self.rho,
            )
            self.generator = None
        elif self.rho is None:
            self.generator = IndependentGaussianCurrentSource(self.num_neurons, self.mean, self.std)
            self.pregenerated = None
        else:
            covariance = torch.matmul(self.std.unsqueeze(-1), self.std.unsqueeze(-2)) * self.rho
            self.generator = MultivariateGaussianCurrentSource(self.num_neurons, self.mean, covariance)
            self.pregenerated = None


class GeneralCurrentSource(BaseCurrentSource):
    def __init__(
        self,
        num_neurons: Optional[int] = None,
        mean: Optional[Tensor] = None,
        std: Optional[Tensor] = None,
        rho: Optional[Tensor] = None,
        *,
        dt: float = 1e-1,
        pregenerate: bool = False,
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if mean is not None and std is not None:
            self.generator: Optional[BaseCurrentSource | Tensor] = GaussianCurrentSource(
                num_neurons,
                mean,
                std,
                rho,
                dt=dt,
                pregenerate=pregenerate,
                num_steps=num_steps,
            )
        elif mean is not None:
            self.register_buffer("generator", mean.clone().detach() * dt)
        else:
            self.generator = None

    def forward(self):
        if self.generator is None:
            return None
        if isinstance(self.generator, BaseCurrentSource):
            return self.generator()
        return self.generator

    def reset(self) -> None:
        if isinstance(self.generator, BaseCurrentSource):
            self.generator.reset()


class PoissonSpikeSource(BaseCurrentSource):
    def __init__(self, num_neurons, freqs: Tensor, *, dt: float = 1e-1, pregenerate: bool = False, num_steps=None, **kwargs) -> None:
        super().__init__()
        self.num_neurons = num_neurons
        self.dt = dt
        self.num_steps = num_steps
        self.is_pregenerated = pregenerate and num_steps is not None
        self.step_count = 0
        self.register_buffer("freqs", freqs * dt)
        self.register_buffer("pregenerated", None, persistent=False)
        self.reset()

    def forward(self) -> Tensor:
        if self.pregenerated is None:
            spikes = torch.rand(self.num_neurons, dtype=self.freqs.dtype, device=self.freqs.device) < self.freqs
        else:
            spikes = self.pregenerated[self.step_count]
        self.step_count += 1
        return spikes.to(self.freqs.dtype)

    def reset(self) -> None:
        self.step_count = 0
        if self.is_pregenerated and self.num_steps is not None:
            shape = sample_shape(self.num_neurons, self.num_steps)
            self.pregenerated = (torch.rand(shape, dtype=self.freqs.dtype, device=self.freqs.device) < self.freqs).to(self.freqs.dtype)
        else:
            self.pregenerated = None


class InhomogeneousPoissonSpikeSource(PoissonSpikeSource):
    def forward(self) -> Tensor:
        if self.pregenerated is None:
            spikes = torch.rand(self.num_neurons, dtype=self.freqs.dtype, device=self.freqs.device) < self.freqs[self.step_count]
        else:
            spikes = self.pregenerated[self.step_count]
        self.step_count += 1
        return spikes.to(self.freqs.dtype)


class ConstantSpikeSource(BaseCurrentSource):
    def __init__(self, current: Tensor, *, dt: float = 1e-2, threshold: float = 1.0, dtype=torch.float) -> None:
        super().__init__()
        self.register_buffer("rate", torch.abs(current * dt))
        self.register_buffer("sign", torch.sign(current))
        self.register_buffer("voltage", torch.zeros_like(current))
        self.dt = dt
        self.threshold = threshold
        self.dtype = dtype
        self.step_count = 0

    def forward(self) -> Tensor:
        self.voltage.add_(self.rate)
        spikes = self.voltage >= self.threshold
        self.voltage.masked_fill_(spikes, 0.0)
        self.step_count += 1
        return (spikes * self.sign).to(self.dtype)

    def reset(self) -> None:
        self.step_count = 0
        self.voltage.zero_()
