# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch import Tensor
from typing import Optional


def sample_size(num_neurons, num_steps=None):
    if num_steps is None:
        num = [1, num_neurons]
    else:
        if isinstance(num_neurons, int):
            num = (num_steps, num_neurons)
        else:
            num = [num_steps] + list(num_neurons)
    return num

class BaseCurrentGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def reset(self, *args, **kwargs):
        raise NotImplementedError

class PregenerateCurrent(BaseCurrentGenerator):
    def __init__(self, pregenerated_current: Tensor, dt=1e-2) -> None:
        super().__init__()
        self.register_buffer('current', pregenerated_current, persistent=False)
        self.step_count = 0
        self.dt = dt
    
    def forward(self):
        current = self.current[self.step_count]
        self.step_count += 1
        return current
    
    def reset(self):
        self.step_count = 0

class ConstantCurrent(BaseCurrentGenerator):
    def __init__(self, current: Tensor, dt=1e-2) -> None:
        super().__init__()
        self.register_buffer('current', current * dt, persistent=False)
        self.step_count = 0
        self.dt = dt
    
    def forward(self):
        self.step_count += 1
        return self.current
    
    def reset(self):
        self.step_count = 0

class GeneralCurrentGenerator(BaseCurrentGenerator):
    def __init__(self, num_neurons: Optional[int] = None, mean: Optional[Tensor] = None,
                 std: Optional[Tensor] = None, rho=None, dt: float = 1e-1, pregenerate: bool = False, num_steps=None, **kwargs):
        super(GeneralCurrentGenerator, self).__init__()
        if mean is not None and std is not None:
            self.generator = GaussianCurrentGenerator(num_neurons, mean, std, rho, dt, pregenerate, num_steps)

        elif mean is not None and std is None:
            self.register_buffer('generator', mean.clone().detach() * dt)

        else:
            self.generator = None

    def forward(self):
        if self.generator is None:
            return None
        elif isinstance(self.generator, GaussianCurrentGenerator):
            return self.generator()
        else:
            return self.generator

    def reset(self):
        if isinstance(self.generator, GaussianCurrentGenerator):
            self.generator.reset()


class PoissonSpikeGenerator(BaseCurrentGenerator):
    """
    Homogeneous Poisson spike train generator.
    The frequency will first product with dt for computing saving in later process
    """
    def __init__(self, num_neurons, freqs: Tensor, dt: float = 1e-1, pregenerate=False, num_steps=None, **kwargs) -> None:
        super(PoissonSpikeGenerator, self).__init__()
        self.num_neurons = num_neurons
        self.register_buffer('freqs', freqs * dt)
        self.dt = dt
        self.num_steps = num_steps
        self.is_pregenerate = pregenerate and self.num_steps is not None
        self.step_count = 0
        self.pregenerated = None 
        self.reset()
    
    def forward(self):
        if self.pregenerated is None:
            x = torch.lt(torch.rand(self.num_neurons, device=self.freqs.device), self.freqs).to(self.freqs.dtype)
        else:
            x = self.pregenerated[self.step_count]
        self.step_count += 1
        return x
    
    def reset(self):
        self.step_count = 0
        if self.is_pregenerate and self.num_steps is not None:
            num = sample_size(self.num_neurons, self.num_steps)
            temp = torch.lt(torch.rand(num, device=self.freqs.device), self.freqs).to(self.freqs.dtype)
            setattr(self, 'pregenerated', temp) 
            self.is_pregenerate = True
        else:
            self.pregenerated = None  
            self.is_pregenerate = False
            
class InhomogeneousPoissonSpikeGenerator(PoissonSpikeGenerator):
    def forward(self):
        if self.pregenerated is None:
            x = torch.lt(torch.rand(self.num_neurons, device=self.freqs.device), self.freqs[self.step_count]).to(self.freqs.dtype)
        else:
            x = self.pregenerated[self.step_count]
        self.step_count += 1
        return x

class ConstantSpikeGenerator(BaseCurrentGenerator):
    def __init__(self, current: Tensor, dt: float = 1e-2, threshold: float = 1., dtype=torch.float) -> None:
        super().__init__()
        self.register_buffer('rate', torch.abs(current * dt))
        self.register_buffer('sign', torch.sign(current))
        self.register_buffer('voltage', torch.zeros_like(current))
        self.step_count = 0
        self.dt = dt
        self.threshold = threshold
        self.dtype = dtype
    
    def forward(self):
        self.voltage += self.rate
        is_spike = self.voltage >= self.threshold
        self.voltage[is_spike] = 0.
        is_spike = is_spike * self.sign
        self.step_count += 1
        return is_spike.to(self.dtype)
    
    def reset(self):
        self.step_count = 0
        self.voltage.zero_()
    


class IndependentGaussianCurrent(BaseCurrentGenerator):
    def __init__(self, num_neurons, mean, std, **kwargs):
        super(IndependentGaussianCurrent, self).__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.num_neurons = num_neurons

    def sample(self, batch_size=torch.Size()):
        return torch.randn(self.num_neurons, device=self.mean.device) * self.std + self.mean
    
    def reset(self):
        pass

class MultiVariateNormalCurrent(BaseCurrentGenerator):
    def __init__(self, num_neurons, mean: Tensor, cov: Tensor, **kwargs) -> None:
        super(MultiVariateNormalCurrent, self).__init__()
        self.num_neurons = num_neurons
        if isinstance(num_neurons, int):
            self.size = None
        else:
            self.size = tuple(num_neurons)[:-1]
        self.register_buffer('flag', torch.zeros(1).to(mean.dtype).to(mean.device))
        self.cov = cov.cpu().numpy()
        self.loc = mean.cpu().numpy()
        self.rng = np.random.default_rng()
    
    def sample(self, batch_size=torch.Size()):
        return torch.from_numpy(self.rng.multivariate_normal(self.loc, self.cov, size=self.size)).to(self.flag.dtype).to(self.flag.device)
    
    def reset(self):
        pass

def _pregenerate_gaussian_current(num_neurons, num_steps: int, mean: Tensor, std: Tensor, rho: Optional[Tensor] = None):
    if rho is None:
        num = sample_size(num_neurons, num_steps)
        pregenerated = torch.randn(num, device=mean.device) * std + mean
    else:
        cov = torch.matmul(std.unsqueeze(-1), std.unsqueeze(-2)) * rho
        generator = np.random.default_rng()
        if isinstance(num_neurons, int):
            size = num_steps
        else:
            size = [num_steps] + list(num_neurons)[:-1]
        pregenerated = generator.multivariate_normal(mean.cpu().numpy(), cov.cpu().numpy(), size=size)
        pregenerated = torch.from_numpy(pregenerated).to(mean.dtype).to(mean.device)
    return pregenerated


class GaussianCurrentGenerator(BaseCurrentGenerator):
    def __init__(self, num_neurons, mean: Tensor, std: Tensor, rho: Optional[Tensor] = None,
                 dt: float = 1e-1, pregenerate=False, num_steps=None, **kwargs):
        super(GaussianCurrentGenerator, self).__init__()
        self.num_neurons = num_neurons
        self.num_steps = num_steps
        self.is_pregenerate = pregenerate and num_steps is not None
        self.dt = dt
        self.register_buffer('mean', mean * dt)
        self.register_buffer('std', std * np.sqrt(dt))
        self.register_buffer('rho', rho)
        self.generator = None
        self.step_count = 0
        self.reset()

    def forward(self, batch_size=torch.Size()):
        if self.is_pregenerate:
            x = self.pregenerated[self.step_count]
        else:
            x = self.generator.sample(batch_size)
        self.step_count += 1
        return x

    def reset(self):
        if self.is_pregenerate:
            temp = _pregenerate_gaussian_current(self.num_neurons, self.num_steps, self.mean, self.std, self.rho)
            setattr(self, 'pregenerated', temp)
        else:
            if self.rho is not None:
                cov = torch.matmul(self.std.unsqueeze(-1), self.std.unsqueeze(-2)) * self.rho
                self.generator = MultiVariateNormalCurrent(self.num_neurons, self.mean, cov)
            else:
                self.generator = IndependentGaussianCurrent(self.num_neurons, self.mean, self.std)
        self.step_count = 0



