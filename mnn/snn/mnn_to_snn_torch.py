# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import defaultdict
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from .. import models
from ..mnn_core.nn.activation_torch import ConstantCurrentActivationTorch
from ..mnn_core.nn.batch_norm_torch import MomentBatchNorm1d
from ..mnn_core.nn.custom_batch_norm_torch import CustomMomentBatchNorm1d
from ..mnn_core.nn.ensemble_torch import MomentBlock
from ..mnn_core.nn.linear_torch import MomentLinear
from ..mnn_core.nn import functional_torch as moment_functional
from .base.currents_torch import BaseCurrentSource, GeneralCurrentSource
from .base.functional_torch import sample_shape
from .base.monitors_torch import SpikeMonitorTorch
from .base.neurons_torch import LifNeurons
from .base.probes_torch import NeuronProbeTorch


class LinearCurrentLayer(torch.nn.Module):
    def __init__(self, weight: Tensor, bias: Optional[Tensor], generator: GeneralCurrentSource) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight.detach().clone())
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = torch.nn.Parameter(bias.detach().clone())
        self.generator = generator

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.generator())


class AffineCurrentLayer(torch.nn.Module):
    def __init__(self, weight: Tensor, generator: GeneralCurrentSource) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight.detach().clone())
        self.generator = generator

    def forward(self, x: Tensor) -> Tensor:
        bias = self.generator()
        if bias is None:
            return self.weight * x
        return self.weight * x + bias


def _linear_current_layer(linear: torch.nn.Module, dt: float, batch_size=None, **kwargs) -> LinearCurrentLayer:
    mean = getattr(linear, "bias", None)
    variance = getattr(linear, "bias_variance", None)
    std = None if variance is None else torch.sqrt(F.softplus(variance))
    if mean is None and std is not None:
        mean = torch.zeros_like(std)
    generator = GeneralCurrentSource(sample_shape(linear.out_features, batch_size), mean, std, dt=dt, **kwargs)
    return LinearCurrentLayer(linear.weight, mean, generator)


def _norm_current_layer(norm: torch.nn.Module, dt: float, batch_size=None, **kwargs) -> AffineCurrentLayer:
    if isinstance(norm, MomentBatchNorm1d):
        batch_norm = norm.batch_norm_mean
        scale = batch_norm.weight / torch.sqrt(batch_norm.running_var + batch_norm.eps)
        mean = -batch_norm.running_mean * scale + batch_norm.bias
        variance = norm.bias_variance
        features = norm.num_features
    elif isinstance(norm, CustomMomentBatchNorm1d):
        scale = norm.weight / torch.sqrt(norm.running_variance + norm.eps) if norm.weight is not None else 1 / torch.sqrt(norm.running_variance + norm.eps)
        mean = -norm.running_mean * scale
        if norm.bias is not None:
            mean = mean + norm.bias
        variance = norm.bias_variance
        features = norm.num_features
    else:
        raise TypeError(f"Unsupported norm type: {type(norm)!r}")
    std = None if variance is None else torch.sqrt(F.softplus(variance))
    generator = GeneralCurrentSource(sample_shape(features, batch_size), mean, std, dt=dt, **kwargs)
    return AffineCurrentLayer(scale, generator)


def _rate_norm_current_layer(norm: torch.nn.BatchNorm1d, dt: float, batch_size=None, **kwargs) -> AffineCurrentLayer:
    scale = norm.weight / torch.sqrt(norm.running_var + norm.eps)
    mean = -norm.running_mean * scale + norm.bias
    generator = GeneralCurrentSource(sample_shape(norm.num_features, batch_size), mean, None, dt=dt, **kwargs)
    return AffineCurrentLayer(scale, generator)


def _neuron(num_neurons, dt: float, neuron_type=torch.nn.Module, **kwargs):
    return neuron_type(num_neurons, dt=dt, **kwargs)


class MomentToSnnMixin:
    def spike_statistic(self, monitor_alias=None, device="cpu", reset_monitor: bool = True) -> None:
        aliases = [monitor_alias] if isinstance(monitor_alias, str) else monitor_alias
        if aliases is None:
            aliases = self.monitor_alias
        for name in aliases:
            monitor = getattr(self, name)
            count, duration = monitor.spike_count(device)
            self.spike_count[name].append(count.unsqueeze(0))
            self.record_duration[name].append(duration)
            if reset_monitor:
                monitor.reset()

    def collect_probe_data(self, probe_alias=None, reset_probe: bool = False):
        aliases = [probe_alias] if isinstance(probe_alias, str) else probe_alias
        if aliases is None:
            aliases = getattr(self, "probe_alias", None)
        if aliases is None:
            return None
        result = {}
        for name in aliases:
            probe = getattr(self, name, None)
            result[name] = None if probe is None else probe.get_data()
            if probe is not None and reset_probe:
                probe.reset()
        return result[aliases[0]] if isinstance(probe_alias, str) else result

    def reset_spike_count_list(self, monitor_alias=None) -> None:
        if monitor_alias is None:
            self.spike_count = defaultdict(list)
            self.record_duration = defaultdict(list)
        else:
            self.spike_count[monitor_alias] = []
            self.record_duration[monitor_alias] = []

    def _reset_modules(self, module_type, debug: bool = False) -> None:
        for index, module in enumerate(self.modules()):
            if isinstance(module, module_type):
                module.reset()
                if debug:
                    print(f"idx: {index}, type:{type(module)}, executed reset!")

    def reset_generator(self, debug: bool = False) -> None:
        self._reset_modules(BaseCurrentSource, debug)

    def reset_monitor(self, debug: bool = False) -> None:
        self._reset_modules(SpikeMonitorTorch, debug)

    def reset_neuron(self, debug: bool = False) -> None:
        self._reset_modules(LifNeurons, debug)

    def reset_probe(self, debug: bool = False) -> None:
        self._reset_modules(NeuronProbeTorch, debug)

    def reset(self, debug: bool = False) -> None:
        self.reset_spike_count_list()
        self.reset_generator(debug)
        self.reset_monitor(debug)
        self.reset_neuron(debug)
        self.reset_probe(debug)

    def moment_statistic(self, sep: bool = False, monitor_alias=None, dtype=torch.float):
        if monitor_alias is None:
            monitor_alias = self.monitor_alias[-1]
        if sep:
            spike_count = torch.cat(self.spike_count[monitor_alias])
            duration = torch.tensor(self.record_duration[monitor_alias], device=spike_count.device).reshape(-1, 1, 1)
            spike_count = torch.permute(spike_count, (0, 2, 1)).to(dtype=dtype)
            mean, covariance = moment_functional.mean_cov_pooling(spike_count)
            return mean / duration, covariance / duration
        spike_count = torch.sum(torch.cat(self.spike_count[monitor_alias]), dim=0, keepdim=True)
        duration = torch.sum(torch.tensor(self.record_duration[monitor_alias], device=spike_count.device))
        spike_count = torch.permute(spike_count, (0, 2, 1)).to(dtype=dtype)
        mean, covariance = moment_functional.mean_cov_pooling(spike_count)
        return mean / duration, covariance / duration

    def mean_rate(self, monitor_alias=None, dim: Optional[int] = 1, sep: bool = False):
        if monitor_alias is None:
            monitor_alias = self.monitor_alias[-1]
        if sep:
            spike_count = torch.cat(self.spike_count[monitor_alias])
            duration = torch.tensor(self.record_duration[monitor_alias], device=spike_count.device).reshape(-1, 1, 1)
        else:
            spike_count = torch.sum(torch.cat(self.spike_count[monitor_alias]), dim=0, keepdim=True)
            duration = torch.sum(torch.tensor(self.record_duration[monitor_alias], device=spike_count.device))
        mean = spike_count / duration
        return torch.mean(mean, dim=dim) if dim is not None else mean

    def add_monitors(self, dt: float = 1e-1, batch_size=None, monitor_size=None, dtype=torch.int, **kwargs) -> None:
        size = sample_shape(self.structure[-1] if monitor_size is None else monitor_size, batch_size)
        self.monitor = SpikeMonitorTorch(size, dt=dt, dtype=dtype, **kwargs)
        self.monitor_alias = ["monitor"]

    def add_probes(self, *args, **kwargs) -> None:
        pass

    def convert_modules(self, dt: float = 1e-1, batch_size=None, neuron_type=LifNeurons, **kwargs) -> None:
        raise NotImplementedError

    def extra_converting(self, *args, **kwargs) -> None:
        pass

    def mnn_to_snn(self, dt: float = 1e-1, batch_size=None, monitor_size=None, neuron_type=LifNeurons, **kwargs) -> None:
        self.convert_modules(dt=dt, batch_size=batch_size, neuron_type=neuron_type, **kwargs)
        self.add_monitors(dt=dt, batch_size=batch_size, monitor_size=monitor_size, **kwargs)
        self.dt = dt
        self.add_probes(dt=dt, batch_size=batch_size, monitor_size=monitor_size, **kwargs)
        self.extra_converting(**kwargs)
        self.reset()

    mnn2snn = mnn_to_snn


class MomentMlpToSnn(models.MomentMlp, MomentToSnnMixin):
    def convert_modules(self, dt: float = 1e-1, batch_size=None, neuron_type=LifNeurons, **kwargs) -> None:
        converted = []
        for index in range(len(self.structure) - 1):
            linear = self.layers[3 * index]
            norm = self.layers[3 * index + 1]
            out_features = self.structure[index + 1]
            converted.extend(
                [
                    _linear_current_layer(linear, dt, batch_size, **kwargs),
                    _norm_current_layer(norm, dt, batch_size, **kwargs),
                    neuron_type(sample_shape(out_features, batch_size), dt=dt, **kwargs),
                ]
            )
        self.layers = torch.nn.Sequential(*converted)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return self.monitor(x)

    @torch.inference_mode()
    def make_predict(self, *args, **kwargs):
        mean, covariance = self.moment_statistic(*args, **kwargs)
        if self.predict is None:
            return mean, covariance
        return self.predict(mean.to(next(self.predict.parameters()).device), covariance.to(next(self.predict.parameters()).device))


class SpikeMomentMlpToSnn(models.SpikeMomentMlp, MomentToSnnMixin):
    def convert_modules(self, dt: float = 1e-1, batch_size=None, neuron_type=LifNeurons, **kwargs) -> None:
        converted = []
        for block in self.layers:
            converted.extend(self._convert_block(block, dt, batch_size, neuron_type, **kwargs))
        self.layers = torch.nn.Sequential(*converted)
        if isinstance(self.predict, MomentBlock):
            self.predict = torch.nn.Sequential(*self._convert_block(self.predict, dt, batch_size, neuron_type, **kwargs))

    @staticmethod
    def _convert_block(block: MomentBlock, dt: float, batch_size, neuron_type, **kwargs):
        return [
            _linear_current_layer(block.linear, dt, batch_size, **kwargs),
            _norm_current_layer(block.norm, dt, batch_size, **kwargs),
            neuron_type(sample_shape(block.out_features, batch_size), dt=dt, **kwargs),
        ]

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        if self.predict is not None:
            x = self.predict(x)
        return self.monitor(x)

    def add_monitors(self, dt: float = 1e-1, batch_size=None, monitor_size=None, dtype=torch.int, **kwargs) -> None:
        if monitor_size is None:
            monitor_size = self.num_classes if self.num_classes is not None else self.structure[-1]
        size = sample_shape(monitor_size, batch_size)
        self.monitor = SpikeMonitorTorch(size, dt=dt, dtype=dtype, **kwargs)
        self.monitor_alias = ["monitor"]

    @torch.inference_mode()
    def make_predict(self, *args, **kwargs):
        return self.moment_statistic(*args, **kwargs)


class RateMomentMlpToSnn(models.MomentRateMlp, MomentToSnnMixin):
    def convert_modules(self, dt: float = 1e-1, batch_size=None, neuron_type=LifNeurons, **kwargs) -> None:
        converted = []
        layer_index = 0
        for index in range(len(self.structure) - 1):
            linear = self.layers[layer_index]
            norm = self.layers[layer_index + 1]
            activation = self.layers[layer_index + 2]
            out_features = self.structure[index + 1]
            linear_bias = None if linear.bias is None else linear.bias * dt
            generator = GeneralCurrentSource(sample_shape(out_features, batch_size), linear_bias, None, dt=1.0, **kwargs)
            converted.extend(
                [
                    LinearCurrentLayer(linear.weight, linear_bias, generator),
                    _rate_norm_current_layer(norm, dt, batch_size, **kwargs),
                    neuron_type(
                        sample_shape(out_features, batch_size),
                        dt=dt,
                        threshold_voltage=activation.threshold_voltage,
                        membrane_conductance=activation.membrane_conductance,
                        refractory_time=activation.refractory_time,
                        **kwargs,
                    ),
                ]
            )
            layer_index += 3
        self.layers = torch.nn.Sequential(*converted)

    def add_monitors(self, dt: float = 1e-1, batch_size=None, monitor_size=None, dtype=torch.float, **kwargs) -> None:
        size = sample_shape(self.structure[-1] if monitor_size is None else monitor_size, batch_size)
        self.monitor = SpikeMonitorTorch(size, dt=dt, dtype=dtype, **kwargs)
        self.monitor_alias = ["monitor"]

    def forward(self, x: Tensor) -> Tensor:
        return self.monitor(self.layers(x))

    @torch.inference_mode()
    def make_predict(self, *args, **kwargs):
        mean = self.mean_rate(*args, **kwargs)
        return self.predict(mean.to(self.predict.weight.device))


def convert_moment_parameters(model: Union[models.MomentMlp, models.SpikeMomentMlp], debug: bool = False) -> dict[str, Optional[Tensor]]:
    params: dict[str, Optional[Tensor]] = {}
    if isinstance(model, models.SpikeMomentMlp):
        blocks = list(model.layers)
        if isinstance(model.predict, MomentBlock):
            blocks.append(model.predict)
        for index, block in enumerate(blocks):
            weight, bias = moment_functional.weight_fusion(block.linear, block.norm)
            params[f"fc{index}.weight"] = weight
            params[f"fc{index}.bias"] = bias
    elif isinstance(model, models.MomentMlp):
        layer_index = 0
        param_index = 0
        while layer_index + 1 < len(model.layers):
            weight, bias = moment_functional.weight_fusion(model.layers[layer_index], model.layers[layer_index + 1])
            params[f"fc{param_index}.weight"] = weight
            params[f"fc{param_index}.bias"] = bias
            layer_index += 3
            param_index += 1
        if isinstance(model.predict, MomentLinear):
            params[f"fc{param_index}.weight"] = model.predict.weight
            params[f"fc{param_index}.bias"] = model.predict.bias
    else:
        raise TypeError(f"Unsupported model type: {type(model)!r}")

    if debug:
        for key, value in params.items():
            if value is None:
                print(f"{key} is None")
            else:
                print(f"{key}: min={torch.min(value)}, max={torch.max(value)}")
    return params
