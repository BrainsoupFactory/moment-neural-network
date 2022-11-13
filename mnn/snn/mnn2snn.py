# -*- coding: utf-8 -*-
import types
import torch
from torch import Tensor
from torch.nn import functional as F
from collections import defaultdict
from .. import mnn_core
from .. import models
from . import base
from .base import SpikeMonitor, GeneralCurrentGenerator, LIFNeurons


@torch.no_grad()
def ln_params_transform(self, dt: float = 1e-1, batch_size=None, **kwargs):
    setattr(self, 'dt', dt)
    mean = self.bias
    if self.bias_var is None:
        std = None
    else:
        std = torch.sqrt(F.softplus(self.bias_var))
        if mean is None:
            mean = torch.zeros_like(std)

    num = base.sample_size(self.out_features, batch_size)
    generator = GeneralCurrentGenerator(num, mean, std, dt=dt, **kwargs)
    setattr(self, 'generator', generator)

def transform_ln_module(ln, dt=1e-1, batch_size=None, **kwargs):
    ln_params_transform(ln, dt, batch_size, **kwargs)
    ln.forward = types.MethodType(ln_forward, ln)
    ln.reset = types.MethodType(reset_generator, ln)
    return ln

def ln_forward(self, x: Tensor) -> Tensor:
    bias = self.generator()
    x = F.linear(x, self.weight, bias)
    return x

@torch.no_grad()
def custom_bn_params_transfer(self, dt: float = 1e-1, batch_size=None, **kwargs):
    if self.weight is None:
        weight = 1 / torch.sqrt((self.running_var + self.eps))
    else:
        weight = self.weight / torch.sqrt((self.running_var + self.eps))
    mean = - self.running_mean * weight
    if self.bias is not None:
        mean += self.bias
    if self.bias_var is None:
        std = None
    else:
        std = torch.sqrt(F.softplus(self.bias_var))
    num = base.sample_size(self.num_features, batch_size)
    generator = GeneralCurrentGenerator(num, mean, std, dt=dt, **kwargs)

    setattr(self, 'generator', generator)
    setattr(self, 'dt', dt)
    setattr(self, 'weight', torch.nn.Parameter(weight))

def custom_bn_forward(self, x: Tensor) -> Tensor:
    bias = self.generator()
    if bias is None:
        x = self.weight * x
    else:
        x = self.weight * x + bias
    return x

def transfer_custom_bn_module(bn, dt=1e-1, batch_size=None, **kwargs):
    custom_bn_params_transfer(bn, dt=dt, batch_size=batch_size, **kwargs)
    bn.forward = types.MethodType(custom_bn_forward, bn)
    bn.reset = types.MethodType(reset_generator, bn)
    return bn

def spike_statistic(self, monitor_alias=None, device='cpu', reset_monitor=True):
    if isinstance(monitor_alias, str): # collect from one specific monitor only
        monitor = getattr(self, monitor_alias)
        assert isinstance(monitor, SpikeMonitor)
        spike_count, time_window = monitor.spike_count(device)
        self.spike_count[monitor_alias].append(spike_count.unsqueeze(0))
        self.record_duration[monitor_alias].append(time_window)
        if reset_monitor:
            monitor.reset()

    elif isinstance(monitor_alias, list):
        for name in monitor_alias: # collect from one specific group of monitors
            monitor = getattr(self, name)
            assert isinstance(monitor, SpikeMonitor)
            spike_count, time_window = monitor.spike_count(device)
            self.spike_count[name].append(spike_count.unsqueeze(0))
            self.record_duration[name].append(time_window)
            if reset_monitor:
                monitor.reset()
    else:
        for name in self.monitor_alias: # collect from all monitors
            monitor = getattr(self, name)
            assert isinstance(monitor, SpikeMonitor)
            spike_count, time_window = monitor.spike_count(device)
            self.spike_count[name].append(spike_count.unsqueeze(0))
            self.record_duration[name].append(time_window)
            if reset_monitor:
                monitor.reset()

def collect_probe_data(self, probe_alias=None, reset_probe=False):
    def fetch_data(key):
        probe = getattr(self, key, None)
        if probe is None:
            data = None
        else:
            assert isinstance(probe, base.BaseProbe)
            data = probe.get_data()
            if reset_probe:
                probe.reset()
        return data

    if isinstance(probe_alias, str):
        data = fetch_data(probe_alias)

    elif isinstance(probe_alias, list):
        data = dict()
        for name in probe_alias:
            data[name] = fetch_data(name)
    else:
        if getattr(self, 'probe_alias', None) is None:
            data = None
        else:
            data = dict()
            for name in self.probe_alias:
                data[name] = fetch_data(name)
    return data


def reset_spike_count_list(self, monitor_alias=None):
    if monitor_alias is None:
        setattr(self, 'spike_count', defaultdict(list))
        setattr(self, 'record_duration', defaultdict(list))
    else:
        self.spike_count[monitor_alias] = []
        self.record_duration[monitor_alias] = []

def mean_rate(self, monitor_alias=None, dim=1, sep=False):
    """
    Default shape of spike count is (1, #trials, #neurons), thus need to average dim(0, 1)
    The output is the tensor of the shape (#recorde, #neuron) for seperate and (1, $neuron) for non sep
    dim=1 by default will average the spike count of different trials
    If dim is None, no average operation will take.
    """
    if monitor_alias is None:
        monitor_alias = self.monitor_alias[-1]
    if sep:
        spike_count = torch.cat(self.spike_count[monitor_alias])
        duration = torch.tensor(self.record_duration[monitor_alias]).reshape(-1, 1, 1)
        mean = spike_count / duration
        if dim is not None:
            mean = torch.mean(mean, dim=dim)
    else:
        # spike count of all time window
        spike_count = torch.sum(torch.cat(self.spike_count[monitor_alias]), dim=0, keepdim=True)
        duration = torch.sum(torch.tensor(self.record_duration[monitor_alias]))
        mean = spike_count / duration
        if dim is not None:
            mean = torch.mean(mean, dim=dim)
    return mean

def moment_statistic(self, sep=False, monitor_alias=None, dtype=torch.float):
    """
    Function to compute trial average mean fire rate and trial-trial variability, hence the dim of trial must vanished.
    """
    if monitor_alias is None:
        monitor_alias = self.monitor_alias[-1]
    if sep:
        spike_count = torch.cat(self.spike_count[monitor_alias])
        duration = torch.tensor(self.record_duration[monitor_alias]).reshape(-1, 1, 1)
        spike_count = torch.permute(spike_count, (0, 2, 1)).to(dtype=dtype) # (#record, #neuron, #trials)
        mean, cov = mnn_core.nn.functional.mean_cov_pooling(spike_count)
        mean = mean / duration
        cov = cov / duration
    else:
        spike_count = torch.sum(torch.cat(self.spike_count[monitor_alias]), dim=0, keepdim=True)
        duration = torch.sum(torch.tensor(self.record_duration[monitor_alias]))
        spike_count = torch.permute(spike_count, (0, 2, 1)).to(dtype=dtype) # (1, #neuron, #trials)
        mean, cov = mnn_core.nn.functional.mean_cov_pooling(spike_count)
        mean = mean / duration
        cov = cov / duration
    return mean, cov

def reset(self, debug=False):
    self.reset_spike_count_list()
    self.reset_generator(debug=debug)
    self.reset_monitor(debug=debug)
    self.reset_neuron(debug=debug)
    self.reset_probe(debug=debug)

def reset_generator(self, debug=False):
    for idx, module in enumerate(self.modules()):
        if isinstance(module, base.BaseCurrentGenerator):
            module.reset()
            if debug:
                print('idx: {}, type:{}, executed reset!'.format(idx, type(module)))

def reset_monitor(self, debug=False):
    for idx, module in enumerate(self.modules()):
        if isinstance(module, base.BaseMonitor):
            module.reset()
            if debug:
                print('idx: {}, type:{}, executed reset!'.format(idx, type(module)))

def reset_neuron(self, debug=False):
    for idx, module in enumerate(self.modules()):
        if isinstance(module, base.BaseNeuronType):
            module.reset()
            if debug:
                print('idx: {}, type:{}, executed reset!'.format(idx, type(module)))

def reset_probe(self, debug=False):
    for idx, module in enumerate(self.modules()):
        if isinstance(module, base.BaseProbe):
            module.reset()
            if debug:
                print('idx: {}, type:{}, executed reset!'.format(idx, type(module)))

def set_funcs2model(model):
    model.spike_statistic = types.MethodType(spike_statistic, model)
    model.collect_probe_data = types.MethodType(collect_probe_data, model)
    model.reset_spike_count_list = types.MethodType(reset_spike_count_list, model)
    model.mean_rate = types.MethodType(mean_rate, model)
    model.reset = types.MethodType(reset, model)
    model.moment_statistic = types.MethodType(moment_statistic, model)
    model.reset_generator = types.MethodType(reset_generator, model)
    model.reset_monitor = types.MethodType(reset_monitor, model)
    model.reset_neuron = types.MethodType(reset_neuron, model)
    model.reset_probe = types.MethodType(reset_probe, model)


class MnnMlpTrans(models.MnnMlp):
    def mnn2snn(self, dt=1e-1, batch_size=None, monitor_size=None, **kwargs):
        self.convert_modules(dt=dt, batch_size=batch_size, **kwargs)
        self.add_monitors(dt=dt, batch_size=batch_size, monitor_size=monitor_size, **kwargs)
        setattr(self, 'dt', dt)
        set_funcs2model(self)
        self.add_probes(dt=1e-1, batch_size=None, monitor_size=None, **kwargs)
        self.extra_converting(**kwargs)
        self.reset()
    
    def convert_modules(self,  dt=1e-1, batch_size=None, neuron_type: base.BaseNeuronType = LIFNeurons, **kwargs):
        mlp = torch.nn.ModuleList()
        for i in range(len(self.structure) - 1):
            out_features = self.structure[i + 1]
            module = self.mlp[3 * i]
            module = transform_ln_module(module, dt, batch_size, **kwargs)
            mlp.append(module)

            module = self.mlp[3 * i + 1]
            module = transfer_custom_bn_module(module, dt, batch_size, **kwargs)
            mlp.append(module)
            num = base.sample_size(out_features, batch_size)
            module = neuron_type(num, dt=dt, **kwargs)
            mlp.append(module)

        self.mlp = torch.nn.Sequential(*mlp)
        #self.predict = transform_ln_module(self.predict)
    
    def add_monitors(self, dt=1e-1, batch_size=None, monitor_size=None, **kwargs):
        if monitor_size is None:
            num = base.sample_size(self.structure[-1], batch_size)
        else:
            num = base.sample_size(monitor_size, batch_size)
        monitor = SpikeMonitor(num, dt=dt, **kwargs)
        setattr(self, 'monitor', monitor)
        setattr(self, 'monitor_alias', ['monitor'])
    
    def add_probes(self, *args, **kwargs):
        pass

    def forward(self, x):
        x = self.mlp(x)
        x = self.monitor(x)
        return x

    @torch.inference_mode()
    def make_predict(self, *args, **kwargs):
        mean, cov = self.moment_statistic(*args, **kwargs)
        device = self.predict.weight.device
        mean = mean.to(device)
        cov = cov.to(device)
        return self.predict(mean, cov)
    
    def extra_converting(self, *args, **kwargs):
        pass


def ensemble_layer_transform(module, dt=1e-1, batch_size=None, neuron_type: base.BaseNeuronType = LIFNeurons, **kwargs):
    out_features = module.out_features
    ln = module.linear
    ln = transform_ln_module(ln, dt, batch_size, **kwargs)

    bn = module.bn
    bn = transfer_custom_bn_module(bn, dt, batch_size, **kwargs)
    num = base.sample_size(out_features, batch_size)
    activate = neuron_type(num, dt=dt, **kwargs)
    return ln, bn, activate


class SnnMlpTrans(models.SnnMlp):
    def mnn2snn(self, dt=1e-1, batch_size=None, monitor_size=None, neuron_type: base.BaseNeuronType = LIFNeurons, **kwargs):
        self.convert_modules(dt=dt, batch_size=batch_size, neuron_type=neuron_type, **kwargs)
        self.add_monitors(dt=dt, batch_size=batch_size, monitor_size=monitor_size, **kwargs)
        setattr(self, 'dt', dt)
        set_funcs2model(self)
        # User can rewrite functions via extra_converting, it does nothing by default
        self.add_probes(dt=1e-1, batch_size=None, monitor_size=None, **kwargs)
        self.extra_converting(**kwargs)
        self.reset()
            
    def forward(self, x):
        x = self.mlp(x)
        x = self.predict(x)
        x = self.monitor(x)
        return x
    
    def add_probes(self, *args, **kwargs):
        pass
    
    def convert_modules(self,  dt=1e-1, batch_size=None, neuron_type: base.BaseNeuronType = LIFNeurons, **kwargs):
        mlp = torch.nn.ModuleList()
        for module in self.mlp:
            modules = ensemble_layer_transform(module, dt, batch_size, neuron_type=neuron_type, **kwargs)
            mlp.extend(modules)
        predict = ensemble_layer_transform(self.predict, dt, batch_size, neuron_type=neuron_type, **kwargs)
        self.mlp = torch.nn.Sequential(*mlp)
        self.predict = torch.nn.Sequential(*predict)
    
    def add_monitors(self, dt=1e-1, batch_size=None, monitor_size=None, **kwargs):
        if monitor_size is None:
            num = base.sample_size(self.predict[0].out_features, batch_size)
        else:
            num = base.sample_size(monitor_size, batch_size)
        monitor = SpikeMonitor(num, dt=dt, **kwargs)
        setattr(self, 'monitor', monitor)
        setattr(self, 'monitor_alias', ['monitor'])

    def extra_converting(self, *args, **kwargs):
        pass
    
    @torch.inference_mode()
    def make_predict(self,*args, **kwargs):
        mean, cov = self.moment_statistic(*args, **kwargs)
        return mean, cov
