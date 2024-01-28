# -*- coding: utf-8 -*-
import types
from typing import Union

import torch
from torch import Tensor
from torch.nn import functional as F
from collections import defaultdict
from .. import mnn_core
from ..mnn_core.nn import functional as mnn_funcs


from .. import models
from . import base
from .base import SpikeMonitor, GeneralCurrentGenerator, LIFNeurons


class MnnBaseTrans:
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
            
    def reset_probe(self, debug=False):
        for idx, module in enumerate(self.modules()):
            if isinstance(module, base.BaseProbe):
                module.reset()
                if debug:
                    print('idx: {}, type:{}, executed reset!'.format(idx, type(module)))
                    
    def reset_neuron(self, debug=False):
        for idx, module in enumerate(self.modules()):
            if isinstance(module, base.BaseNeuronType):
                module.reset()
                if debug:
                    print('idx: {}, type:{}, executed reset!'.format(idx, type(module)))
    
    def reset_monitor(self, debug=False):
        for idx, module in enumerate(self.modules()):
            if isinstance(module, base.BaseMonitor):
                module.reset()
                if debug:
                    print('idx: {}, type:{}, executed reset!'.format(idx, type(module)))
                    
    def reset_generator(self, debug=False):
        for idx, module in enumerate(self.modules()):
            if isinstance(module, base.BaseCurrentGenerator):
                module.reset()
                if debug:
                    print('idx: {}, type:{}, executed reset!'.format(idx, type(module)))
    
    def reset(self, debug=False):
        self.reset_spike_count_list()
        self.reset_generator(debug=debug)
        self.reset_monitor(debug=debug)
        self.reset_neuron(debug=debug)
        self.reset_probe(debug=debug)
    
    
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
            mean, cov = mnn_funcs.mean_cov_pooling(spike_count)
            mean = mean / duration
            cov = cov / duration
        else:
            spike_count = torch.sum(torch.cat(self.spike_count[monitor_alias]), dim=0, keepdim=True)
            duration = torch.sum(torch.tensor(self.record_duration[monitor_alias]))
            spike_count = torch.permute(spike_count, (0, 2, 1)).to(dtype=dtype) # (1, #neuron, #trials)
            mean, cov = mnn_funcs.mean_cov_pooling(spike_count)
            mean = mean / duration
            cov = cov / duration
        return mean, cov
    
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
    
    def add_monitors(self, dt=1e-1, batch_size=None, monitor_size=None, dtype=torch.int, **kwargs):
        if monitor_size is None:
            num = base.sample_size(self.structure[-1], batch_size)
        else:
            num = base.sample_size(monitor_size, batch_size)
        monitor = SpikeMonitor(num, dt=dt, dtype=dtype,**kwargs)
        setattr(self, 'monitor', monitor)
        setattr(self, 'monitor_alias', ['monitor'])
    
    def add_probes(self, *args, **kwargs):
        pass
    
    def convert_modules(self,  dt=1e-1, batch_size=None, neuron_type: base.BaseNeuronType = LIFNeurons, **kwargs):
        raise NotImplementedError
    
    def extra_converting(self, *args, **kwargs):
        pass
    
    @torch.inference_mode()
    def make_predict(self, *args, **kwargs):
        mean, cov = self.moment_statistic(*args, **kwargs)
        device = self.predict.weight.device
        mean = mean.to(device)
        cov = cov.to(device)
        return self.predict(mean, cov)
    
    def mnn2snn(self, dt=1e-1, batch_size=None, monitor_size=None,  neuron_type: base.BaseNeuronType = LIFNeurons, **kwargs):
        self.convert_modules(dt=dt, batch_size=batch_size, neuron_type=neuron_type,**kwargs)
        self.add_monitors(dt=dt, batch_size=batch_size, monitor_size=monitor_size, **kwargs)
        setattr(self, 'dt', dt)
        self.add_probes(dt=dt, batch_size=batch_size, monitor_size=monitor_size, **kwargs)
        self.extra_converting(**kwargs)
        self.reset()


@torch.no_grad()
def ln_params_transform(self, dt: float = 1e-1, batch_size=None, **kwargs):
    setattr(self, 'dt', dt)
    mean = self.bias
    if getattr(self, 'bias_var', None) is None:
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
    if getattr(self, 'bias_var', None) is None:
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
    return bn

def custom_bn2d_forward(self, x: Tensor) -> Tensor:
    bias = self.generator()
    x = torch.permute(x, dims=(0, 2, 3, 1)).contiguous()
    if bias is None:
        x = self.weight * x
    else:
        x = self.weight * x + bias
    x = torch.permute(x, dims=(0, 3, 1, 2)).contiguous()
    return x

def transfer_custom_bn2d_module(bn, dt=1e-1, **kwargs):
    custom_bn_params_transfer(bn, dt=dt, batch_size=None, **kwargs)
    bn.forward = types.MethodType(custom_bn2d_forward, bn)
    return bn

@torch.no_grad()
def conv2d_params_transform(self, dt: float = 1e-1):
    setattr(self, 'dt', dt)
    if self.bias is not None:
        self.bias =  torch.nn.Parameter(self.bias * dt)

def conv2d_forward(self, x: Tensor) -> Tensor:
    x = self._conv_forward(x, self.weight, self.bias)
    return x

def transfer_conv2d_module(conv2d, dt=1e-1):
    conv2d_params_transform(conv2d, dt)
    conv2d.forward = types.MethodType(conv2d_forward, conv2d)
    return conv2d

def avg_pool2d_forward(self, x: Tensor):
    return F.avg_pool2d(x, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad, self.divisor_override)

def transfer_avg_pool2d_module(avg_pool):
    avg_pool.forward = types.MethodType(avg_pool2d_forward, avg_pool)
    return avg_pool


class MnnMlpTrans(models.MnnMlp, MnnBaseTrans):
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
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        x = self.monitor(x)
        return x

def ensemble_layer_transform(module, dt=1e-1, batch_size=None, neuron_type: base.BaseNeuronType = LIFNeurons, **kwargs):
    out_features = module.out_features
    ln = module.linear
    ln = transform_ln_module(ln, dt, batch_size, **kwargs)

    bn = module.bn
    bn = transfer_custom_bn_module(bn, dt, batch_size, **kwargs)
    num = base.sample_size(out_features, batch_size)
    activate = neuron_type(num, dt=dt, **kwargs)
    return ln, bn, activate


class SnnMlpTrans(models.SnnMlp, MnnBaseTrans):  
    def forward(self, x):
        x = self.mlp(x)
        x = self.predict(x)
        x = self.monitor(x)
        return x
    
    def convert_modules(self,  dt=1e-1, batch_size=None, neuron_type: base.BaseNeuronType = LIFNeurons, **kwargs):
        mlp = torch.nn.ModuleList()
        for module in self.mlp:
            modules = ensemble_layer_transform(module, dt, batch_size, neuron_type=neuron_type, **kwargs)
            mlp.extend(modules)
        predict = ensemble_layer_transform(self.predict, dt, batch_size, neuron_type=neuron_type, **kwargs)
        self.mlp = torch.nn.Sequential(*mlp)
        self.predict = torch.nn.Sequential(*predict)
    
    @torch.inference_mode()
    def make_predict(self,*args, **kwargs):
        mean, cov = self.moment_statistic(*args, **kwargs)
        return mean, cov


def convert_parameters(model: Union[models.MnnMlp, models.SnnMlp], debug=False):
    """
    The function convert the parameters of a MNN MLP (with or without linear decoder) 
    to the format of linear module (namely, we fuse the parameters of summation layer and batch norm layer).
    The return is a dict containing the weight and bias of each layer, prefix is fc<idx>, where idx start with 0.  
    """
    model = model.cpu()
    snn_params = {}
    for i, m in enumerate(model.mlp):
        if isinstance(m, mnn_core.nn.LinearDuo):
            weight, bias = mnn_funcs.weight_fusion(m, model.mlp[i + 1])
            snn_params['fc{}.weight'.format(i // 3)] = weight
            snn_params['fc{}.bias'.format(i // 3) ] = bias
        elif isinstance(m, mnn_core.nn.EnsembleLinearDuo):
            weight, bias = mnn_funcs.weight_fusion(m.linear, m.bn)
            snn_params['fc{}.weight'.format(i)] = weight
            snn_params['fc{}.bias'.format(i) ] = bias
    
    if isinstance(model.predict, mnn_core.nn.EnsembleLinearDuo):
        weight, bias = mnn_funcs.weight_fusion(model.predict.linear, model.predict.bn)
        snn_params['fc{}.weight'.format(i + 1)] = weight
        snn_params['fc{}.bias'.format(i + 1) ] = bias
    else:
        weight, bias = model.predict.weight, model.predict.bias
        snn_params['fc{}.weight'.format(i // 3 +1)] = weight
        snn_params['fc{}.bias'.format(i // 3 +1)] = bias
    
    if debug:
        print('After conversion to SNN, parameters are in the following range:')
        for key, value in snn_params.items():
            if value is not None:
                print('{}: min={}, max={}'.format(key, torch.min(value), torch.max(value)))
            else:
                print('{} is None!'.format(key))
    return snn_params

class MnnMlpMeanOnlyTrans(models.MnnMlpMeanOnly, MnnBaseTrans):
 
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
            
            module = self.mlp[3 * i + 2]
            num = base.sample_size(out_features, batch_size)
            module = neuron_type(num, dt=dt, V_th=module.V_th, L=module.L, T_ref=module.T_ref, **kwargs)
            mlp.append(module)

        self.mlp = torch.nn.Sequential(*mlp)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x)
        x = self.monitor(x)
        return x
    
    @torch.inference_mode()
    def make_predict(self, *args, **kwargs):
        mean = self.mean_rate(*args, **kwargs)
        device = self.predict.weight.device
        mean = mean.to(device)
        return self.predict(mean)
    

    def convert_modules(self,  dt=1e-1, batch_size=None, neuron_type: base.BaseNeuronType = LIFNeurons, **kwargs):
        feature_extractor = torch.nn.ModuleList()
        test_data = torch.ones(self.input_shape).unsqueeze(0)
        for m in self.feature_extractor:
            if isinstance(m, torch.nn.Conv2d) or isinstance(m,torch.nn.BatchNorm2d):
                with torch.no_grad():
                    if m.bias is not None:
                        m.bias =  torch.nn.Parameter(m.bias * dt)
            if isinstance(m, mnn_core.nn.ConstantCurrentActivation):
                num_neuron = base.sample_size(test_data.size()[1:], batch_size)
                m = neuron_type(num_neuron, dt=dt,  V_th=m.V_th, L=m.L, T_ref=m.T_ref, **kwargs)
            else:
                with torch.inference_mode():
                    m.eval()
                    test_data = m(test_data)
            feature_extractor.append(m)
        self.feature_extractor = torch.nn.Sequential(*feature_extractor)
        
        if self.classifier_cfg is not None:
            mlp = torch.nn.ModuleList()
            for i in range(len(self.classifier_cfg['structure']) - 1):
                out_features = self.classifier_cfg['structure'][i + 1]
                ln, bn = self.classifier.mlp[3 * i], self.classifier.mlp[3 * i + 1]
                with torch.no_grad():
                    if ln.bias is not None:
                        ln.bias =  torch.nn.Parameter(ln.bias * dt)
                    if bn.bias is not None:
                        bn.bias =  torch.nn.Parameter(bn.bias * dt)
                num = base.sample_size(out_features, batch_size)
                module = neuron_type(num, dt=dt, **kwargs)
                mlp.extend([ln, bn, module])
            self.classifier.mlp = torch.nn.Sequential(*mlp)
    
    def add_monitors(self, dt=1e-1, batch_size=None, monitor_size=None, dtype=torch.float, **kwargs):
        if monitor_size is None:
            if isinstance(self.classifier, models.MnnMlpMeanOnly):
                num = self.classifier_cfg['structure'][-1]
            else:
                num = self.classifier.in_features
            num = base.sample_size(num, batch_size)
        else:
            num = base.sample_size(monitor_size, batch_size)
        monitor = SpikeMonitor(num, dt=dt, dtype=dtype, **kwargs)
        setattr(self, 'monitor', monitor)
        setattr(self, 'monitor_alias', ['monitor'])
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extractor(x)
        x = torch.flatten(x, start_dim=1)
        if isinstance(self.classifier, models.MnnMlpMeanOnly):
            x = self.classifier.mlp(x)
        x = self.monitor(x)
        return x
    
    @torch.inference_mode()
    def make_predict(self, *args, **kwargs):
        mean = self.mean_rate(*args, **kwargs)
        if isinstance(self.classifier, models.MnnMlpNoRho):
            device = self.classifier.predict.weight.device
            mean = mean.to(device)
            predicts = self.classifier.predict(mean)
        else:
            device = self.classifier.weight.device
            mean = mean.to(device)
            predicts = self.classifier(mean)
        return predicts