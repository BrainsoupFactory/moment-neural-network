# -*- coding: utf-8 -*-
import torch
import numpy as np
from .. import mnn_core
from torch import Tensor
from .. import utils
from . import base, mnn2snn

def modify_value_by_condition(data, condition):
    u, cov = data
    if 'mask_mean' in condition:
        u = torch.zeros_like(u)
    elif 'shuffle_cov' in condition:
        cov = cov * torch.eye(cov.size(-1), device=cov.device)
    elif 'corr_only' in condition:
        u = torch.zeros_like(u)
        _, rho = mnn_core.nn.functional.compute_correlation(cov)
        #std = torch.ones_like(std)
        cov = rho
    elif 'mean_only' in condition:
        cov = torch.zeros_like(cov)
    return u, cov

class MnnSnnValidate:
    def __init__(self, args, running_time=20, dt=1e-2, num_trials=100, monitor_size=None, 
                pregenerate=False, resume_best=False, train=False, init_vol=None, alias='', input_type='gaussian',**kwargs) -> None:
        
        args = self.resume_config(args=args, resume_best=resume_best)
        self.args = args
        self.running_time = running_time
        self.train = train

        self.pregenerate = pregenerate
        self.dt = dt
        self.num_steps = int(running_time / dt)
        self.num_trials = num_trials
        self.monitor_size = monitor_size
             
        self.init_vol = init_vol
        self.alias = alias
        self.prefix = 'run{}_dt{}'.format(running_time, dt)
        self.input_type = input_type

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.prepare_dump_dir()
        self.generate_dataset()
        self.generate_model()
        self.extra_works(**kwargs)

    def extra_works(self, *args, **kwargs):
        pass

    def resume_config(self, args, resume_best):
        save_path = getattr(args, 'dump_path', './checkpoint/') + args.dir + '/'
        args.resume_best = resume_best
        args.save_path = save_path
        args.config = save_path + args.save_name + '_config.yaml'
        args = utils.training_tools.set_config2args(args)
        return args
    
    def reset(self):
        self.snn.reset()
        self.snn.eval()
        self.mnn.eval()
    
    def custom_reset(self):
        self.snn.reset_spike_count_list()
        self.snn.reset_generator()
        self.snn.reset_monitor()
        self.snn.reset_probe()
    
    def __len__(self):
        return len(self.dataset)
    
    def prepare_dump_dir(self):
        save_path = getattr(self.args, 'dump_path', './checkpoint/') + self.args.dir + '/{}_snn_validate_result/'.format(self.args.save_name)
        setattr(self, 'spike_dump_path', save_path)
        utils.training_tools.RecordMethods.make_dir(save_path)

    def generate_dataset(self):
        train_set, test_set = utils.training_tools.make_image_fold_dataset(self.args)
        if self.train:
            setattr(self, 'dataset', train_set)
        else:
            setattr(self, 'dataset', test_set)
    
    def generate_model(self):
        mnn_model = self.generate_mnn_model()
        snn_model = self.generate_snn_model()
        snn_model.load_state_dict(mnn_model.state_dict())
        snn_model.mnn2snn(dt=self.dt, batch_size=self.num_trials, monitor_size=self.monitor_size, 
        pregenerate=self.pregenerate, num_steps=self.num_steps, init_vol=self.init_vol, **getattr(self.args, 'NEURONS', {}))
        setattr(self, 'mnn', mnn_model)
        setattr(self, 'snn', snn_model)
        self.mnn.cuda(self.args.local_rank)
        self.snn.cuda(self.args.local_rank)
        self.reset()
        
    def generate_mnn_model(self):
        model = utils.training_tools.model_generator(self.args.save_path, 
        self.args.save_name, to_cuda=True, 
        resume_model=True, resume_best=self.args.resume_best, 
        local_rank=self.args.local_rank)
        return model
    
    def generate_snn_model(self):
        model_type = self.args.MODEL['meta']['mlp_type']
        model_args = self.args.MODEL[model_type]
        if model_type == 'snn_mlp':
            transformed = mnn2snn.SnnMlpTrans(**model_args)
        else:
            transformed = mnn2snn.MnnMlpTrans(**model_args)
        return transformed

    def data2cuda(self, data):
        data = data.cuda(self.args.local_rank)
        return data
    
    def save_result(self, idx, overwrite=True, probe_alias=None, reset_probe=True,**result):
        data_source = 'train' if self.train else 'test'
        result['input_type'] = self.input_type
        result['idx'] = idx
        result['data_source'] = data_source
        result['spike_count'] = self.snn.spike_count
        result['record_duration'] = self.snn.record_duration
        result['probe_data'] = self.collect_probe_data(probe_alias=probe_alias, reset_probe=reset_probe)
        save_name = self.alias + self.prefix + '_{}_{}_idx_{}.snnval'.format(self.input_type, data_source, idx)
        if not overwrite:
            save_name = utils.training_tools.RecordMethods.rename_duplicate_file(self.spike_dump_path, save_name)
        torch.save(result, self.spike_dump_path + save_name)
    
    def collect_probe_data(self, probe_alias=None, reset_probe=True):
        data = self.snn.collect_probe_data(probe_alias=probe_alias, reset_probe=reset_probe)
        return data

    def dump_spike_train(self, idx, overwrite=True, monitor_alias=None):
        data_source = 'train' if self.train else 'test'
        save_name = self.alias + self.prefix + '_{}_{}_idx_{}.spt'.format(self.input_type, data_source, idx)
        result = {}
        if isinstance(monitor_alias, str):
            monitor = getattr(self.snn, monitor_alias)
            assert isinstance(monitor, base.SpikeMonitor)
            result[monitor_alias] = monitor.get_data()
        elif isinstance(monitor_alias, list):
            for name in monitor_alias:
                monitor = getattr(self.snn, name)
                assert isinstance(monitor, base.SpikeMonitor)
                result[name] = monitor.get_data()
        else:
            for name in self.snn.monitor_alias:
                monitor = getattr(self.snn, name)
                assert isinstance(monitor, base.SpikeMonitor)
                result[name] = monitor.get_data()

        if not overwrite:
            save_name = utils.training_tools.RecordMethods.rename_duplicate_file(self.spike_dump_path, save_name)
        torch.save(result, self.spike_dump_path + save_name)
    
    def predict_policy(self, data):
        if isinstance(data, Tensor):
            mean = data
        else:
            mean, _ = data
        pred = torch.max(mean.reshape(1, -1), dim=-1)[-1]
        return pred
    
    @torch.inference_mode()
    def mnn_validate_one_sample(self, idx):
        (mean, cov), target = self.dataset[idx]
        mean = self.data2cuda(mean.unsqueeze(0))
        cov = self.data2cuda(cov.unsqueeze(0))
        mean, cov = self.mnn((mean, cov))
        pred = self.predict_policy((mean, cov))
        return mean, cov, pred, target
    
    def prepare_inputs(self, idx):
        (mean, cov), _ = self.dataset[idx]
        condition = getattr(self.args, 'cov_condition', 'full')
        mean = self.data2cuda(mean)
        cov = self.data2cuda(cov)
        mean, cov = modify_value_by_condition((mean, cov), condition)
        input_neuron = mean.size()[-1]
        if self.input_type == 'gaussian':
            std, rho = mnn_core.nn.functional.compute_correlation(cov)
            if 'shuffle_cov' in condition:
                rho = None
            input_current = base.GaussianCurrentGenerator(num_neurons=(self.num_trials, input_neuron), 
            mean=mean,std=std,rho=rho, dt=self.dt, pregenerate=self.pregenerate,num_steps=self.num_steps).cuda(self.args.local_rank)
        else:
            input_current = base.PoissonSpikeGenerator(
                num_neurons=(self.num_trials, input_neuron), 
                freqs=mean, 
                dt=self.dt,
                pregenerate=self.pregenerate, 
                num_steps=self.num_steps).cuda(self.args.local_rank)
        return input_current
    
    @torch.inference_mode()
    def run_one_simulation(self, idx, record=True, dump_spike_train=False, overwrite=True,**kwargs):
        inputs = self.prepare_inputs(idx)
        for _ in range(self.num_steps):
            x = inputs()
            if getattr(self.args, 'background_noise', None) is not None:
                x = x + torch.randn_like(x) * self.args.background_noise * np.sqrt(self.dt)
            _ = self.snn(x)
        if record:
            if dump_spike_train:
                self.dump_spike_train(idx, overwrite=overwrite)
            self.snn.spike_statistic()  
    
    @torch.inference_mode()
    def validate_one_sample(self, idx, do_reset=False, print_log=False, **kwargs):
        if do_reset:
            self.reset()
        else:
            self.custom_reset()
        mnn_mean, mnn_cov, mnn_pred, target = self.mnn_validate_one_sample(idx)
        self.run_one_simulation(idx, **kwargs)
        snn_mean, snn_cov = self.snn.make_predict()
        pred = self.predict_policy((snn_mean, snn_cov))
        if print_log:
            print('{}, Img idx: {}, target: {}, pred: {}'.format(self.dataset, idx, target, pred))
        self.save_result(idx=idx,mnn_output=(mnn_mean, mnn_cov, mnn_pred), target=target,
                         snn_output=(snn_mean, snn_cov, pred), running_time=self.running_time, dt=self.dt, **kwargs) 

def sample_poisson_spike(freqs, dt, num_neuron, num_steps, device='cpu', dtype=torch.float):
    num = base.sample_size(num_neuron, num_steps)
    return (torch.rand(num, device=device) < freqs * dt).to(dtype)


def sparse_spike_train_statistics(spike_train: Tensor, time_window: float, start_time_step: int = None):
    """
    Note that state_time_step is the time step w.r.t dt
    """
    if start_time_step is not None:
        spike_train = torch.index_select(spike_train, dim=0, index=torch.arange(start_time_step, spike_train.size(0)))
    spike_count = torch.sparse.sum(spike_train, dim=0).to_dense().to(torch.float).T
    mean = torch.mean(spike_count, dim=-1) / time_window
    cov = torch.cov(spike_count) / time_window
    return mean, cov

