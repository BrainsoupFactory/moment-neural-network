# -*- coding: utf-8 -*-
import yaml
import torch
import os
import warnings
from torch import Tensor, distributed as dist, multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader


class RecordMethods:
    @staticmethod
    def make_dir(dir_path: str):
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def save_state(file_path, net_state, trained_epochs, acc, **kwargs):
        state = {
            'net': net_state,
            'epochs': trained_epochs,
            'acc': acc
        }
        for key in kwargs.keys():
            state[key] = kwargs[key]

        torch.save(state, file_path)

    @staticmethod
    def writing_log(log_path: str, info: str, encoding='utf-8', mode='a+'):
        with open(log_path, encoding=encoding, mode=mode) as f:
            f.write(info)

    @staticmethod
    def load_state_dict(net: torch.nn.Module, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        return net

    @staticmethod
    def record_hyper_parameter(path: str, name: str, **kwargs):
        with open(path + '{:}_config.yaml'.format(name), 'w') as f:
            yaml.dump(kwargs, f, default_flow_style=False)

    @staticmethod
    def rename_duplicate_file(file_path, file_name, suffix_pos=None):
        temp = file_name
        i = 1
        while os.path.exists(file_path + temp):
            if suffix_pos is None:
                name, suffix = file_name.split('.')
            else:
                name = file_name[:suffix_pos-1]
                suffix = file_name[suffix_pos:]
            name += '({})'.format(i)
            i += 1
            temp = name + '.{}'.format(suffix)
        return temp


class InputPreprocess:
    def __init__(self, device='cpu', dtype=None, mask_mean=False, mask_cov=False):
        self.device = device
        self.dtype = dtype
        self.mask_mean = mask_mean
        self.mask_cov = mask_cov

    def __call__(self, inputs):
        return self.mnn_inputs_preprocess(inputs)

    def mnn_inputs_preprocess(self, inputs):
        if not isinstance(inputs, Tensor):
            temp = list()
            if not isinstance(inputs[0], Tensor):
                for item in inputs:
                    temp.append(self._tuple_to(item))
                inputs = tuple(temp)
            else:
                inputs = self._tuple_to(inputs)
        else:
            inputs = self.to_device_and_dtype(inputs, self.device, self.dtype)
        return inputs

    def _tuple_to(self, data):
        u, cov = data
        u = self.to_device_and_dtype(u, self.device, self.dtype, self.mask_mean)
        cov = self.to_device_and_dtype(cov, self.device, self.dtype, self.mask_cov)
        return u, cov

    @staticmethod
    def to_device_and_dtype(data: Tensor, device='cpu', dtype=None, mask=False):
        if dtype == 'float':
            data = data.to(torch.float32)
        elif dtype == 'double':
            data = data.to(torch.float64)
        data = data.to(device)
        if mask:
            return torch.zeros_like(data)
        else:
            return data


class PredictMethods:
    """
    Parameter free predictor collections
    """
    @staticmethod
    def max_mean_predictor(outputs):
        if not isinstance(outputs, Tensor):
            _, predicted = outputs[0].max(1)
        else:
            _, predicted = outputs.max(1)

        return predicted

    @staticmethod
    def min_risk_predictor(u, s, gamma):
        risk = - u + gamma * s / 2
        _, predicted = torch.min(risk, dim=-1)
        return predicted


class BinaryPredictor:
    def __init__(self, threshold: float = 0.5):
        assert 0 < threshold < 1
        self.threshold = threshold

    def __call__(self, output: Tensor):
        return torch.gt(output, self.threshold).to(torch.long).view(-1)


class ScoreMethods:
    @staticmethod
    def equal_protocol(predicted, targets):
        return predicted.eq(targets.view(predicted.size())).sum().item()

    @staticmethod
    def regression_protocol(predicted, targets, threshold: float, gamma: int = 1):
        return torch.sum(torch.abs(predicted - targets) < threshold) / gamma


def check_nan(x, raise_err=True):
    if torch.sum(torch.isnan(x)) > 0:
        print('Input Has NaN!')
        if raise_err:
            raise ValueError
    else:
        print('Input has no NaN!')


def batch_numpy2tensor(device, *args):
    temp = []
    for i in args:
        temp.append(torch.from_numpy(i).to(device).to(torch.float64))
    return temp


def batch_cat_tensor(dim=0, *args):
    temp = []
    for i in args:
        temp.append(torch.cat(i, dim=dim))
    return temp


class DistributedOps:
    @staticmethod
    def setup(rank, world_size, backend='nccl'):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # initialize the process group
        dist.init_process_group(backend, rank=rank, world_size=world_size)

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    @staticmethod
    def checkpoint(model, save_path, save_name, rank):
        if rank == 0:
            torch.save(model.state_dict(), save_path + save_name + '.pt')
        dist.barrier()

    @staticmethod
    def prepare_dataloader(dataset, rank, world_size, batch_size=256, pin_memory=False, num_workers=0, shuffle=True,
                           drop_last=False):
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last)
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers,
                                drop_last=drop_last, shuffle=False, sampler=sampler)
        return dataloader

    @staticmethod
    def wrap_model(model, rank):
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        return model

    @staticmethod
    def ddp_runs(func, world_size):
        mp.spawn(func, args=(world_size,), nprocs=world_size)

    @staticmethod
    def reduce_mean(tensor, nprocs):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt = rt / nprocs
        return rt

def remove_file(path):
    if os.path.exists(path):
        os.remove(path)
    else:
        warnings.warn('File not exist! {}'.format(path))
    

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")