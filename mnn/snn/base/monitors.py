# -*- coding: utf-8 -*-
import torch
from torch import Tensor
from .base_type import BaseMonitor

class SpikeMonitor(BaseMonitor):
    def __init__(self, num_neurons, dt=1e-1, **kwargs):
        super(SpikeMonitor, self).__init__()

        self.register_buffer('monitor', torch.zeros(num_neurons).unsqueeze(0).to(torch.int).to_sparse())
        self.dt = dt
        self.num_neurons = num_neurons
        self.step_count = 0

    def reset(self):
        device = self.monitor.device
        setattr(self, 'monitor', torch.zeros(self.num_neurons).unsqueeze(0).to(torch.int).to_sparse().to(device))
        self.step_count = 0

    def forward(self, x: Tensor):
        self.step_count += 1
        is_spike = x.to(torch.int)
        setattr(self, 'monitor', torch.cat([self.monitor, is_spike.unsqueeze(0).to_sparse()], dim=0))
        return x

    def spike_count(self, device='cpu'):
        try:
            count = torch.sparse.sum(self.monitor, dim=0).to(device).to_dense()
        except RuntimeError:
            count = torch.zeros(self.num_neurons, device=device)
        return count, self.step_count * self.dt

    def dump_monitor_state(self, save_path):
        torch.save(self.state_dict(), save_path)
    
    def get_data(self):
        return self.monitor.clone().detach().cpu()