# -*- coding: utf-8 -*-
import torch
from collections import defaultdict
from typing import Union, List
import math
from .base_type import BaseProbe


class NeuronProbe(BaseProbe):
    def __init__(self,
                attr: Union[str, List], 
                dt: float = 1e-2, 
                probe_interval: int = 1, 
                indices=None, **kwargs) -> None:
        super().__init__()
        self.step_count: int = 0
        self.dt = dt
        self.probe_interval = math.ceil(probe_interval / dt)
        self.indices = indices
        self.attr = attr
        self.data = defaultdict(list)
    
    def reset(self, *args, **kwargs):
        self.step_count = 0
        self.data = defaultdict(list)
    
    def collect_data(self, neurons: torch.nn.Module, key: str, neurons_alias=None):
        if self.indices is None:
            data: torch.Tensor = getattr(neurons, key).clone().detach().cpu()
        else:
            data: torch.Tensor = getattr(neurons, key)[self.indices].clone().detach().cpu()
        if neurons_alias is None:
            self.data[key].append(data.unsqueeze(0))
        else:
            if isinstance(self.data[neurons_alias], list):
                self.data[neurons_alias] = defaultdict(list)
            self.data[neurons_alias][key].append(data.unsqueeze(0))
    
    def forward(self, neurons: torch.nn.Module, neurons_alias=None):
        self.step_count += 1
        if self.step_count % self.probe_interval == 0:
            if isinstance(self.attr, str):
                self.collect_data(neurons, self.attr, neurons_alias=neurons_alias)
            else:
                for key in self.attr:
                    self.collect_data(neurons, key, neurons_alias=neurons_alias)
    
    def dump_collected_data(self, save_path):
        torch.save(self.data, save_path)
    
    def get_data(self):
        return self.data