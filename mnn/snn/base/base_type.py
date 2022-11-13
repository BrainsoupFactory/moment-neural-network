# -*- coding: utf-8 -*-
import torch

class BaseCurrentGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def reset(self, *args, **kwargs):
        raise NotImplementedError


class BaseMonitor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def reset(self, *args, **kwargs):
        raise NotImplementedError
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_data(self, *args, **kwargs):
        raise NotImplementedError

class BaseNeuronType(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def reset(self, *args, **kwargs):
        raise NotImplementedError


class BaseProbe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def reset(self, *args, **kwargs):
        raise NotImplementedError
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError
    
    def get_data(self, *args, **kwargs):
        raise NotImplementedError