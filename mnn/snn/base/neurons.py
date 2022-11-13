# -*- coding: utf-8 -*-
import torch
from torch import Tensor
from  .base_type import BaseNeuronType


class LIFNeurons(BaseNeuronType):
    __constants__ = ['L', 'V_th', 'V_res', 'V_spk', 'dt', 'T_ref']

    def __init__(self, num_neurons, L=1/20, V_th=20, V_res=0., V_spk=50., dt=1e-1, T_ref=5., init_vol=None,spike_dtype=torch.float,
                **kwargs):
        super(LIFNeurons, self).__init__()
        if init_vol == 'uniform':
            self.register_buffer('V', torch.rand(num_neurons) * V_th)
        else:
            self.register_buffer('V', torch.ones(num_neurons) * V_res)
        self.init_vol = init_vol
        self.register_buffer('is_active', torch.ones(num_neurons).to(torch.bool))
        self.register_buffer('ref_state', torch.zeros(num_neurons))
        self.register_buffer('L', torch.ones(1) * L)
        self.register_buffer('V_th', torch.ones(1) * V_th)
        self.register_buffer('V_res', torch.ones(1) * V_res)
        self.register_buffer('V_spk', torch.ones(1) * V_spk)
        self.register_buffer('dt', torch.ones(1) * dt)
        self.register_buffer('T_ref', torch.ones(1) * T_ref)
        self.num_neurons = num_neurons
        self.spike_dtype = spike_dtype

    def reset(self):
        if self.init_vol == 'uniform':
            self.V =  torch.rand(self.num_neurons, device=self.V.device) * self.V_th
        else:
            self.V.data.fill_(self.V_res.item())
        self.is_active.data.fill_(True)
        self.ref_state.data.fill_(0.)
    
    def _update_current(self, input_current):
        return input_current
    
    def _update_voltage(self, input_current):
        is_ref = torch.logical_not(self.is_active)
        self.V[self.is_active] += -self.V[self.is_active] * self.L * self.dt + input_current[self.is_active]
        self.ref_state[is_ref] += self.dt
    
    def _update_state(self):
        is_spike = torch.ge(self.V, self.V_th)
        is_recover = torch.ge(self.ref_state, self.T_ref)
        self.ref_state[is_recover] = 0.
        self.is_active[is_recover] = True

        self.is_active[is_spike] = False
        self.V[is_spike] = self.V_res
        return is_spike

    def forward(self, input_current: Tensor):
        input_current = self._update_current(input_current)
        self._update_voltage(input_current)
        output = self._update_state()
        return output.to(self.spike_dtype)
    
    def __str__(self) -> str:
        return super().__str__()
    
    def extra_repr(self) -> str:
        info = 'L: {}, V_th: {}, V_res: {}, dt: {}, T_ref: {}'.format(self.L.item(), self.V_th.item(), self.V_res.item(), self.dt.item(), self.T_ref.item())
        return info