# -*- coding: utf-8 -*-
"""
Created on Thu May  2 20:54:44 2024

@author: qiyangku
"""

import torch
import numpy as np
from projects.dtb.ma_conductance import Moment_Activation_Cond


class Cond_MNN_DTB():
    def __init__(self, device='cpu'):
        self.device = device
        # initialize activation functions
        self.exc_activation = Moment_Activation_Cond() 
        self.inh_activation = Moment_Activation_Cond()
        
        # TODO: update neuronal parameters, which are different for E/I pops 
        # self.exc_activation
        
        # TODO: inter-regional synapses have its own conductance
        # which should be treated as a different channel.
        
        # initializing DTI-constrained connectivity in-degree 
        K_EE, K_EI, K_IE, K_II, K_EE_long = calculate_degree()
        self.N = K_EE.shape[0] # number of brain regions
        
        self.K_EE = torch.tensor(K_EE, device=device).unsqueeze(1) # dimensionality: Nx1 
        self.K_EI = torch.tensor(K_EI, device=device).unsqueeze(1)
        self.K_IE = torch.tensor(K_IE, device=device).unsqueeze(1)
        self.K_II = torch.tensor(K_II, device=device).unsqueeze(1)
        self.K_EE_long = torch.tensor(K_EE_long, device=device)
        
        self.dt_mnn = 0.1 #simulation time step for MNN
        
        # calculate background current stats
        Le = 0.05 # leak conductance of exc neuron; unit in micro S
        Li = 0.1 # leak conductance of inh neuron
        self.bg_mean_to_exc = 0.36/Le #nA/uS; output is mV; 50 leak conductance in nS
        self.bg_mean_to_inh = 0.54/Li
        self.bg_std_to_exc = 1.0/Le
        self.bg_std_to_inh = 1.2/Li
        
        self.xregion_gain = 0.0 # cross regional exc connection gain
        
        
    def forward(self, ue, se, ui, si):
        '''
        ue, se: mean/std of excitatory neurons  dims: batch x N
        ui, si: mean/std of inhibitory neurons
        '''
        
        # TODO: long range excitation should be a separate channel
        
        # STEP 1: excitatory neurons
        # inputs to excitatory population
        # TODO: no need a new channel for long-range EE, just add the scaling factor here
        
        exc_input_mean = torch.mm(ue, self.K_EE) + self.xregion_gain*torch.mm( ue, self.K_EE_long.T )
        inh_input_mean = torch.mm(ui, self.K_EI)
        exc_input_std = torch.mm(se*se, self.K_EE) + self.xregion_gain*self.xregion_gain*torch.mm(se*se, self.K_EE_long.T)
        exc_input_std = exc_input_std.pow(0.5)
        inh_input_std = torch.mm(si*si, self.K_EI).pow(0.5)
        
        # convert conductance input to current
        eff_input_mean, eff_input_std, tau_eff = self.exc_activation.cond2curr(exc_input_mean, exc_input_std, inh_input_mean, inh_input_std)
        # add background current
        eff_input_mean += self.bg_mean_to_exc
        eff_input_std = eff_input_std.pow(2.0)+self.bg_std_to_exc*self.bg_std_to_exc
        eff_input_std = eff_input_std.pow(0.5)
        # calculate moment activation 
        exc_mean_out, exc_std_out = self.exc_activation.activate( eff_input_mean, eff_input_std, tau_eff)
        
        
        # STEP 2: inhibitory neurons
        # a better way is to recycle the code?
        exc_input_mean = torch.mm(ue, self.K_IE)
        inh_input_mean = torch.mm(ui, self.K_II)
        exc_input_std = torch.mm(se*se, self.K_IE).pow(0.5)
        inh_input_std = torch.mm(si*si, self.K_II).pow(0.5)
        
        eff_input_mean, eff_input_std, tau_eff = self.inh_activation.cond2curr(exc_input_mean, exc_input_std, inh_input_mean, inh_input_std)
        
        eff_input_mean += self.bg_mean_to_inh
        eff_input_std = eff_input_std.pow(2.0)+self.bg_std_to_inh*self.bg_std_to_inh
        eff_input_std = eff_input_std.pow(0.5)
        
        inh_mean_out, inh_std_out = self.inh_activation.activate(eff_input_mean, eff_input_std, tau_eff)
        
        return exc_mean_out, exc_std_out, inh_mean_out, inh_std_out
        
        
    def run(self, T):
        
        # initial condition
        ue = torch.zeros(1,self.N, device=self.device) #
        se = ue.clone()
        ui = torch.zeros(1,self.N, device=self.device)
        si = ui.clone()
        
        nsteps = int(T/self.dt_mnn)
        
        for i in range(nsteps):
            #print(i)
            exc_mean_out, exc_std_out, inh_mean_out, inh_std_out = self.forward(ue,se,ui,si)
            
            ue += self.dt_mnn*(-ue + exc_mean_out)
            se += self.dt_mnn*(-se + exc_std_out)
            
            ui += self.dt_mnn*(-ui + inh_mean_out)
            si += self.dt_mnn*(-si + inh_std_out)
            
        return ue, se, ui, si
        
if __name__=='__main__':
    device='cpu'
    T = 10
    
    mnn = Cond_MNN_DTB(device=device)
    ue, se, ui, si = mnn.run(T=T)
    print('Done!')
    
    plt.close('all')
    plt.plot(ue.flatten(),'.')
    plt.plot(ui.flatten(),'.')
    
    
        