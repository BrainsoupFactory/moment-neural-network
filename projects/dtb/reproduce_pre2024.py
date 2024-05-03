# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:50:57 2024

@author: qiyangku
"""

import torch
import numpy as np
from projects.dtb.ma_conductance import Moment_Activation_Cond
from matplotlib import pyplot as plt

tau_E = np.linspace(0, 100, 51) # see Fig. 2 in PRE 2024

KE=400 # excitatory in-degree?
KI=100
input_rate = np.array([5e-3, 10e-3, 20e-3]) # sp/ms, same for both exc and inh inputs
we = 0.1 # 0.5
wi = 0.4 # 0.4, 1.0, 10


U = np.zeros((len(tau_E), len(input_rate)))
S = np.zeros((len(tau_E), len(input_rate)))
E = np.zeros((len(tau_E), len(input_rate))) # effective reversal potential


for i in range(len(tau_E)):

    ma = Moment_Activation_Cond()
    ma.tau_E = tau_E[i]
    
    # Note in PRE 2024, input is multiplied by tau_E and tau_I
    # TODO: but this is not intrinsic to the model, so we absorb this parameter inside the moment activation
    exc_input_mean = we*KE*input_rate
    exc_input_std = we*np.sqrt(KE*input_rate)
    inh_input_mean = wi*KI*input_rate
    inh_input_std = wi*np.sqrt(KI*input_rate)
    
    eff_input_mean, eff_input_std, tau_eff = ma.cond2curr(exc_input_mean,exc_input_std,inh_input_mean,inh_input_std)
    mean_out = ma.forward_fast_mean( eff_input_mean, eff_input_std, tau_eff)
    std_out = ma.forward_fast_std( eff_input_mean, eff_input_std, tau_eff, mean_out)
    
    U[i,:] = mean_out
    S[i,:] = std_out
    E[i,:] = eff_input_mean*tau_eff


plt.close('all')

plt.figure(figsize=(3.5,6))
plt.subplot(2,1,1)
plt.plot(tau_E, E)
plt.ylabel('Eff. rev potential (mV)')

plt.subplot(2,1,2)
plt.plot(tau_E, U)
plt.xlabel('exc time constant (ms)')
plt.ylabel('firing rate (sp/ms)')

plt.tight_layout()




    
    

        