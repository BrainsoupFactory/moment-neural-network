# moment activation for condutance-based neurons

from mnn.mnn_core.mnn_utils import Mnn_Core_Func, Param_Container
import numpy as np
from matplotlib import pyplot as plt
import torch

class Moment_Activation_Cond(Mnn_Core_Func):
    def __init__(self):
        super().__init__()
        ''' strictly follow PRE 2024'''
        
        self.tau_L = 20 # membrane time constant
        self.tau_E = 4 # excitatory synaptic time scale (ms)
        self.tau_I = 10 # inhibitory synaptic time scale (ms)
        self.VL = -60 # leak reverseal potential
        self.VE = 0 # excitatory reversal potential
        self.VI = -80 # inhibitory reversal potential
        #self.tau_eff = None # effective time constant
        self.vol_th = -50 # firing threshold
        self.vol_rest = -60 # reset potential
        
        self.L = 1/self.tau_L
                
        self.sE = 1.0 # modifier to conductance (can be voltage dependent)
        self.sI = 1.0
        
        # TODO: horrible scalability; consider a list of g, input_mean, input_std
        # so it supports any number of channels
        

        self.t_ref = 2 # ms; NB inhibitory neuron has different value

        
    def cond2curr(self, exc_input_mean, exc_input_std, inh_input_mean, inh_input_std):
        ''' This step should be called synaptic activation (vs neuronal activation)
        map conductance-based to current-based spiking neuron
        using the effective time constant approximation'''
        
        tau_eff = self.tau_L/(1+self.sE*exc_input_mean + self.sI*inh_input_mean) # effective time constant
        V_eff = tau_eff/self.tau_L*(self.VL + self.sE*exc_input_mean*self.VE \
                               + self.sI*inh_input_mean*self.VI) #effective reversal potential
        
        # approximating multiplicative noise;
        #h_E = np.sqrt(self.tau_E)*self.tau_E/tau_L*self.gE*(self.VE-V_eff)*exc_input_std
        #h_I = np.sqrt(self.tau_I)*self.tau_I/tau_L*self.gI*(self.VI-V_eff)*inh_input_std
        
        h_E = np.sqrt(self.tau_E)/self.tau_L*self.sE*(self.VE-V_eff)*exc_input_std
        h_I = np.sqrt(self.tau_I)/self.tau_L*self.sI*(self.VI-V_eff)*inh_input_std

        
        # effective input mean/std
        eff_input_mean = V_eff/tau_eff
        tmp = tau_eff*tau_eff/(tau_eff+self.tau_E)*h_E*h_E
        tmp = tmp + tau_eff*tau_eff/(tau_eff+self.tau_I)*h_I*h_I
        eff_input_std = np.power(tmp,0.5)

        return eff_input_mean, eff_input_std, tau_eff

    def forward_fast_mean(self, ubar, sbar, tau_eff):
        '''Calculates the mean output firing rate given the mean & std of input firing rate'''
        
        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.vol_th /tau_eff - ubar) < (self.cut_off / np.sqrt(tau_eff) * sbar)
        indx2 = indx0 & indx1

        mean_out = np.zeros(ubar.shape)

        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally
        ub = (self.vol_th /tau_eff[indx2] - ubar[indx2]) / (sbar[indx2] / np.sqrt(tau_eff[indx2]))
        lb = (self.vol_rest /tau_eff[indx2] - ubar[indx2]) / (sbar[indx2] / np.sqrt(tau_eff[indx2]))

        temp_mean = 2 *tau_eff[indx2] * (self.Dawson1.int_fast(ub) - self.Dawson1.int_fast(lb))

        mean_out[indx2] = 1 / (temp_mean + self.t_ref)

        # Region 2 is calculated with analytical limit as sbar --> 0
        indx3 = np.logical_and(~indx0, ubar <= self.vol_th /tau_eff)
        indx4 = np.logical_and(~indx0, ubar > self.vol_th /tau_eff)
        mean_out[indx3] = 0.0
        mean_out[indx4] = 1 / (self.t_ref -  tau_eff[indx4] * np.log(1 - self.vol_th /tau_eff[indx4] / ubar[indx4]))

        return mean_out

    def forward_fast_std(self, ubar, sbar, tau_eff, u_a):
        '''Calculates the std of output firing rate given the mean & std of input firing rate'''

        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.vol_th /tau_eff - ubar) < (self.cut_off / np.sqrt(tau_eff) * sbar)
        indx2 = indx0 & indx1

        fano_factor = np.zeros(ubar.shape)  # Fano factor

        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally
        ub = (self.vol_th /tau_eff[indx2] - ubar[indx2]) / (sbar[indx2] / np.sqrt(tau_eff[indx2]))
        lb = (self.vol_rest /tau_eff[indx2] - ubar[indx2]) / (sbar[indx2] / np.sqrt(tau_eff[indx2]))

        # cached mean used
        varT = 8 *tau_eff[indx2]*tau_eff[indx2] * (self.Dawson2.int_fast(ub) - self.Dawson2.int_fast(lb))
        fano_factor[indx2] = varT * u_a[indx2] * u_a[indx2]

        # Region 2 is calculated with analytical limit as sbar --> 0
        fano_factor[~indx0] = (ubar[~indx0] < self.vol_th /tau_eff[~indx0]) + 0.0

        std_out = np.sqrt(fano_factor * u_a)
        return std_out
    
    def activate(self, eff_input_mean, eff_input_std, tau_eff):
        # for pytorch
        device = eff_input_mean.device
        
        eff_input_mean = eff_input_mean.cpu().numpy()
        eff_input_std = eff_input_std.cpu().numpy()
        tau_eff = tau_eff.cpu().numpy()
        
        mean_out = self.forward_fast_mean(eff_input_mean, eff_input_std, tau_eff)
        std_out = self.forward_fast_std(eff_input_mean, eff_input_std, tau_eff, mean_out)
        
        mean_out = torch.tensor(mean_out, device=device)
        std_out = torch.tensor(std_out, device=device)
        return mean_out, std_out
    

        

def plot_the_map():
    ma = Moment_Activation_Cond()
    # print all properties and methods of the class
    print(vars(ma))

    n = 51
    exc_rate = np.linspace(0,2,n) # firng rate in kHz
    inh_rate = np.linspace(0,0.1,n)
    
    X, Y = np.meshgrid(exc_rate, inh_rate, indexing='xy')
    
    eff_input_mean, eff_input_std, tau_eff = ma.cond2curr(X,X,Y,Y)
    #add external input current here if needed

    mean_out = ma.forward_fast_mean( eff_input_mean, eff_input_std, tau_eff)
    std_out = ma.forward_fast_std( eff_input_mean, eff_input_std, tau_eff, mean_out)

    # print('Output spike stats:')
    # print(mean_out)
    # print(std_out)

    extent = (exc_rate[0],exc_rate[-1],inh_rate[0],inh_rate[-1])
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(eff_input_mean*tau_eff, origin='lower', extent=extent, aspect='auto')
    plt.ylabel('Inh input rate (sp/ms)')
    plt.title('Eff. rev. pot.')
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.imshow(eff_input_std, origin='lower', extent=extent, aspect='auto')
    plt.title('Eff. input std')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.imshow(mean_out, origin='lower', extent=extent, aspect='auto')
    plt.xlabel('Exc input rate (sp/ms)')
    plt.ylabel('Inh input rate (sp/ms)')
    plt.title('Output mean')
    plt.colorbar()

    plt.subplot(2,2,4)
    plt.imshow(std_out, origin='lower', extent=extent, aspect='auto')
    plt.xlabel('Exc input rate (sp/ms)')
    plt.title('Output std')
    plt.tight_layout()
    plt.colorbar()

    plt.savefig('projects/dtb/temp.png',dpi=300)


if __name__=='__main__':
    plot_the_map()