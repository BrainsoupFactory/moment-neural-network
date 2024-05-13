# given we and wi in MNN
# auto tune tau_E and tau_I such that neuronal activation matches with snn
# NB tuned parameter should work well for all inputs
# i.e. tuning done at individual neuron level, no need to worry about network

# PSEUDO CODE
# for excitatory populations
#   for all pairs of (w_ee, w_ei, w_ex) - there are NE pairs 
#       sample all exc_rate and inh_rate, find optimal tau_E
# for inhibitory populations
#   for all pairs of (w_ie, w_ii) - there are NI pairs 
#       sample all exc_rate and inh_rate, find optimal tau_I
# so the complexity is not too bad - it's linear in the number of populations

import torch
import matplotlib.pyplot as plt
import sys, os
from projects.dtb.snn_simulator_torch import CondInteNFire, gen_config
import numpy as np
from scipy.optimize import minimize, basinhopping
from projects.dtb.ma_conductance import Moment_Activation_Cond
from projects.dtb.optim_empirical_tau import func

class SNNInputGenerator():
    def __init__(self, config, input_mean=None, input_std=None, input_corr=None, rho=None, w=None, ei_spk_stats=None, exc_rate=None, inh_rate=None, device='cpu'):
        self.NE = config['NE']
        self.NI = config['NI']
        self.N = config['NE']+config['NI']
        self.batchsize = config['batchsize']
        self.device = device
        
        self.input_mean = input_mean
        self.input_std = input_std

        self.w = w
        self.exc_rate = exc_rate
        self.inh_rate = inh_rate

        self.ei_spk_stats = ei_spk_stats  # [exc_spk_mean, exc_spk_var, inh_spk_mean, inh_spk_var]

        self.exc_t2spk = torch.zeros(self.batchsize, self.N, device=device, dtype=torch.int64)
        self.inh_t2spk = torch.zeros(self.batchsize, self.N, device=device, dtype=torch.int64)

        try:
            if not input_corr:    
                input_corr = torch.ones(self.N,self.N, device=device)*rho
                input_corr.fill_diagonal_(1.0)
            
            self.input_corr = input_corr        
            cov = input_corr*input_std*input_std.T
            # Perform Cholesky decomposition
            if np.abs(rho)>1e-6:
                self.chol = torch.linalg.cholesky(cov, upper=True)#.mH # upper cholesky
            else:
                self.chol = None
        except:
            pass            

        return
    
    def gen_corr_normal_rv(self, dt, device='cpu'): #generate correlated gaussian noise
        '''
        OUTPUT: correlated normal random variables (sample size x num elements)
        '''  
        dt = torch.tensor(dt).to(device)
        # NB: assume self.input_mean has size (1,N)
        standard_normal = torch.randn( self.batchsize, self.N, device=device)
        correlated_samples = self.input_mean*dt + torch.sqrt(dt)*torch.matmul(standard_normal, self.chol)
        return correlated_samples
    
    def gen_uncorr_normal_rv(self, dt, device='cpu'):
        dt = torch.tensor(dt).to(device)
        standard_normal = torch.randn( self.batchsize, self.N, device=device)
        correlated_samples = self.input_mean*dt + torch.sqrt(dt)*standard_normal*self.input_std
        return correlated_samples
    
    def gen_uncorr_spk(self, dt, device='cpu'):
        # w, rate dimension: 1 x #neurons
        pe = dt*self.exc_rate
        pi = dt*self.inh_rate        
        # use poisson instead
        spk  = (torch.poisson(pe) - torch.poisson(pi))*self.w
        #spk_e = torch.rand(self.batchsize, self.N, device=device)<pe
        #spk_i = torch.rand(self.batchsize, self.N, device=device)<pi
        #spk = (spk_e.float()-spk_i.float())*self.w
        return spk

    @staticmethod
    def gen_gamma_spk(dt, spk_mean, spk_var, time2nextspike):
        ''' Generate spikes with gamma distributed ISI
            isi_mean = shape*scale (unit: kHz)
            isi_var = shape*scale^2 (unit: kHz)
        '''
        #TODO: creating new instances of Gamma objects can be very slow!

        # calculate ISI parameters given spk count stats
        isi_mean = 1 / (spk_mean+1e-10)
        isi_var = torch.pow(isi_mean, 3) * spk_var        
        
        beta = isi_mean/isi_var
        alpha = isi_mean * isi_mean / isi_var
        
        is_spike = time2nextspike==0      # emit a spike if time-to-next-spike reaches zero
        isi = torch.distributions.Gamma(alpha[is_spike], beta[is_spike]).sample() #sample new ISI
        indx = isi_var[is_spike]==0
        isi[indx] = isi_mean[is_spike][indx] #replace isi with its mean if isi_var is zero.
        time2nextspike[is_spike] = (isi/dt).to(torch.int64) # update time-to-next-spike
        time2nextspike[~is_spike] += -1 # count down time
        
        return is_spike
    
    def gen_gamma_spk_EI(self,dt, device='cpu'):
        #ei_spk_stats  # [exc_spk_mean, exc_spk_var, inh_spk_mean, inh_spk_var]
        exc_spk = self.gen_gamma_spk(dt, self.ei_spk_stats[0], self.ei_spk_stats[1], self.exc_t2spk)
        inh_spk = self.gen_gamma_spk(dt, self.ei_spk_stats[2], self.ei_spk_stats[3], self.inh_t2spk)
        total_current  = (exc_spk.float() - inh_spk.float())*self.w
        #print(total_current[:,-1].mean()/dt) #-1 should correspond to ubar=4, sbar=5;
        #print(total_current[:,-1].std()/np.sqrt(dt))
        # debug
        #dt = torch.tensor(dt).to(device)
        #standard_normal = torch.randn( self.batchsize, self.N, device=device)
        #total_current = 2.0*dt + torch.sqrt(dt)*standard_normal*3.0
        
        return total_current

    def gen_uncorr_spk_one_channel(self, dt, device='cpu'):
        pe = dt*self.input_mean
        pe = pe.expand(self.batchsize, -1) # expand view no new copy
        spk  = self.w*torch.poisson(pe)
        return spk
    

def simulate_spiking_neuron(config, we, wi):
    
    batchsize = config['batchsize']#1000 # number of independent samples (make this large to avoid no spikes)
    num_neurons = config['NE']
    device= config['device']

    rho = 0 # for mean, std mapping need independent samples
    
    #we = 0.1
    #wi = 0.4
    m = int(np.sqrt(num_neurons))
    exc_input_rate = torch.linspace(0,1,m, device=device)
    inh_input_rate = torch.linspace(0,1,m, device=device)
    x_input_rate = 0.0
    
    X, Y = torch.meshgrid(exc_input_rate, inh_input_rate, indexing='xy')
    # inner dim: std, outer dim: mean
    X = X.flatten().unsqueeze(0)
    Y = Y.flatten().unsqueeze(0)

    print('Setting up SNN model...')
    num_neurons = X.shape[1]
    
    x_input_mean = x_input_rate*torch.ones(1, num_neurons, device=device)
    x_input_std = np.sqrt(x_input_rate)*torch.ones(1, num_neurons, device=device)

    print('rho = ', rho)    
    print('Using uncorrelated input.')
    exc_input_gen = SNNInputGenerator(config, input_mean=X, w=we, device=device).gen_uncorr_spk_one_channel
    inh_input_gen = SNNInputGenerator(config, input_mean=Y, w=wi, device=device).gen_uncorr_spk_one_channel
    x_input_gen = SNNInputGenerator(config, input_mean=x_input_mean, input_std=x_input_std, device=device).gen_uncorr_normal_rv

    snn_model = CondInteNFire(config, [exc_input_gen, inh_input_gen, x_input_gen])

    print('Simulating SNN...')
    spk_count, V, t, spk_count_history = snn_model.run( config['T_snn'] , record_interval=config['record_interval'], show_message=True, device = device) # ms

    
    data_dict = {'spk_count': spk_count.cpu().numpy(),
    'config':config,
    'exc_input_rate':exc_input_rate.cpu().numpy(),
    'inh_input_rate':inh_input_rate.cpu().numpy(),
    'x_input_rate':x_input_rate,
    'rho':rho,    
    'spk_count_history':spk_count_history,
    'we':we,
    'wi':wi,
    't':t,
    }
    
    return data_dict


def autotune(dat, config):    

    ''' Initialize moment activation 
        Note: E/I neurons may have different parameters
    '''
    ma = Moment_Activation_Cond()
    #then update model parameters to be consistent with SNN
    ma.tau_L = 20 # membrane time constant
    ma.tau_E = 4 # excitatory synaptic time scale (ms)
    ma.tau_I = 10 # inhibitory synaptic time scale (ms)
    ma.VL = -70 # leak reverseal potential
    ma.VE = 0 # excitatory reversal potential
    ma.VI = -70 # inhibitory reversal potential
    ma.vol_th = -50 # firing threshold
    ma.vol_rest = -60 # reset potential
    ma.L = 1/ma.tau_L
    ma.sE = 1
    ma.sI = 1
    ma.t_ref = 2 # ms; NB inhibitory neuron has different value
    
    # loading spiking neuron parameters
    we=dat['we']
    wi=dat['wi']
    exc_rate = dat['exc_input_rate'] # firng rate in kHz
    inh_rate = dat['inh_input_rate']
    X, Y = np.meshgrid(exc_rate, inh_rate, indexing='xy')
        
    exc_input_mean = we*X
    exc_input_std = we*np.sqrt(X)
    inh_input_mean = wi*Y
    inh_input_std = wi*np.sqrt(Y)
    
    T = config['T_snn']-config['discard'] # ms
    
    mean_firing_rate = dat['spk_count'].mean(0)/T
    firing_var = dat['spk_count'].var(0)/T
    
    # start optimizing

    x0 = [1.0, 1.0] #initial value
    #method = 'BFGS'
    method = 'Nelder-Mead'
    
    params = (ma, mean_firing_rate, firing_var, exc_input_mean.flatten(), exc_input_std.flatten(), inh_input_mean.flatten(), inh_input_std.flatten()
    )
           
    res = minimize(func, x0, args=params, method=method, tol=1e-16)
    
    print('Correction factor found to be: ', res.x)
    
    return res


if __name__=='__main__':
    torch.set_default_dtype(torch.float64) #for accurate corrcoef estimate
    
    #device = 'cpu'
    device = 'cuda'
    batchsize = 1000 # number of independent samples
    T = 1e3 # simulation time (ms) for spiking neurons
    num_neurons = 21**2 # number of neurons must be a square
    config = gen_config(batchsize, num_neurons, T, device=device, dt_snn = 0.01 )
    config['record_interval'] = None #100 # ms. record spike count every x ms; None = don't record

    we, wi = 0.1, 0.4
    dat = simulate_spiking_neuron(config, we, wi)
    res = autotune(dat, config)
