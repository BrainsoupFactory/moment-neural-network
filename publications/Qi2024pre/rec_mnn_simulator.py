# -*- coding: utf-8 -*-


# import matplotlib.pyplot as plt
# #from mnn_core.maf import MomentActivation
# import numpy as np
# import time

import torch
from mnn.mnn_core import mnn_activate_no_rho #import moment activation
from mnn.mnn_core.mnn_utils import Mnn_Core_Func
#from projects.crit.static_recurrent_layer import gen_config
#from projects.crit.activation_lookup import MomentActivationLookup
import numpy as np
import time, sys, os
import logging
from pprint import pprint
from matplotlib import pyplot as plt
#np.random.seed(1)

def gen_config(N=100, ie_ratio=4.0, bg_rate=10.0, w=0.1, w_bg=0.1, hetero=None, device='cpu'): #generate config file
    ''' Default parameter values. Recommended not to change anything here, but update it per application.'''
    config = {
    'Vth': 20, #mV, firing threshold, default 20
    'Vres': 0, #mV reset potential; default 0
    'Tref': 5, #ms, refractory period, default 5
    'NE': int(0.8*N),
    'NI': int(0.2*N),
    'ie_ratio': ie_ratio,     #I-E ratio
    'wee':{'mean': w, 'std': 1e-6},
    'wei':{'mean': -w*ie_ratio, 'std': 1e-6},
    'wie':{'mean': w, 'std': 1e-6},    
    'wii':{'mean': -w*ie_ratio, 'std': 1e-6},
    'w_bg': w_bg, # background excitation weight
    'bg_rate': bg_rate, # external firing rate kHz; rate*in-degree*weight = 0.01*1000*0.1 = 1 kHz
    #'wie':{'mean': 5.9, 'std': 0.0},    
    #'wii':{'mean': -9.4, 'std': 0.0},        
    'conn_prob': 0.1, #connection probability; N.B. high prob leads to poor match between mnn and snn
    'degree_hetero': hetero, # in-degree heterogeneity (var/mean); if none, use ER network
    'sparse_weight': False, #use sparse weight matrix; not necessarily faster but saves memory
    'randseed':None,
    'delay': 0.0, # default 0.1; synaptic delay (uniform) in Brunel it's around 2 ms (relative to 20 ms mem time scale)
    'dt':0.5, # integration time step for mnn; default 0.02 for oscillation; use 0.1 without oscillation
    'T':10, #simualtion duration
    'record_ts':False,
    'device': device
    }
    return config

class InputGenerator():
    def __init__(self, config):
        self.NE = config['NE']
        self.NI = config['NI']
        self.N = config['NE']+config['NI']
        self.w_bg = config['w_bg']
        self.device = config['device']
        
        return

    def uncorr_input(self, input_rate_array):
        # input argument dimensions: batchsize x #neurons
        input_mean = self.w_bg*input_rate_array*torch.ones(1,self.N, device=self.device) #input current mean        
        input_var = self.w_bg*self.w_bg*input_rate_array*torch.ones(1,self.N, device=self.device)
        return input_mean, input_var                   
        
# issue: ie_ratio is a property of the 


class InputGenerator_constant():
    def __init__(self, config):
        self.NE = config['NE']
        self.NI = config['NI']
        self.N = config['NE']+config['NI']
        self.uext = config['uext']        
        #define external input mean
        #
        self.input_mean = self.uext*torch.ones(self.N,1) #input current mean        
        #calculate external input cov (zero cov)
        self.input_cov = torch.zeros(self.N,self.N)
        
        return

def gen_synaptic_weight(config):
    Ne = config['NE']
    Ni = config['NI']
    N = Ne + Ni

    if config['randseed'] is None:
        W = torch.randn(N, N)
        coin_toss = torch.rand(N, N)
    else:
        torch.random.manual_seed(config['randseed'])
        W = torch.randn(N, N)
        coin_toss = torch.rand(N, N)

    # Excitatory weight
    W[:Ne, :Ne] = W[:Ne, :Ne] * config['wee']['std'] + config['wee']['mean']
    W[Ne:, :Ne] = W[Ne:, :Ne] * config['wie']['std'] + config['wie']['mean']
    W[:, :Ne] = torch.abs(W[:, :Ne])

    # Inhibitory weight
    W[:Ne, Ne:] = W[:Ne, Ne:] * config['wei']['std'] + config['wei']['mean']
    W[Ne:, Ne:] = W[Ne:, Ne:] * config['wii']['std'] + config['wii']['mean']
    W[:, Ne:] = -torch.abs(W[:, Ne:])

    # Apply connection probability (indegree should then be Poisson)
    W[coin_toss > config['conn_prob']] = 0

    # Remove diagonal (self-connection)
    W.fill_diagonal_(0)
    
    return W.to(config['device'])

#TODO: reimplement in pytorch
def gen_synaptic_weight_constant_degree(config):
    '''
    Generate synaptic weight matrix with fixed degree
    The point is to remove all sources of quenched noise
    
    '''
    Ne = config['NE']
    Ni = config['NI']
    N = Ne+Ni

    # constant degree for E/I inputs; NB this only depends on pre-synaptic neurons
    KE = int(config['conn_prob']*Ne)
    KI = int(config['conn_prob']*Ni)
    
    W = np.zeros((N,N))
    
    for i in range(N):
        if config['randseed'] is None:            
            rand_indx = np.random.choice(Ne, KE, replace=False)
            W[i,rand_indx] = config['wee']['mean']
            rand_indx = np.random.choice(Ni, KI, replace=False) + Ne
            W[i,rand_indx] = config['wii']['mean']
        else:
            rng = np.random.default_rng( config['randseed'] )            
            rand_indx = rng.choice(Ne, KE, replace=False)
            W[i,rand_indx] = config['wee']['mean']
            rand_indx = rng.choice(Ni, KI, replace=False) + Ne
            W[i,rand_indx] = config['wii']['mean']
    
    #remove diagonal (self-conneciton)
    #np.fill_diagonal(W,0)
    
    if config['sparse_weight']:
        W = sp.sparse.csr_matrix(W) # W.dot() is efficient but not ().dot(W)        
    return W

def draw_gamma(n, mu, v):
    ''' draw gamma random variables given mean and variance'''
    if v > 0:
        k = mu**2 / v
        theta = v / mu
        K = np.random.gamma(k, theta, n)
    else:
        K = mu*np.ones(n)        
    return K

#TODO: reimplement in pytorch
def gen_synaptic_weight_vary_heterogeneity(config):
    '''
    Generate synaptic weight matrix with fixed degree
    The point is to remove all sources of quenched noise
    
    '''
    Ne = config['NE']
    Ni = config['NI']
    N = Ne+Ni

    # constant degree for E/I inputs; NB this only depends on pre-synaptic neurons
    mean_KE = int(config['conn_prob']*Ne)
    mean_KI = int(config['conn_prob']*Ni)
    hetro = config['degree_hetero'] # in-degree heterogeneity: var/mean in-degree; 1 = ER network
    
    KE = draw_gamma(N, mean_KE, int(hetro*mean_KE) ) 
    KI = draw_gamma(N, mean_KI, int(hetro*mean_KI) ) 
    
    W = np.zeros((N,N))
    
    for i in range(N):
        if config['randseed'] is None:            
            rand_indx = np.random.choice(Ne, int(KE[i]), replace=False)
            W[i,rand_indx] = config['wee']['mean']
            rand_indx = np.random.choice(Ni, int(KI[i]), replace=False) + Ne
            W[i,rand_indx] = config['wii']['mean']
        else:
            rng = np.random.default_rng( config['randseed'] )            
            rand_indx = rng.choice(Ne, int(KE[i]), replace=False)
            W[i,rand_indx] = config['wee']['mean']
            rand_indx = rng.choice(Ni, int(KI[i]), replace=False) + Ne
            W[i,rand_indx] = config['wii']['mean']
    
    #remove diagonal (self-conneciton)
    #np.fill_diagonal(W,0)
    
    if config['sparse_weight']:
        W = sp.sparse.csr_matrix(W) # W.dot() is efficient but not ().dot(W)        
    return W


class RecurrentMNN():
    def __init__(self, config, W):
        self.NE = config['NE']
        self.NI = config['NI'] # number of neuron per layer
        self.N = self.NE + self.NI
        self.dt = config['dt'] #integration time-step # we only care about steady state, so make this larger for speed
        self.tau = config['tau']
        
        self.W = W #synaptic weight matrix (csr)
        self.W_trans = W.T # store transpose (csc)
        #self.input_gen = input_gen # input generator, class object
        #self.ma = mnn_activate_no_rho # zzc's implementation (cpy only)
        self.ma = Mnn_Core_Func()
        #self.maf = MomentActivationLookup(device=config['device']) # look up table (GPU)
        #self.maf = MomentActivation() #vanilla implementation (cpu only)
        #self.maf.Vth = config['Vth']
        #self.maf.Vres = config['Vres']
        #self.maf.Tref = config['Tref']
        self.delay = config['delay']
        self.T = config['T']
        self.record_ts = config['record_ts']
        self.bg_mean = config['bg_rate']
        self.bg_var = config['bg_rate']
        
        # apply neuronal parameter
        self.ma.vol_th = config['Vth']
        self.ma.vol_rest = config['Vres']
        self.ma.t_ref = config['Tref']

        
        
    def run(self, ff_mean, ff_var):
        '''# 
        Custom forward pass. Inputs are mean/std of external input currents
        '''
        
        self.nsteps = int(self.T/self.dt)
        #self.delay_steps = int(self.delay/self.dt)
        self.delay_steps = int(self.delay/self.dt)+1
        self.batchsize = ff_mean.shape[0]

        #ff_mean = ff_mean#.unsqueeze(-1)
        #ff_var = ff_var#.unsqueeze(-1)

        # initial condition
        u = torch.zeros(self.batchsize,self.N, device=ff_mean.device) #just 1D array, no column/row 
        s = torch.zeros(self.batchsize,self.N, device=ff_mean.device)
        
        if self.record_ts: # cached data for synaptic delay
            U = torch.zeros(self.batchsize, self.N, self.nsteps, device='cpu')
            S = torch.zeros(self.batchsize, self.N, self.nsteps, device='cpu')
            Ubar = U.clone()
            Sbar = S.clone()

        
        #cache_U = torch.zeros(self.batchsize, self.N, self.delay_steps+1, device=ff_mean.device) # NB for 1 step of delay, need to save step+1 entries
        #cache_S = torch.zeros(self.batchsize, self.N, self.delay_steps+1, device=ff_mean.device)
        cache_U = torch.zeros(self.batchsize, self.N, self.delay_steps, device=ff_mean.device)
        cache_S = torch.zeros(self.batchsize, self.N, self.delay_steps, device=ff_mean.device)
        
        read_pointer = -1
        write_pointer = 0
        
        a = self.dt/self.tau
        
        for i in range(self.nsteps):
            if i % int(self.nsteps/10) == 0:
                logging.debug('Iteration {}/{}'.format(i,self.nsteps))
            
            # read oldest cached data
            #u_delayed = cache_U[:,:,-1]#.unsqueeze(-1)
            #s_delayed = cache_S[:,:,-1]#.unsqueeze(-1)
            
            # update cache
            #cache_U = torch.roll(cache_U,1,dims = 2)
            #cache_S = torch.roll(cache_S,1,dims = 2)                
            #cache_U[:,:,0] = u 
            #cache_S[:,:,0] = s
            
            u_delayed = cache_U[:,:,read_pointer]#.unsqueeze(-1)
            s_delayed = cache_S[:,:,read_pointer]#.unsqueeze(-1)
            
            cache_U[:,:,write_pointer] = u
            cache_S[:,:,write_pointer] = s
            
            #advance cached time by 1 step
            read_pointer = np.mod(read_pointer-1,self.delay_steps)
            write_pointer = np.mod(write_pointer-1, self.delay_steps)
            
            
            curr_mean = torch.mm(u_delayed, self.W_trans) + self.bg_mean + ff_mean
            curr_std = torch.sqrt( torch.mm( s_delayed.pow(2), self.W_trans)  + self.bg_var + ff_var) #change name std to var later
            
            #curr_mean = torch.sparse.mm(u_delayed, self.W_trans) + self.bg_mean + ff_mean  
            #curr_std = torch.sqrt( torch.sparse.mm( s_delayed.pow(2), self.W_trans)  + self.bg_var + ff_var) #change name std to var later
            
            #maf_u, maf_s = self.ma(curr_mean.squeeze(), curr_std.squeeze()) # input dim should be batch x #neurons
            
            
            # not so elegant implementation of moment activation
            curr_mean = curr_mean.squeeze().cpu().numpy()
            curr_std = curr_std.squeeze().cpu().numpy()
            maf_u = self.ma.forward_fast_mean(curr_mean, curr_std)
            maf_s = self.ma.forward_fast_std(curr_mean, curr_std, maf_u)
            maf_u = torch.from_numpy(maf_u).to(torch.float64).to(ff_mean.device)
            maf_s = torch.from_numpy(maf_s).to(torch.float64).to(ff_mean.device)

            # evolve one step in time
            u = (1-a)*u + a*maf_u.squeeze()
            s = (1-a)*s + a*maf_s.squeeze()

            if self.record_ts: #save time series data
                U[:,:,i] = u.cpu()
                S[:,:,i] = s.cpu()
                #Ubar[:,:,i] = curr_mean.cpu()
                #Sbar[:,:,i] = curr_std.cpu()

         # output the variance instead std
        if self.record_ts: # use this if loss takes in multiple time steps.
            return U, S
        else:
            return u, s

def run(exp_id, indx, device='cpu', savefile = False):
    
    input_rate_array = torch.tensor([20,40], device=device).unsqueeze(1)
    
    ie_ratio_array = np.linspace(0, 8.0, 41)
    mnn_delay_array = np.array([1.5/1.168])
    
    X,Y = np.meshgrid(ie_ratio_array, mnn_delay_array)
    ie_ratio = X.flatten()[indx]
    delay = Y.flatten()[indx]
    
    config = gen_config(N=12500, ie_ratio=ie_ratio, bg_rate=0.0, device=device)
    config['T'] = 400 # ms
    config['dt'] = 0.02 # ms
    config['tau'] = 1 # calibrated time constant in ms
    config['record_ts'] = True
    config['Vth']=20 # no effect for now; using default parameters
    config['Vres']=10
    config['Tref']=2    
    config['delay'] = delay
    config['device']=device
    config['randseed']=0

    pprint(config)

    print('Generating synaptic weights...')
    W = gen_synaptic_weight(config)
    print('Initializing static recurrent layers...')
    rec_mnn = RecurrentMNN(config, W)
    input_gen= InputGenerator(config)
    input_mean, input_var = input_gen.uncorr_input(input_rate_array)
    
    print('Testing forward pass...')
    t0=time.perf_counter()
    out_mean, out_std = rec_mnn.run( input_mean, input_var )
    print('Time elapsed: ', int(time.perf_counter()-t0))
    
    pop_mean = out_mean.mean(1).squeeze()
    pop_std = out_std.mean(1).squeeze()

    if savefile:
        path =  './projects/pre2023/runs/{}/'.format( exp_id )
        dat = {
            'mnn_mean': out_mean[:,:,-1].cpu().numpy(),
            'mnn_std': out_std[:,:,-1].cpu().numpy(),
            'mnn_mean_ts': None, #save time series
            'mnn_std_ts': None, 
            'pop_mean': pop_mean,
            'pop_std': pop_std,
            'config': config,
            'indx':indx,
            'exp_id':exp_id,
            'W': None, #W.cpu().numpy(), don't save weight, too big, save seed is enough
            'ie_ratio_array': ie_ratio_array,
            'input_rate_array': input_rate_array.cpu().numpy(),
            'mnn_delay_array': mnn_delay_array,
            'input_mean': input_mean.cpu().numpy(),
            'input_var': input_var.cpu().numpy(),
        }
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        file_name = str(indx).zfill(3)
        
        np.savez(path +'{}.npz'.format(file_name), **dat)#, spike_count=spike_count)
        print('Results saved to: '+path +'{}.npz'.format(file_name))
        
        # plot preview
        plt.figure(figsize=(3.5,3))
        t = np.linspace(0, config['T'], out_mean.shape[-1])
        plt.subplot(2,1,1)
        for i in range(out_mean.shape[0]):
            plt.plot(t, out_mean[i,:,:].mean(0).squeeze())
        plt.ylabel('$\mu$')        
        plt.subplot(2,1,2)
        for i in range(out_mean.shape[0]):
            plt.plot(t, out_std[i,:,:].mean(0).squeeze())
        plt.ylabel('$\sigma$')
        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(path+file_name+'.png')
        plt.close('all')
        

if __name__=='__main__':    
    torch.set_default_dtype(torch.float64)
    logging.basicConfig(level=logging.DEBUG) #this prints debug messages
    device = 'cuda'

    if len(sys.argv)>1:
        indx = int(sys.argv[1])
        exp_id = sys.argv[2]
        print('Running trial ', indx)
    else:
        indx = 3
        exp_id = 'test_mnn_delay'

    run(exp_id, indx, device=device, savefile = True)

    print('Done!')

    

