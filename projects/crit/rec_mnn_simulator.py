# -*- coding: utf-8 -*-


# import matplotlib.pyplot as plt
# #from mnn_core.maf import MomentActivation
# import numpy as np
# import time

import torch
from mnn.mnn_core import mnn_activate_no_rho #import moment activation
from projects.crit.static_recurrent_layer import gen_config
#from projects.crit.activation_lookup import MomentActivationLookup
import numpy as np
import time, sys, os
import logging
from pprint import pprint

#np.random.seed(1)



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
        with torch.random.manual_seed(config['randseed']):
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
        self.tau = 1 #synaptic? time constant
        
        self.W = W #synaptic weight matrix (csr)
        self.W_trans = W.T # store transpose (csc)
        #self.input_gen = input_gen # input generator, class object
        self.ma = mnn_activate_no_rho # zzc's implementation (cpy only)
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
        
        
    def run(self, ff_mean, ff_var):
        '''# 
        Custom forward pass. Inputs are mean/std of external input currents
        '''
        
        self.nsteps = int(self.T/self.dt)
        self.delay_steps = int(self.delay/self.dt)
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

        
        cache_U = torch.zeros(self.batchsize, self.N, self.delay_steps+1, device=ff_mean.device) # NB for 1 step of delay, need to save step+1 entries
        cache_S = torch.zeros(self.batchsize, self.N, self.delay_steps+1, device=ff_mean.device)
            
        a = self.dt/self.tau
        
        for i in range(self.nsteps):
            if i % int(self.nsteps/10) == 0:
                logging.debug('Iteration {}/{}'.format(i,self.nsteps))
            
            # read oldest cached data
            u_delayed = cache_U[:,:,-1]#.unsqueeze(-1)
            s_delayed = cache_S[:,:,-1]#.unsqueeze(-1)
            
            # update cache
            cache_U = torch.roll(cache_U,1,dims = 2)
            cache_S = torch.roll(cache_S,1,dims = 2)                
            cache_U[:,:,0] = u 
            cache_S[:,:,0] = s
            
            
            curr_mean = torch.mm(u_delayed, self.W_trans) + self.bg_mean + ff_mean
            curr_std = torch.sqrt( torch.mm( s_delayed.pow(2), self.W_trans)  + self.bg_var + ff_var) #change name std to var later
            
            #curr_mean = torch.sparse.mm(u_delayed, self.W_trans) + self.bg_mean + ff_mean  
            #curr_std = torch.sqrt( torch.sparse.mm( s_delayed.pow(2), self.W_trans)  + self.bg_var + ff_var) #change name std to var later
            
            maf_u, maf_s = self.ma(curr_mean.squeeze(), curr_std.squeeze()) # input dim should be batch x #neurons
            #maf_u, maf_s = curr_mean.clone(), curr_std.clone()
            #maf_u, maf_s = mnn_activate_no_rho(curr_mean, curr_std)
            #maf_u, maf_s = OriginMnnActivation(curr_mean, curr_std) # issue with if statement check corr: u.size(-1) != 1 and cov.dim() > u.dim()
            # which fails when u is unsqueezed

            # evolve one step in time
            u = (1-a)*u + a*maf_u.squeeze()
            s = (1-a)*s + a*maf_s.squeeze()

            if self.record_ts: #save time series data
                U[:,:,i] = u.cpu()
                S[:,:,i] = s.cpu()
                Ubar[:,:,i] = curr_mean.cpu()
                Sbar[:,:,i] = curr_std.cpu()

         # output the variance instead std
        if self.record_ts: # use this if loss takes in multiple time steps.
            return U, S
        else:
            return u, s

def run(exp_id, indx, device='cpu', savefile = False):
    ie_ratio_array = torch.linspace(0,8,41).unsqueeze(1)
    input_rate_array = torch.linspace(0,40,41, device=device).unsqueeze(1)
    
    ie_ratio = ie_ratio_array[indx]

    config = gen_config(N=12500, ie_ratio=ie_ratio, bg_rate=0.0, device=device)
    config['T'] = 10 # a.u.
    config['dt'] = 0.1 # a.u.
    config['record_ts'] = True
    config['Vth']=20 # no effect for now; using default parameters
    config['Vres']=0
    config['Tref']=5    
    config['device']=device

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

    if savefile:
        path =  './projects/crit/runs/{}/'.format( exp_id )
        dat = {
            'mnn_mean': out_mean.cpu().numpy(),
            'mnn_std': out_std.cpu().numpy(),
            'config': config,
            'indx':indx,
            'exp_id':exp_id,
            'W': W.cpu().numpy(),
            'ie_ratio_array': ie_ratio_array.cpu().numpy(),
            'input_rate_array': input_rate_array.cpu().numpy(),
            'input_mean': input_mean.cpu().numpy(),
            'input_var': input_var.cpu().numpy(),
        }
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        file_name = str(indx).zfill(3)
        
        np.savez(path +'{}.npz'.format(file_name), **dat)#, spike_count=spike_count)
        print('Results saved to: '+path +'{}.npz'.format(file_name))

if __name__=='__main__':    
    torch.set_default_dtype(torch.float64)
    logging.basicConfig(level=logging.DEBUG) #this prints debug messages
    device = 'cuda'

    if len(sys.argv)>1:
        indx = int(sys.argv[1])
        exp_id = sys.argv[2]
        print('Running trial ', indx)
    else:
        indx = 5
        exp_id = 'test_mnn'

    run(exp_id, indx, device=device, savefile = True)

    print('Done!')

    

