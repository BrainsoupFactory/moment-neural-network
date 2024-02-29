import torch
import torch.nn as nn
#from mnn.mnn_core import mnn_activate_no_rho #import moment activation
from projects.crit.activation_lookup import MomentActivationLookup
# TODO: double check code for if it uses std or var
import numpy as np
import time
import logging

def gen_config(N=100, ie_ratio=4.0, bg_rate=10.0, w=0.1, w_bg=0.1): #generate config file
    ''' Default parameter values. Recommended not to change anything here, but update it per application.'''
    config = {
    'Vth': 20, #mV, firing threshold, default 20
    'Vres': 10, #mV reset potential; default 0
    'Tref': 2, #ms, refractory period, default 5
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
    'sparse_weight': False, #use sparse weight matrix; not necessarily faster but saves memory
    'randseed':None,
    'delay': 0.0, # default 0.1; synaptic delay (uniform) in Brunel it's around 2 ms (relative to 20 ms mem time scale)
    'dt':0.5, # integration time step for mnn; default 0.02 for oscillation; use 0.1 without oscillation
    'T':10, #simualtion duration
    'record_ts':False,
    }

    return config

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
    
    return W



class StaticRecurrentLayer_backup(nn.Module):
    '''A static recurrent layer with fixed parameters;
        Use custom backward pass based on linear response theory
        Only supports mean and variance, but not corr.
        Replaces static neural activation.

        To be depreciated.
    '''
    def __init__(self, W, config):
        super(StaticRecurrentLayer, self).__init__()
        
        self.NE = config['NE']
        self.NI = config['NI'] # number of neuron per layer
        self.N = self.NE+self.NI
        self.dt = config['dt'] #integration time-step # we only care about steady state, so make this larger for speed
        self.tau = 1 #synaptic? time constant        
        self.W = W
        self.bg_input = config['bg_input']

        
        self.maf = MomentActivation()
        self.maf.Vth = config['Vth']
        self.maf.Vres = config['Vres']
        self.maf.Tref = config['Tref']
        self.delay = config['delay']
        
        self.batchsize = config['batchsize']
        
    def forward(self, ff_mean, ff_std):
        '''# 
        Custom forward pass. Inputs are mean/std of external input currents
        '''
        T = 10
        record_ts = True
        
        self.nsteps = int(T/self.dt)
        self.delay_steps = int(self.delay/self.dt)             
        
        # initial condition
        u = torch.zeros(self.batchsize,self.N) #just 1D array, no column/row 
        s = torch.zeros(self.batchsize,self.N)
        
        if record_ts: # cached data for synaptic delay
            U = torch.zeros(self.batchsize, self.N, self.nsteps)
            S = torch.zeros(self.batchsize, self.N, self.nsteps)
        
        cache_U = torch.zeros(self.batchsize, self.N, self.delay_steps+1 ) # NB for 1 step of delay, need to save step+1 entries
        cache_S = torch.zeros(self.batchsize, self.N, self.delay_steps+1 )
            
        a = self.dt/self.tau
        
        for i in range(self.nsteps):
            logging.debug('Iteration {}/{}'.format(i,self.nsteps))
            if record_ts: #save time series data
                U[:,:,i] = u
                S[:,:,i] = s
            
            # read oldest cached data
            u_delayed = cache_U[:,:,-1].unsqueeze(-1)
            s_delayed = cache_S[:,:,-1].unsqueeze(-1)                
            
            # update cache
            cache_U = torch.roll(cache_U,1,dims = 2)
            cache_S = torch.roll(cache_S,1,dims = 2)                
            cache_U[:,:,0] = u 
            cache_S[:,:,0] = s
            
            # calculate synaptic current; stimulus is added here
            mean_curr = self.W @ u_delayed + self.bg_input + ff_mean  
            std_curr = torch.sqrt((self.W**2) @ (s_delayed**2) + ff_std**2)
                        
            maf_u, maf_s = mnn_activate_no_rho(curr_mean, curr_std)
            
            # evolve one step in time
            u = (1-a)*u + a*maf_u.squeeze()
            s = (1-a)*s + a*maf_s.squeeze()
        
        #cache the grad of moment activation, after the loop is done
        #self.grad_uu, self.grad_us = maf.grad_mean( mean_curr.numpy(), std_curr.numpy() )
        #self.grad_su, self.grad_ss = maf.grad_std( mean_curr.numpy(), std_curr.numpy() )
        
        if record_ts: # only use this for debugging.
            return U, S
        else:
            return u, s

    def backward(self, grad_output):
        # TODO: Custom backward pass
        # called when calling loss.backward()
        # grad_output = dL/dy
        # what i need to calculate is dy/dx, but no explicit storage of it needed
        # grad_input = dL/dx
        # Gradient with respect to the input
        
        #dL/dx = dL/dy * dy/dx
        
        # try both and test speed! then stick to one
        # Method 1: solve linear system dL/dx * dx/dy = dL/dy
        #  batch*N*N, size a bit too big?
        # Method 2: store dy/dx, then do matrix mult


        return grad_input

class StaticRecurrentLayer(nn.Module):
    '''A static recurrent layer with fixed parameters;
        Only supports mean and variance, but not corr.
        Replaces static neural activation.
    '''
    def __init__(self, W, config):
        super(StaticRecurrentLayer, self).__init__()
        
        self.NE = config['NE']
        self.NI = config['NI'] # number of neuron per layer
        self.N = self.NE+self.NI
        self.dt = config['dt'] #integration time-step # we only care about steady state, so make this larger for speed
        self.tau = 1 #synaptic? time constant        
        self.W = W.T #transpose weight for faster matrix multiplication
        # self.W = self.W.to_sparse_csc() # compressed sparse column format (actually slower on GPU, but saves space)
        self.nsteps_kept = 5 # only keep last k steps for truncated BPTT

        #calculate background current stats
        bg_rate = config['bg_rate']
        we = config['w_bg']        
        self.bg_mean = we*bg_rate
        self.bg_var = we*we*bg_rate 
        
        self.delay = config['delay']
        self.record_ts = config['record_ts']
        #self.batchsize = config['batchsize']
        
        self.T = config['T']
        self.record_ts = config['record_ts']

        self.ma = MomentActivationLookup() #initialize moment activation
        
    def forward(self, ff_mean, ff_std):
        '''# 
        Custom forward pass. Inputs are mean/std of external input currents
        '''
  
        
        self.nsteps = int(self.T/self.dt)
        self.delay_steps = int(self.delay/self.dt)
        self.batchsize = ff_mean.shape[0]

        ff_mean = ff_mean#.unsqueeze(-1)
        ff_std = ff_std#.unsqueeze(-1)

        # initial condition
        u = torch.zeros(self.batchsize,self.N) #just 1D array, no column/row 
        s = torch.zeros(self.batchsize,self.N)
        
        if self.record_ts: # cached data for synaptic delay
            U = torch.zeros(self.batchsize, self.N, self.nsteps)
            S = torch.zeros(self.batchsize, self.N, self.nsteps)
        
        cache_U = torch.zeros(self.batchsize, self.N, self.delay_steps+1 ) # NB for 1 step of delay, need to save step+1 entries
        cache_S = torch.zeros(self.batchsize, self.N, self.delay_steps+1 )
            
        a = self.dt/self.tau
        
        for i in range(self.nsteps):
            logging.debug('Iteration {}/{}'.format(i,self.nsteps))
            if self.record_ts: #save time series data
                U[:,:,i] = u
                S[:,:,i] = s
            
            # read oldest cached data
            u_delayed = cache_U[:,:,-1]#.unsqueeze(-1)
            s_delayed = cache_S[:,:,-1]#.unsqueeze(-1)                
            
            # update cache
            cache_U = torch.roll(cache_U,1,dims = 2)
            cache_S = torch.roll(cache_S,1,dims = 2)                
            cache_U[:,:,0] = u 
            cache_S[:,:,0] = s
            
            
            curr_mean = torch.mm(u_delayed, self.W) + self.bg_mean + ff_mean  
            curr_std = torch.sqrt( torch.mm( s_delayed.pow(2), self.W)  + self.bg_var + ff_std) #change name std to var later
            
            #curr_mean = torch.sparse.mm(u_delayed, self.W) + self.bg_mean + ff_mean  
            #curr_std = torch.sqrt( torch.sparse.mm( s_delayed.pow(2), self.W)  + self.bg_var + ff_std) #change name std to var later
            
            maf_u, maf_s = self.ma(curr_mean.squeeze(), curr_std.squeeze()) # input dim should be batch x #neurons
            #maf_u, maf_s = mnn_activate_no_rho(curr_mean, curr_std)
            #maf_u, maf_s = OriginMnnActivation(curr_mean, curr_std) # issue with if statement check corr: u.size(-1) != 1 and cov.dim() > u.dim()
            # which fails when u is unsqueezed

            # evolve one step in time
            u = (1-a)*u + a*maf_u.squeeze()
            s = (1-a)*s + a*maf_s.squeeze()

            if self.nsteps - i == self.nsteps_kept: 
                u = u.detach()
                s = s.detach()
         # output the variance instead std
        if self.record_ts: # use this if loss takes in multiple time steps.
            return U, torch.pow(S,2)
        else:
            return u, torch.pow(s,2)


if __name__=='__main__':
    
    torch.cuda.set_device(0)
    
    logging.basicConfig(level=logging.DEBUG) #this prints debug messages
    config = gen_config(N=12500, ie_ratio=4.0, bg_rate=20.0) 
    print('Generating synaptic weights...')
    W = gen_synaptic_weight(config)
    print('Initializing static recurrent layers...')
    static_rec_layer = StaticRecurrentLayer(W, config)
    print('Testing forward pass...')
    t0=time.perf_counter()

    batchsize = 100
    u = torch.rand( batchsize, config['NE']+config['NI'])
    s = torch.rand( batchsize, config['NE']+config['NI'])
    U,S = static_rec_layer( u, s )
    print('Time elapsed: ', int(time.perf_counter()-t0))

#    plt.plot(U[0,:,:].mean(0))  #show whether result convergences
#    issue: no graphics on server... either print to file, or run testing sciprt on my local machine

    print('\007')
