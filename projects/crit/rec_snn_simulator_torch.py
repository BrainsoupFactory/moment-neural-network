# -*- coding: utf-8 -*-
"""
Modified to incorporate delay
Cache spike history
Re-implemented in pytorch to support GPU
@author: dell
"""

import numpy as np
import torch
import scipy as sp
import matplotlib.pyplot as plt
import time
from projects.crit.static_recurrent_layer import gen_synaptic_weight, gen_config

def update_config_snn(config): #update config file for snn
    ''' Use base configs from MNN and add SNN specific configs'''
    # these are default values don't modify it here!
    config_snn = {
    'dT': 100, #ms spike count time window (0.1 ms in Brunel 2000)
    'dt_snn': 0.01, # ms, snn integration time step
    'T_snn':int(1e3), #snn simulation duration
    'delay_snn': 1.5, # (1.5 ms in Brunel 2000)
    'batchsize':10, # parallelize multiple trials with minibatch to shorten simulation time
    }

    config.update(config_snn)

    return config

class SNNInputGenerator():
    def __init__(self, config):
        self.NE = config['NE']
        self.NI = config['NI']
        self.N = config['NE']+config['NI']
        self.bg_rate = config['bg_rate']
        self.w_bg = config['w_bg']
        self.batchsize = config['batchsize']
        self.dt = config['dt_snn']
        
        if self.bg_rate*self.dt >1:
            print('Warning: insufficient temporal resolution!')
        
        return
    
    def ind_poisson(self, dt, device = 'cpu'):
        #dt = simulation time step (ms)
        p = self.bg_rate*self.dt # requires p<1        
        input_current = self.w_bg*(torch.rand(self.batchsize, self.N) < p)
        return input_current.to(device)
    

#!!! might as well take this opportunity to add batch dimension!!!

class InteNFireRNN():
    def __init__(self, config, WT, input_gen):
        ''' W = weight matrix, input_gen = generator for external input'''
        self.L = 1/20 #ms
        self.Vth = config['Vth']
        self.Vres = config['Vres']   
        self.Tref = config['Tref'] #ms
        self.Vspk = 50 #for visualization purposes only
        self.dt = config['dt_snn'] #integration time step (ms)
        self.NE = config['NE']
        self.NI = config['NI']
        
        self.num_neurons = config['NE'] + config['NI']      
        self.batchsize = config['batchsize']
        self.delay = config['delay_snn']
        
        self.WT = WT #transpose of synaptic weight matrix
        self.input_gen = input_gen # input generator, class object
        
    
    def forward(self, v, tref, is_spike, ff_current):
        #compute voltage
        v += -v*self.L*self.dt + torch.matmul(is_spike, self.WT) + ff_current
        
        #compute spikes
        is_ref = (tref > 0.0) & (tref < self.Tref)
        is_spike = (v > self.Vth) & ~is_ref
        is_sub = ~(is_ref | is_spike)
                
        v[is_spike] = self.Vspk
        v[is_ref] = self.Vres
        
        #update refractory period timer
        tref[is_sub] = 0.0
        tref[is_ref | is_spike] += self.dt
        return v, tref, is_spike
        
    
    
    def run(self, T, device = 'cpu', ntrials = None, record_v = False, show_message = False):
        '''Simulate integrate and fire neurons
        T = simulation duration (ms)        
        '''
        
        self.T = T #min(10e3, 100/maf_u) #T = desired number of spikes / mean firing rate
        num_timesteps = int(self.T/self.dt)
        delay_steps = int(self.delay/self.dt) # works when delay is zero
        
        tref = torch.zeros(self.batchsize, self.num_neurons, device=device) #tracker for refractory period
        #v = np.random.rand(self.num_neurons,1)*self.Vth #initial voltage
        v = torch.zeros(self.batchsize, self.num_neurons, device=device)
        is_spike = torch.zeros(self.batchsize, self.num_neurons, device=device)
        
        #SpkTime = [[] for i in range(self.num_neurons)]
        spk_history = np.empty((0,3),dtype=np.uint32) # sample_id x neuron_id x time
        
        t = np.arange(0, self.T , self.dt)
        
        if record_v:
            V = torch.zeros( self.num_neurons, num_timesteps , device = 'cpu') #probably out of memory on gpu
        else:            
            V = None
        
        #v, tref, is_spike
        cache_spk = torch.zeros(self.batchsize, self.num_neurons, delay_steps+1, device = device)
        
        read_pointer = -1
        write_pointer = 0
        
        start_time = time.time()
        for i in range(num_timesteps):
            
            # read oldest cached data
            spk_delayed = cache_spk[:,:,read_pointer]
            
            read_pointer = np.mod(read_pointer-1,delay_steps)
            write_pointer = np.mod(write_pointer-1, delay_steps)
            cache_spk[:,:,write_pointer] = is_spike
            
            #!!! spike input: independent Poisson
            input_current = self.input_gen.ind_poisson(self.dt, device=device)
            
            with torch.no_grad():
                
                v, tref, is_spike = self.forward(v, tref, spk_delayed, input_current)
            
                if record_v:
                    V[:,i] = v[0,:].flatten() #saves only 1 sample from the batch to limit memory consumption
                
                spk_indices = torch.nonzero(is_spike).to('cpu').numpy().astype(np.uint32) #each row is a 2-tuple (sample_id, neuron_id)
            
            spk_time = np.full( (spk_indices.shape[0],1), i , dtype=np.uint32)
            spk_indices = np.hstack( (spk_indices, spk_time ))  # add spike time as column                
            spk_history = np.vstack( (spk_history, spk_indices) ) # 
            
            if show_message and (i+1) % int(num_timesteps/10) == 0:
                progress = (i+1)/num_timesteps*100
                elapsed_time = (time.time()-start_time)/60
                print('Progress: {:.2f}%; Time elapsed: {:.2f} min'.format(progress, elapsed_time ), flush=True)
                
        return spk_history, V, t

def spk_time2count(spk_history, timewindow, config):
    '''Calculate spike count from spike time over a given time window'''
    # binsize: unit in ms
    nbins = int( config['T_snn']/timewindow)
    #bin_edges = np.linspace(0, config['T_snn'], nbins+1)
    num_neurons = config['NE']+config['NI']
    spk_count = np.zeros((config['batchsize'], num_neurons, nbins))
    #print(spk_count.shape)
    # i = iterate through neurons
    # j = iterate through trials
    # then for each i,j, use histogram to count # spikes 
    for i in range(config['batchsize']):
        for j in range(num_neurons):
            indx =  np.logical_and(spk_history[:,0]==i, spk_history[:,1]==j)
            h, bin_edges = np.histogram( spk_history[indx,2]*config['dt_snn'], nbins, range=(0, config['T_snn']))
            spk_count[i,j,:] = h
        return spk_count

def spk_time2csr(spk_history, nneurons, nsteps, sample_id = 0):
    '''
    Converts list of lists of spike time into compressed sparse row matrix
    '''
    sample_filter = (spk_history[:,0] == sample_id)  #find all spks belonging to sample data i
    row_indx = spk_history[sample_filter,1].flatten()
    col_indx = spk_history[sample_filter,2].flatten()    
    value = np.ones(len(row_indx))
    sparse_matrix = sp.sparse.csr_matrix(  (value, (row_indx,col_indx) ), shape=(nneurons, nsteps))

    return sparse_matrix   

#%%    
if __name__=='__main__':
    exp_id = 'test_rec_snn'
    #config = gen_config(N=12500, ie_ratio=4, bg_rate=20)
    N = 12500
    ie_ratio = 6.0
    bg_rate = 40.0
    w = 0.1
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
    'bg_rate': bg_rate, # external firing rate kHz; rate*in-degree*weight = 0.01*1000*0.1 = 1 kHz
    'conn_prob': 0.1, #connection probability; N.B. high prob leads to poor match between mnn and snn
    'sparse_weight': False, #use sparse weight matrix; not necessarily faster but saves memory
    'randseed':None,
    #'dT': 200, #ms spike count time window
    #'delay': 0.1, # synaptic delay (uniform) in Brunel it's around 2 ms (relative to 20 ms mem time scale)
    #'dt':0.02, # integration time step for mnn
    }
    
    config['T_snn']=200 #ms
    config['dt_snn']=0.01 #ms
    config['delay_snn'] = 1.5 #1.5 #ms
    config['batchsize'] = 2
    
    #torch.set_default_dtype(torch.float32)
    
    device = 'cuda'
    config['device']=device

    W = gen_synaptic_weight(config) #doesn't take too much time with 1e4 neurons    
    #W = torch.tensor(W, device = device, dtype=torch.float32)
    #W = torch.tensor(W, device = device, dtype=torch.float64)
    input_gen = SNNInputGenerator(config)
    snn_model = InteNFireRNN(config, W.T , input_gen)
    spk_history, V, t = snn_model.run( config['T_snn'] , show_message=True, device = device) # ms


#%%
    nsteps = int(config['T_snn']/config['dt_snn'])
    nneurons = config['NE']+config['NI']    
    sparse_matrix = spk_time2csr(spk_history, nneurons, nsteps, sample_id = 0)
    
    spk_count = np.asarray(np.sum(sparse_matrix, axis=0))
    binwidth = 10
    spk_count = spk_count.reshape( int(spk_count.shape[1]/binwidth) , binwidth ).sum(axis=1)
    
    tt= np.linspace(0,t[-1], spk_count.shape[0])    
    
    plt.figure(figsize=(8,2))
    plt.bar(tt , spk_count, width = tt[1]-tt[0] )
    plt.xlim([t[0],t[-1]])
    plt.tight_layout()
    plt.savefig('tmp_snn_pop_spike_count.png')
    plt.close('all')


