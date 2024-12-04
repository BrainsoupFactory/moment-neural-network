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
#from projects.crit.static_recurrent_layer import gen_synaptic_weight, gen_config


def gen_config(batchsize, num_neurons, T_snn, device='cpu', dt_snn=0.01): #generate config file
    ''' Default parameter values. Recommended not to change anything here, but update it per application.'''
    config = {
        # neuronal parameter
        'tau_L' : 20, # membrane time constant
        'tau_E' : 4, # excitatory synaptic time scale (ms)
        'tau_I' : 10, # inhibitory synaptic time scale (ms)
        'tau_x' : 4, # external input time scale
        'VL' : -60, # leak reverseal potential
        'VE' : 0, # excitatory reversal potential
        'VI' : -80, # inhibitory reversal potential
        'Vth' : -50, # firing threshold
        'Vres' : -60, # reset potential
        'Tref' : 2, # ms; NB inhibitory neuron has different value
        # simulation parameter
        'NE': num_neurons,
        'NI': 0,
        'sparse_weight': False, #use sparse weight matrix; not necessarily faster but saves memory
        'randseed':None,
        'delay_snn': 0.0, # default 0.1; synaptic delay (uniform) in Brunel it's around 2 ms (relative to 20 ms mem time scale)
        'T_snn':T_snn, #simualtion duration
        'record_ts':False,
        'device': device,
        'dt_snn': dt_snn,
        'batchsize':batchsize,
        'discard': 100, # ms, discard first x ms of data
    }
        
    return config



class CondInteNFire():
    def __init__(self, config, input_gen):
        ''' Conductance-based integrate and fire neuron model. input_gen = generator for external input'''
        self.Vspk = 50 #for visualization purposes only
        self.dt = config['dt_snn'] #integration time step (ms)
        self.NE = config['NE']
        self.NI = config['NI']        
        self.num_neurons = config['NE'] + config['NI']      
        self.batchsize = config['batchsize']
        self.delay = config['delay_snn']
        #self.WT = WT #transpose of synaptic weight matrix
        self.exc_input_gen = input_gen[0] # input generator, class object
        self.inh_input_gen = input_gen[1]
        self.x_input_gen = input_gen[2]
        self.discard = config['discard']
        
        # neuronal parameters
        self.tau_L = config['tau_L'] # membrane time constant
        self.tau_E = config['tau_E'] # excitatory synaptic time scale (ms)
        self.tau_I = config['tau_I'] # inhibitory synaptic time scale (ms)
        self.tau_x = config['tau_x'] # external input time scale (ms)
        self.VL = config['VL'] # leak reverseal potential
        self.VE = config['VE'] # excitatory reversal potential
        self.VI = config['VI'] # inhibitory reversal potential
        self.Vth = config['Vth'] # firing threshold
        self.Vres = config['Vres'] # reset potential
        self.Tref = config['Tref'] # ms; NB inhibitory neuron has different value
        self.L = 1/self.tau_L
        
        # unused parameters, for now
        self.sE = 1 #excitatory conductance; microsiemens
        self.sI = 1 #inhibitory conductance; microsiemens
        self.C = 1 # conductance nF
        
    def forward(self, v, tref, is_spike, je, ji, jx, ext_input, inh_input, x_input):
        #compute synaptic activation
        if self.tau_E>0:
            je += -je/self.tau_E*self.dt + ext_input
        else:
            je = ext_input
        
        if self.tau_I>0:
            ji += -ji/self.tau_I*self.dt + inh_input
        else:
            ji = inh_input

        if self.tau_x>0:
            jx += -jx/self.tau_x*self.dt + x_input
        else:
            jx = x_input

        #compute currents
        leak_curr = self.L*(self.VL-v)
        ext_curr = self.L*self.sE*(self.VE-v)*je
        inh_curr = self.L*self.sI*(self.VI-v)*ji
        x_curr = self.L*jx
        
        #compute voltage
        # TODO: warning: line below doesn't work with instant synapse
        v += (leak_curr+ext_curr+inh_curr+x_curr)*self.dt
        
        #compute spikes
        is_ref = (tref > 0.0) & (tref < self.Tref)
        is_spike = (v > self.Vth) & ~is_ref
        is_sub = ~(is_ref | is_spike)
                
        v[is_spike] = self.Vspk
        v[is_ref] = self.Vres
        
        #update refractory period timer
        tref[is_sub] = 0.0
        tref[is_ref | is_spike] += self.dt
        return v, tref, is_spike, je, ji
        
    
    
    def run(self, T, device = 'cpu', record_interval = None, record_v = False, show_message = False):
        '''Simulate integrate and fire neurons
        T = simulation duration (ms)        
        '''
        
        self.T = T #min(10e3, 100/maf_u) #T = desired number of spikes / mean firing rate
        num_timesteps = int(self.T/self.dt)
        delay_steps = int(self.delay/self.dt)+1 # works when delay is zero

        tref = torch.zeros(self.batchsize, self.num_neurons, device=device) #tracker for refractory period
        je = torch.zeros(self.batchsize, self.num_neurons, device=device)
        ji = torch.zeros(self.batchsize, self.num_neurons, device=device)
        jx = torch.zeros(self.batchsize, self.num_neurons, device=device)
        v = self.Vres + torch.rand(self.batchsize, self.num_neurons, device=device)*(self.Vth-self.Vres) #initial voltage
        #v = torch.zeros(self.batchsize, self.num_neurons, device=device)
        is_spike = torch.zeros(self.batchsize, self.num_neurons, device=device)
        spk_count = torch.zeros(self.batchsize, self.num_neurons, device=device)
        #SpkTime = [[] for i in range(self.num_neurons)]
        #spk_history = np.empty((0,3),dtype=np.uint32) # sample_id x neuron_id x time
        #spk_history = np.empty((self.max_num_spks,3),dtype=np.uint32) # sample_id x neuron_id x time, pre-allocate memory
        # don't record spike history
        
        t = np.arange(0, self.T , self.dt)
        
        if record_v:
            V = torch.zeros( self.num_neurons, num_timesteps , device = 'cpu') #probably out of memory on gpu
        else:            
            V = None
        
        if record_interval:
            num_time_pts = int(self.T/record_interval)
            spk_count_history = torch.zeros( self.batchsize, self.num_neurons, num_time_pts,  device = 'cpu')
            record_counter=0
        else:
            spk_count_history = None
        #v, tref, is_spike
        cache_spk = torch.zeros(self.batchsize, self.num_neurons, delay_steps, device = device)
        
        read_pointer = -1
        write_pointer = 0
        
        #total_num_spks = 0 # track total number of spikes

        start_time = time.time()
        for i in range(num_timesteps):
            
            # read oldest cached data
            spk_delayed = cache_spk[:,:,read_pointer]
            # write current state to cache
            cache_spk[:,:,write_pointer] = is_spike

            #advance cached time by 1 step
            read_pointer = np.mod(read_pointer-1,delay_steps)
            write_pointer = np.mod(write_pointer-1, delay_steps)
            
            exc_input = self.exc_input_gen(self.dt, device=device)
            inh_input = self.inh_input_gen(self.dt, device=device)
            x_input = self.x_input_gen(self.dt, device=device) # temporary test
            
            with torch.no_grad():
                
                v, tref, is_spike, je, ji = self.forward(v, tref, spk_delayed, je,ji,jx, exc_input, inh_input, x_input)
            
                if record_v:
                    V[:,i] = v[0,:].flatten() #saves only 1 sample from the batch to limit memory consumption

                if i> int(self.discard/self.dt): #discard first 100 ms
                    spk_count += is_spike     
            #    spk_indices = torch.nonzero(is_spike).to('cpu').numpy().astype(np.uint32) #each row is a 2-tuple (sample_id, neuron_id)
            
            if record_interval: # record spike history
                if  i % int(record_interval/self.dt) == 0:
                    #print('Recording spike count history at step ',i)
                    spk_count_history[:,:,record_counter] = spk_count
                    record_counter+=1

            if show_message and (i+1) % int(num_timesteps/10) == 0:
                progress = (i+1)/num_timesteps*100
                elapsed_time = (time.time()-start_time)/60
                print('Progress: {:.2f}%; Time elapsed: {:.2f} min'.format(progress, elapsed_time ), flush=True)
        
        #spk_history = spk_history[:total_num_spks,:] # discard data in pre-allocated but unused memory

        return spk_count, V, t, spk_count_history

