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
from mnn.utils.dataloaders.mnist_loader import classic_mnist_loader
from torchvision import transforms
from projects.crit.mnn2snn import MNN2SNN

# def update_config_snn(config): #update config file for snn
#     ''' Use base configs from MNN and add SNN specific configs'''
#     # these are default values don't modify it here!
#     config_snn = {
#     'dT': 100, #ms spike count time window (0.1 ms in Brunel 2000)
#     'dt_snn': 0.01, # ms, snn integration time step
#     'T_snn':int(1e3), #snn simulation duration
#     'delay_snn': 1.5, # (1.5 ms in Brunel 2000)
#     'batchsize':10, # parallelize multiple trials with minibatch to shorten simulation time
#     }

#     config.update(config_snn)

#     return config

class SNNInputGenerator():
    def __init__(self, config):
        self.NE = config['NE']
        self.NI = config['NI']
        self.N = config['NE']+config['NI']
        self.bg_rate = config['bg_rate']
        self.w_bg = config['w_bg']
        self.batchsize = config['batchsize']
        self.dt = config['dt_snn']
        self.trans_factor = 1.0 # stimulus transduction factor
        # TODO: include trans_factor into trainer_config so it's consistent between mnn and snn
        
        # prepare dataloader
        # TODO: alternatively don't use dataloader, but dataset instead so I can control which samples to load?
        batch_size = self.batchsize
        train_dataloader, validation_dataloader = classic_mnist_loader(data_dir = './datasets/', train_batch=batch_size, test_batch=batch_size,  \
        transform_train = transforms.Compose([transforms.RandomCrop(28, padding=2), transforms.ToTensor()]), \
        transform_test = transforms.Compose([transforms.ToTensor()])   )

        self.dataloader = train_dataloader

        if self.bg_rate*self.dt >1:
            print('Warning: insufficient temporal resolution!')
        
        return
    
    def ind_poisson(self, device = 'cpu'):
        #dt = simulation time step (ms)
        p = self.bg_rate*self.dt # requires p<1        
        bg_current = self.w_bg*(torch.rand(self.batchsize, self.N, device=device) < p)
        return bg_current
    
    def load_data(self, device='cpu'):
        '''Load data using pytorch dataloader convention'''
        data, target = next(iter(self.dataloader)) # return the first batch of data
        input_samples = data.view(data.shape[0], -1) #flatten the image
        #for i_batch, (data, target) in enumerate(self.dataloader):
        #    input_samples = data.view(data.shape[0], -1) #flatten the image
        #    break        
        return input_samples.to(device), target.to(device)

    def spike_encoder(self, input_samples, device='cpu'):
        ''' stimulus transduction - encode input data to spikes'''

        p = input_samples*self.trans_factor*self.dt # requires p<1
        input_spikes = torch.rand( input_samples.shape, device=device) < p
        return input_spikes+0.0 #turn to float (default precision)
    

class InteNFireRNNtask():
    def __init__(self, config, WrecT, WffT, WoutT, input_gen):
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
        
        self.WrecT = WrecT #transpose of recurrent synaptic weight matrix
        self.WffT = WffT #transpose of feedforward_rec synaptic weight matrix
        self.WoutT = WoutT #transpose of readout weight matrix
        
        self.input_gen = input_gen # input generator, class object
        
        
    def forward_rec(self, v, tref, is_spike, ff_current):
        #compute voltage
        v += -v*self.L*self.dt + torch.matmul(is_spike, self.WrecT) + ff_current
        
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
    
    def forward_ff(self, input_spikes, device='cpu'):
        bg_current = self.input_gen.ind_poisson(device=device)
        ff_current = torch.matmul( input_spikes, self.WffT) + bg_current
        return ff_current

    # TODO: below is to be implemented
    def readout(self, cum_spk_count, target):

        output = torch.matmul(cum_spk_count/self.T, self.WoutT)
        _, predicted = torch.max(output, 1)
        correct_predictions = (predicted == target).sum().item()
        accuracy = correct_predictions/target.shape[0]

        return output, accuracy 
    
    def run(self, T, device = 'cpu', ntrials = None, record_v = False, show_message = False):
        '''Simulate integrate and fire neurons
        T = simulation duration (ms)        
        '''
        
        self.T = T #min(10e3, 100/maf_u) #T = desired number of spikes / mean firing rate
        num_timesteps = int(self.T/self.dt)
        delay_steps = int(self.delay/self.dt)+1 # works when delay is zero

        self.max_num_spks = int(0.15*self.batchsize*self.num_neurons*T)  # stop early if spikes exceed a limit
        
        tref = torch.zeros(self.batchsize, self.num_neurons, device=device) #tracker for refractory period
        #v = np.random.rand(self.num_neurons,1)*self.Vth #initial voltage
        v = torch.zeros(self.batchsize, self.num_neurons, device=device)
        is_spike = torch.zeros(self.batchsize, self.num_neurons, device=device)
        cum_spk_count = torch.zeros(self.batchsize, self.num_neurons, device=device)
        
        #SpkTime = [[] for i in range(self.num_neurons)]
        #spk_history = np.empty((0,3),dtype=np.uint32) # sample_id x neuron_id x time
        spk_history = np.empty((self.max_num_spks,3),dtype=np.uint32) # sample_id x neuron_id x time, pre-allocate memory
        acc_history = torch.zeros(num_timesteps)

        t = np.arange(0, self.T , self.dt)
        
        if record_v:
            V = torch.zeros( self.num_neurons, num_timesteps , device = 'cpu') #probably out of memory on gpu
        else:            
            V = None
        
        #v, tref, is_spike
        cache_spk = torch.zeros(self.batchsize, self.num_neurons, delay_steps, device = device)
        
        read_pointer = -1
        write_pointer = 0
        
        total_num_spks = 0 # track total number of spikes

        input_samples, target = self.input_gen.load_data(device=device) #load the first n samples

        start_time = time.time()
        for i in range(num_timesteps):
            
            # read oldest cached data
            spk_delayed = cache_spk[:,:,read_pointer]
            # write current state to cache
            cache_spk[:,:,write_pointer] = is_spike

            #advance cached time by 1 step
            read_pointer = np.mod(read_pointer-1,delay_steps)
            write_pointer = np.mod(write_pointer-1, delay_steps)
            
            with torch.no_grad():
                input_spikes = self.input_gen.spike_encoder( input_samples, device=device) 
                input_current = self.forward_ff(input_spikes, device=device)                
                v, tref, is_spike = self.forward_rec(v, tref, spk_delayed, input_current)
            
                if record_v:
                    V[:,i] = v[0,:].flatten() #saves only 1 sample from the batch to limit memory consumption
                
                spk_indices = torch.nonzero(is_spike).to('cpu').numpy().astype(np.uint32) #each row is a 2-tuple (sample_id, neuron_id)

                cum_spk_count += is_spike

                output, accuracy = self.readout(cum_spk_count, target)
                acc_history[i] = accuracy
            
            if total_num_spks+spk_indices.shape[0] > self.max_num_spks:
                print('Total number of spikes have exceeded pre-allocated limit!')  # stop early if spikes exceed a limit
                print('Existing simulation...')
                break

            spk_history[total_num_spks:total_num_spks+spk_indices.shape[0], :2] = spk_indices #update sample id and neuron id
            spk_history[total_num_spks:total_num_spks+spk_indices.shape[0], 2] = i #update spike time
            
            total_num_spks += spk_indices.shape[0] # update total number of spikes
        

            if show_message and (i+1) % int(num_timesteps/10) == 0:
                # TODO: store accuracy, since we are interested in convergence rate
                print('Classification accuracy for {} samples: {}'.format(output.shape[0],accuracy))

                progress = (i+1)/num_timesteps*100
                elapsed_time = (time.time()-start_time)/60
                print('Progress: {:.2f}%; Time elapsed: {:.2f} min'.format(progress, elapsed_time ), flush=True)
        
        spk_history = spk_history[:total_num_spks,:] # discard data in pre-allocated but unused memory        

        return spk_history, V, t, acc_history

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
    torch.set_default_dtype(torch.float32)
    
    #exp_id = 'test_rec_snn'
    
    device = 'cuda'

    print('Loading trained MNN and generate SNN parameters...')
    m2s = MNN2SNN('mnn_vary_ie_ratio_5trials_fix_validation')
    print(m2s.meta_data)
    data_indx = 12 # range ie_ratio from 2 to 8 with 0.5 increment, so indx=6 would be 5.0, indx=12 would be 8.0 
    trainer_config, model_state_dict = m2s.load_result(data_indx)
    trainer_config = m2s.update_config_snn(trainer_config)
    config = trainer_config['hidden_layer_config']
    config['batchsize']=1000
    config['T_snn'] = 10 # ms; we only need a short period of simulation
    
    print(config)

    Wff, current_ff, Wrec, Wout  = m2s.merge_bn_param(model_state_dict)
    # TODO: replace following temporary place holder with saved weight matrix consistent with MNN
    Wrec = gen_synaptic_weight(config)    

    print('Preparing dataloader...')
    input_gen = SNNInputGenerator(config)

    print('Running SNN simulation...')
    snn_model = InteNFireRNNtask(config, Wrec.T, Wff.T, Wout.T, input_gen)
    spk_history, V, t, acc_history = snn_model.run( config['T_snn'] , show_message=True, device = device) # ms

    print('Plotting results...')
    nsteps = int(config['T_snn']/config['dt_snn'])
    nneurons = config['NE']+config['NI']    
    sparse_matrix = spk_time2csr(spk_history, nneurons, nsteps, sample_id = 0)
    
    spk_count = np.asarray(np.sum(sparse_matrix, axis=0))
    binwidth = 10 # dt = 0.01 ms; binwidth of 10 is equal to 0.1 ms window
    spk_count = spk_count.reshape( int(spk_count.shape[1]/binwidth) , binwidth ).sum(axis=1)
    
    pop_firing_rate = spk_count/nneurons/0.1*1e3
    print('Avg pop firing rate (sp/s)', pop_firing_rate.mean())

    tt= np.linspace(0,t[-1], spk_count.shape[0])    
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.bar(tt , pop_firing_rate, width = tt[1]-tt[0] )
    plt.xlim([t[0],t[-1]])
    plt.tight_layout()
    plt.xlabel('time (ms)')
    plt.ylabel('Pop firing rate (sp/s)')
    
    plt.subplot(2,1,2)
    plt.plot(t[::10], acc_history[::10])
    plt.xlabel('Time (ms)')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('tmp_snn_pop_spike_count.png')
    plt.close('all')


