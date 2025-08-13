# -*- coding: utf-8 -*-
"""
Analysis of noise correlation in two-neuron recurrent spiking network.
@author: Yang Qi
"""

import numpy as np
import torch
import scipy as sp
import matplotlib.pyplot as plt
import time, os
from mnn.mnn_core.mnn_utils import Param_Container, Mnn_Core_Func


def gen_config(N=2, w=0.1, mean_ext=1, var_ext=1, device='cpu'): #generate config file
    ''' Default parameter values. Recommended not to change anything here, but update it per application.'''
    config = {
    'L':1/20, # leak conductance
    'Vth': 20, #mV, firing threshold, default 20
    'Vres': 0, #mV reset potential; default 0
    'Tref': 5, #ms, refractory period, default 5
    'N': N, # number of neurons
    'w': w, # recurrent weight
    'mean_ext': mean_ext, # external current mean
    'var_ext': var_ext, # external current var    
    'randseed':None,
    'dt_snn':0.01, # integration time step (ms)
    'T_snn':1000, #simualtion duration (ms)
    'record_ts':False,
    'device': device,
    'dT': 1000, #ms spike count time window
    'batchsize':1000, # multiple trials for calculating spike count stats
    }
    return config

def gen_synaptic_weight(config):
    W = torch.ones(config['N'],config['N'])*config['w']
    # Remove diagonal (self-connection)
    W.fill_diagonal_(0)
    return W.to(config['device'])


class InputGenerator():
    def __init__(self, config):
        self.N = config['N']
        self.mean_ext = config['mean_ext']
        self.var_ext = config['var_ext']
        self.dt = config['dt_snn']
        self.device=config['device']
        self.batchsize = config['batchsize']
        self.L = config['L']
        return

    def uncorr_gaussian_noise(self):
        # input argument dimensions: batchsize x #neurons
        #a = np.exp(-self.L*self.dt)
        #input_current = self.mean_ext*(1-a)/self.L + np.sqrt(self.var_ext*self.dt)*torch.randn(self.batchsize,self.N, device=self.device)        
        input_current = self.mean_ext*self.dt + np.sqrt(self.var_ext*self.dt)*torch.randn(self.batchsize,self.N, device=self.device)        
        return input_current

class InteNFireRNN():
    def __init__(self, config, W, input_gen):
        ''' W = weight matrix, input_gen = generator for external input'''
        self.L = config['L'] #ms
        self.Vth = config['Vth']
        self.Vres = config['Vres']   
        self.Tref = config['Tref'] #ms
        self.Vspk = 50 #for visualization purposes only
        self.dt = config['dt_snn'] #integration time step (ms)
        self.num_neurons = config['N']      
        self.batchsize = config['batchsize']
        self.W = W #transpose of synaptic weight matrix
        self.input_gen = input_gen # input generator, class object    
        self.device=config['device']    
        
    def forward(self, v, tref, is_spike, ff_current):
        #compute voltage
        v += -v*self.L*self.dt + torch.mm(is_spike.to(self.W.dtype), self.W.T) + ff_current
        #v = v*np.exp(-self.L*self.dt) + torch.mm(is_spike.to(self.W.dtype), self.W.T) + ff_current
        
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
        
    
    
    def run(self, T, device = 'cpu', record_v = False, show_message = False):
        '''Simulate integrate and fire neurons
        T = simulation duration (ms)        
        '''
        device=self.device
        self.T = T
        num_timesteps = int(self.T/self.dt)
        
        #self.max_num_spks = int(0.15*self.batchsize*self.num_neurons*T)  # stop early if spikes exceed a limit
        
        tref = torch.zeros(self.batchsize, self.num_neurons, device=device) #tracker for refractory period
        v = torch.rand(self.batchsize, self.num_neurons, device=device)*self.Vth #initial voltage
        is_spike = torch.zeros(self.batchsize, self.num_neurons, device=device)
        
        #spk_history = np.empty((self.max_num_spks,3),dtype=np.uint32) # sample_id x neuron_id x time, pre-allocate memory
        
        t = np.arange(0, self.T , self.dt)
        
        if record_v:
            V = torch.zeros( self.num_neurons, num_timesteps , device = 'cpu') #probably out of memory on gpu
        else:            
            V = None
        
        spk_count = torch.zeros(self.batchsize, self.num_neurons, device=device) # track total number of spikes

        start_time = time.time()
        for i in range(num_timesteps):
            
            input_current = self.input_gen.uncorr_gaussian_noise()
            
            with torch.no_grad():
                
                v, tref, is_spike = self.forward(v, tref, is_spike, input_current)
            
                if record_v:
                    V[:,i] = v[0,:].flatten() #saves only 1 sample from the batch to limit memory consumption
                
                #spk_indices = torch.nonzero(is_spike).to('cpu').numpy().astype(np.uint32) #each row is a 2-tuple (sample_id, neuron_id)
            spk_count += is_spike 
            
            # if total_num_spks+spk_indices.shape[0] > self.max_num_spks:
            #     print('Total number of spikes have exceeded pre-allocated limit!')  # stop early if spikes exceed a limit
            #     print('Existing simulation...')
            #     break
            #spk_history[total_num_spks:total_num_spks+spk_indices.shape[0], :2] = spk_indices #update sample id and neuron id
            #spk_history[total_num_spks:total_num_spks+spk_indices.shape[0], 2] = i #update spike time
            #total_num_spks += spk_indices.shape[0] # update total number of spikes

            if show_message and (i+1) % int(num_timesteps/10) == 0:
                progress = (i+1)/num_timesteps*100
                elapsed_time = (time.time()-start_time)/60
                print('Progress: {:.2f}%; Time elapsed: {:.2f} min'.format(progress, elapsed_time ), flush=True)
        
        #spk_history = spk_history[:total_num_spks,:] # discard data in pre-allocated but unused memory

        return spk_count, V, t



def get_stats_snn(mean_ext,var_ext,w, T_snn=1e3, batchsize=int(100e3)):    
    
    config = gen_config(N=2, w=w, mean_ext=mean_ext, var_ext=var_ext, device='cuda')
    config['T_snn']=T_snn
    config['batchsize']= batchsize

    W = gen_synaptic_weight(config) #doesn't take too much time with 1e4 neurons    
    input_gen = InputGenerator(config)
    snn_model = InteNFireRNN(config, W.T , input_gen)
    spk_count, V, t = snn_model.run( config['T_snn'] , show_message=False, record_v = False) # ms

    mean_firing_rate = torch.mean(spk_count, dim=0)/config['T_snn']
    fano_factor = torch.var(spk_count,dim=0)/torch.mean(spk_count,dim=0)
    corr_coef = torch.corrcoef(spk_count.T)[0, 1]

    print('Mean firing rate (sp/ms):', mean_firing_rate.cpu().numpy())
    print('Fano factor:', fano_factor.cpu().numpy())
    print('Correlation coefficient:', corr_coef.cpu().numpy())

    return mean_firing_rate.cpu().numpy(), fano_factor.cpu().numpy(), corr_coef.cpu().numpy(), config

def get_stats_mnn(mean_ext,var_ext,w):
    ma = Mnn_Core_Func()
    result = sp.optimize.root( lambda xy: fixed_pt_eq(xy, ma, w, mean_ext, var_ext), (0.01,0.1))
    mnn_mean, mnn_var = result.x
    
    curr_mean = w*mnn_mean + mean_ext
    curr_std = np.sqrt(w*w*mnn_var+var_ext)
    temp = ma.forward_fast_chi(curr_mean, curr_std, mnn_mean, np.sqrt(mnn_var))
    lin_res_coef = temp*np.sqrt(mnn_var)/(curr_std+1e-16)
    mnn_corr = w*lin_res_coef

    print('mnn mean firing rate: ', mnn_mean)
    print('mnn fano factor: ', mnn_var/mnn_mean)
    print('mnn corr coef: ', mnn_corr)
    return mnn_mean, mnn_var/mnn_mean, mnn_corr

def para_sweep_stats_vs_w(exp_id='para_sweep_stats_vs_w', mean_ext=0.5, var_ext=9):

    m = 11
    w_array = np.linspace(-5,5,m)

    M = np.zeros((m,2))
    FF = np.zeros((m,2))
    R = np.zeros((m,2))

    for i in range(m):
        print('Running... {}/{}'.format( i+1,m))
        snn_mean, snn_ff, snn_corr_coef, config = get_stats_snn(mean_ext,var_ext,w_array[i])
        M[i,0] = snn_mean.mean()
        FF[i,0] = snn_ff.mean()
        R[i,0] = snn_corr_coef

        mnn_mean, mnn_ff, mnn_corr = get_stats_mnn(mean_ext,var_ext,w_array[i])
        M[i,1] = mnn_mean
        FF[i,1] = mnn_ff
        R[i,1] = mnn_corr
    
    path = './projects/rmnn/runs/{}/'.format(exp_id)
    os.makedirs(path, exist_ok=True)
    filename = 'mean_ext_{}_var_ext_{}.npz'.format(mean_ext,var_ext)
    np.savez(path+filename, M=M, FF=FF, R=R, w_array=w_array, mean_ext=mean_ext, var_ext=var_ext)
    return

def plot_stats_vs_w(exp_id='para_sweep_stats_vs_w', mean_ext=0.5, var_ext=9):
    path = './projects/rmnn/runs/{}/'.format(exp_id)
    filename = 'mean_ext_{}_var_ext_{}.npz'.format(mean_ext,var_ext)
    dat = np.load(path+filename)
    w_array = dat['w_array']
    M, FF, R = dat['M'], dat['FF'], dat['R']
    plt.figure(figsize=(3.5*3,3.5))
    plt.suptitle('mean_ext={}, var_ext={}'.format(mean_ext,var_ext), fontsize=12, y=0.96)
    plt.subplot(1,3,1)
    plt.plot(w_array, M[:,1], color='gray')
    plt.plot(w_array, M[:,0],'.k')
    plt.xlabel('w')
    plt.ylabel('Mean firing rate (sp/ms)')
    plt.legend(['MNN','SNN'])
    plt.subplot(1,3,2)
    plt.plot(w_array,FF[:,1], color='gray')
    plt.plot(w_array,FF[:,0],'.k')    
    plt.xlabel('w')
    plt.ylabel('Fano factor')
    plt.subplot(1,3,3)
    plt.plot(w_array,R[:,1]*2, color='gray')
    plt.plot(w_array,R[:,0],'.k')
    plt.xlabel('w')
    plt.ylabel('Corr. coef.')
    plt.tight_layout()
    plt.savefig(path+'mean_ext_{}_var_ext_{}.pdf'.format(mean_ext,var_ext))

def fixed_pt_eq(xy, ma, w, mean_ext, var_ext):
    curr_mean = w*xy[0] + mean_ext
    curr_std = np.sqrt(w*w*xy[1]+var_ext)
    mean_out = ma.forward_fast_mean(curr_mean, curr_std)
    std_out = ma.forward_fast_std(curr_mean,curr_std,mean_out) #<-- probably in correct.
    var_out = std_out**2
    return -xy[0]+mean_out, -xy[1]+var_out


def para_sweep_w_mean_var(exp_id='para_sweep_w_mean_var'):
    
    mean_array = np.linspace(-0.5,2,6)
    var_array = np.linspace(0.5,3,6)**2
    w_array = np.linspace(-5,5,11)

    arr_shape = (len(mean_array), len(var_array),len(w_array))
    m = np.prod(arr_shape)
    
    M = np.zeros((m,2))
    FF = np.zeros((m,2))
    R = np.zeros((m,2))

    for indx in range(m):
        print('Running... {}/{}'.format( indx+1,m))

        i,j,k = np.unravel_index(indx, arr_shape)

        snn_mean, snn_ff, snn_corr_coef, config = get_stats_snn(mean_array[i],var_array[j],w_array[k], T_snn=10e3)
        M[indx,0] = snn_mean.mean()
        FF[indx,0] = snn_ff.mean()
        R[indx,0] = snn_corr_coef

        mnn_mean, mnn_ff, mnn_corr = get_stats_mnn(mean_array[i],var_array[j],w_array[k])
        M[indx,1] = mnn_mean
        FF[indx,1] = mnn_ff
        R[indx,1] = mnn_corr
    
    path = './projects/rmnn/runs/{}/'.format(exp_id)
    os.makedirs(path, exist_ok=True)
    filename = exp_id+'.npz'
    np.savez(path+filename, M=M, FF=FF, R=R, w_array=w_array, mean_array=mean_array, var_array=var_array, config=config)
    return

def plot_stats_w_mean_var(exp_id='para_sweep_w_mean_var'):
    path = './projects/rmnn/runs/{}/'.format(exp_id)
    filename = 'para_sweep_w_mean_var.npz'
    dat = np.load(path+filename)
    print(list(dat)) #>> ['M', 'FF', 'R', 'w_array', 'mean_array', 'var_array']
    mean_array = dat['mean_array']
    var_array = dat['var_array']
    w_array = dat['w_array']
    M, FF, R = dat['M'], dat['FF'], dat['R']
    
    shape = (len(mean_array), len(var_array), len(w_array), 2)
    M = M.reshape(shape)
    FF=FF.reshape(shape)
    R=R.reshape(shape)

    # plot mean
    panel = 0
    plt.figure(figsize=(12,10))
    plt.suptitle('Mean firing rate (sp/s)', fontsize=18, y=0.99)
    for i in range(shape[0]):
        for j in range(shape[1]):
            panel+=1
            plt.subplot(shape[0],shape[1], panel)
            plt.plot(w_array, 1e3*M[i,j,:,1], color='gray')
            plt.plot(w_array, 1e3*M[i,j,:,0], '.k')
            if i==5:
                plt.xlabel('w')
            if i==0:
                plt.title('ext_var={}'.format(var_array[j]))
            if j==0:
                plt.ylabel('ext_mean={}'.format(mean_array[i]))
    plt.tight_layout()
    plt.savefig(path+'mean_firing_rate_w_mean_var.pdf')

    panel = 0
    plt.figure(figsize=(12,10))
    plt.suptitle('Fano factor', fontsize=18, y=0.99)
    for i in range(shape[0]):
        for j in range(shape[1]):
            panel+=1
            plt.subplot(shape[0],shape[1], panel)
            plt.plot(w_array, FF[i,j,:,1], color='gray')
            plt.plot(w_array, FF[i,j,:,0], '.k')
            if i==5:
                plt.xlabel('w')
            if i==0:
                plt.title('ext_var={}'.format(var_array[j]))
            if j==0:
                plt.ylabel('ext_mean={}'.format(mean_array[i]))
    plt.tight_layout()
    plt.savefig(path+'fano_factor_w_mean_var.pdf')
            

    panel = 0
    plt.figure(figsize=(12,10))
    plt.suptitle('Correlation coefficient', fontsize=18, y=0.99)
    for i in range(shape[0]):
        for j in range(shape[1]):
            panel+=1
            plt.subplot(shape[0],shape[1], panel)
            plt.plot(w_array, 2*R[i,j,:,1], color='gray')
            plt.plot(w_array, R[i,j,:,0], '.k')
            if i==5:
                plt.xlabel('w')
            if i==0:
                plt.title('ext_var={}'.format(var_array[j]))
            if j==0:
                plt.ylabel('ext_mean={}'.format(mean_array[i]))
    plt.tight_layout()
    plt.savefig(path+'corr_coef_w_mean_var.pdf')

    return

if __name__=='__main__':
    torch.set_default_dtype(torch.float64)
    #para_sweep_stats_vs_w(mean_ext=2, var_ext=9)
    #plot_stats_vs_w(mean_ext=2, var_ext=9)
    
    para_sweep_w_mean_var(exp_id='para_sweep_w_mean_var_longer_time')
    plot_stats_w_mean_var(exp_id='para_sweep_w_mean_var_longer_time')

    # to run the script in background and print to log file: -u for instant flush
    #nohup python -u -m projects.rmnn.two_neuron_recurrent > output.log 2>&1 &

    

    