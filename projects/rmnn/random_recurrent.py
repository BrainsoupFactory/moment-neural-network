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
    'T_mnn': 20,
    'dt_mnn': 0.2,
    'p': None, # probability of connection
    }

    return config

def gen_synaptic_weight(config, type='random'):
    if type == 'random':
        W = torch.randn(config['N'],config['N'], device=config['device'])*config['w']
    elif type == 'fixed':
        W = torch.ones(config['N'],config['N'], device=config['device'])*config['w']
    
    W.fill_diagonal_(0)
    
    if config['p'] is not None:
        W = W/np.sqrt(W.shape[0]*config['p']) # normalize by avg indegree
        W[torch.rand(W.shape)>config['p']]=0
    else:
        W = W/np.sqrt(W.shape[0]) # normalize by # of neurons

    return W


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
        self.W = W #synaptic weight matrix
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
        
    
    
    def run(self, T, record_v = False, show_message = False):
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

class RecurrentMNN():
    def __init__(self, config, W):
        self.N = config['N']
        self.dt = config['dt_mnn']
        self.tau = 1
        self.W = W #synaptic weight matrix (csr)
        #self.input_gen = input_gen # input generator, class object
        self.ma = Mnn_Core_Func()
        self.T = config['T_mnn']
        self.record_ts = config['record_ts']
        
        
    def run(self, ff_mean, ff_var):
        '''# 
        Custom forward pass. Inputs are mean/std of external input currents
        '''
        self.nsteps = int(self.T/self.dt)
        self.batchsize = ff_mean.shape[0]
        #ff_mean = ff_mean#.unsqueeze(-1)
        #ff_var = ff_var#.unsqueeze(-1)

        # initial condition
        u = torch.zeros(self.batchsize,self.N, device=ff_mean.device) #just 1D array, no column/row 
        c = torch.eye(self.N, device=ff_mean.device).unsqueeze(0).expand(self.batchsize, -1, -1)
        
        if self.record_ts: # cached data for synaptic delay
            U = torch.zeros(self.batchsize, self.N, self.nsteps, device='cpu')
            C = torch.zeros(self.batchsize, self.N, self.N, self.nsteps, device='cpu')
        #     Ubar = U.clone()
        #     Sbar = S.clone()
    
        a = self.dt/self.tau
        
        for i in range(self.nsteps):
            if i % int(self.nsteps/10) == 0:
                print('Running {}/{}...'.format(i,self.nsteps))
            
            curr_mean = torch.mm(u, self.W.T) + ff_mean
            curr_var = torch.einsum('ni,bij,nj->bn', self.W, c, self.W) + ff_var
            curr_std = torch.sqrt( curr_var )

            # numpy 
            curr_mean = curr_mean.cpu().numpy()
            curr_std = curr_std.cpu().numpy()
            ma_mean = self.ma.forward_fast_mean(curr_mean, curr_std)
            ma_std = self.ma.forward_fast_std(curr_mean, curr_std, ma_mean)
            ma_chi = self.ma.forward_fast_chi(curr_mean, curr_std, ma_mean, ma_std)
            lrc = ma_chi*ma_std/(curr_std+1e-16) #linear response
            
            ma_mean = torch.tensor(ma_mean).to('cuda')
            ma_std = torch.tensor(ma_std).to('cuda')
            lrc = torch.tensor(lrc).to('cuda')

            temp = lrc.unsqueeze(-1)*torch.matmul(self.W, c)
            ma_cov = temp+temp.transpose(1, 2)
            ma_cov = torch.diagonal_scatter(ma_cov, ma_std.pow(2.0), dim1=1, dim2=2) # set diagonal to exact solution

            # evolve one step in time
            u = (1-a)*u + a*ma_mean
            c = (1-a)*c + a*ma_cov

            if self.record_ts: #save time series data
                U[:,:,i] = u.cpu()
                C[:,:,:,i] = c.cpu()
            #     Ubar[:,:,i] = curr_mean.cpu()
            #     Sbar[:,:,i] = curr_std.cpu()

         # output the variance instead std
        if self.record_ts: # use this if loss takes in multiple time steps.
            return u, c, U, C
        else:
            return u, c
    


def rmnn_debug():
    config = gen_config(N=2, w=4, mean_ext=1, var_ext=1, device='cuda')
    config['record_ts']=True

    W = gen_synaptic_weight(config, type='fixed')

    rec_mnn = RecurrentMNN(config, W)
    
    # ext_mean = torch.linspace(-0.5,2,6,device=config['device']).unsqueeze(1)
    # ext_var = 9*torch.ones(6,device=config['device']).unsqueeze(1)

    ext_mean = torch.ones(1,device=config['device']).unsqueeze(1)
    ext_var = 9*torch.ones(1,device=config['device']).unsqueeze(1)
    
    t0=time.perf_counter()
    out_mean, out_cov, ts_mean, ts_cov = rec_mnn.run( ext_mean, ext_var )
    print('Time elapsed: ', int(time.perf_counter()-t0))

    out_var = out_cov.diagonal(dim1=1, dim2=2)
    out_FF = out_var/(1e-16+out_mean)
    out_std = torch.sqrt(out_var)
    out_corr = out_cov/out_std.unsqueeze(1)/out_std.unsqueeze(2)

    print(out_mean)
    print(out_FF)
    print(out_corr[:,0,1])

    t = np.linspace(0,config['T_mnn'],ts_mean.shape[-1])

    plt.figure(figsize=(3.5*3,3))
    plt.subplot(1,3,1)
    plt.plot(t,ts_mean[:,0,:].T)
    plt.ylabel('Mean firing rate (sp/ms)')
    plt.xlabel('Time (a.u.)')
    plt.subplot(1,3,2)
    ts_ff = ts_cov[:,0,0,:]/ts_mean[:,0,:]
    plt.plot(t,ts_ff.T)
    plt.ylim([0,1.5])
    plt.ylabel('Fano factor')
    plt.xlabel('Time (a.u.)')
    plt.subplot(1,3,3)
    ts_corr = ts_cov[:,0,1,:]/torch.sqrt(ts_cov[:,0,0,:])/torch.sqrt(ts_cov[:,1,1,:]) 
    plt.plot(t,ts_corr.T)
    plt.ylabel('Corr coef')
    plt.xlabel('Time (a.u.)')
    plt.tight_layout()
    plt.savefig('temp.png')

    path =  './projects/rmnn/runs/rmnn_debug/'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(path+'check_convergence.png')
    return
        

def get_stats_snn(config, W):    

    input_gen = InputGenerator(config)
    snn_model = InteNFireRNN(config, W , input_gen)
    spk_count, V, t = snn_model.run( config['T_snn'] , show_message=True, record_v = False) # ms

    mean_firing_rate = torch.mean(spk_count, dim=0)/config['T_snn']
    fano_factor = torch.var(spk_count,dim=0)/torch.mean(spk_count,dim=0)
    corr_coef = torch.corrcoef(spk_count.T)

    return mean_firing_rate.cpu().numpy(), fano_factor.cpu().numpy(), corr_coef.cpu().numpy()

def get_stats_mnn(config, W):
    rec_mnn = RecurrentMNN(config, W)
    ext_mean = config['mean_ext']*torch.ones(2,config['N'],device=config['device']) # keep redundant batch dimension to avoid bugs
    ext_var = config['var_ext']*torch.ones(2,config['N'],device=config['device'])
    
    out_mean, out_cov = rec_mnn.run( ext_mean, ext_var )
    #out_mean, out_cov = rec_mnn.run_experimental( ext_mean, ext_var )
    
    out_var = out_cov.diagonal(dim1=1, dim2=2)
    out_FF = out_var/(1e-16+out_mean)
    out_std = torch.sqrt(out_var)
    out_corr = out_cov/out_std.unsqueeze(1)/out_std.unsqueeze(2)

    return out_mean.cpu().numpy(), out_FF.cpu().numpy(), out_corr.cpu().numpy()

def rand_rmnn_debug(mean_ext=0.5, var_ext=9, flush_snn=True):
    exp_id = 'rand_rmnn_debug'
    num_neurons = 32
    m = 4
    w = 5
    config = gen_config(N=num_neurons, w=w, mean_ext=mean_ext, var_ext=var_ext, device='cuda')
    config['T_snn']=1e3
    config['batchsize']= int(100e3)

    path = './projects/rmnn/runs/{}/'.format(exp_id)
    os.makedirs(path, exist_ok=True)
    snn_save = 'snn_mean_ext_{}_var_ext_{}.npz'.format(mean_ext,var_ext)

    if flush_snn: # rerun SNN results
        W = gen_synaptic_weight(config, type='random')
        
        snn_mean, snn_ff, snn_corr_coef = get_stats_snn(config, W)
        np.savez(path+snn_save, snn_mean=snn_mean, snn_ff=snn_ff, snn_corr_coef=snn_corr_coef, w=w, mean_ext=mean_ext, var_ext=var_ext, W=W.cpu().numpy())
    else:
        dat = np.load(path+snn_save)
        snn_mean, snn_ff, snn_corr_coef = dat['snn_mean'], dat['snn_ff'], dat['snn_corr_coef']
        W = dat['W']
    
    W = torch.tensor(W, device=config['device'])
    mnn_mean, mnn_ff, mnn_corr = get_stats_mnn(config, W)   
    mnn_save = 'mnn_mean_ext_{}_var_ext_{}.npz'.format(mean_ext,var_ext)
    np.savez(path+mnn_save, mnn_mean=mnn_mean, mnn_ff=mnn_ff, mnn_corr=mnn_corr, w=w, mean_ext=mean_ext, var_ext=var_ext, W=W.cpu().numpy())
    
    print('Done!')

    # plot resutls
    plt.figure(figsize=(3.5*3,3.5))
    plt.suptitle('mean_ext={}, var_ext={}'.format(mean_ext,var_ext), fontsize=12, y=0.96)
    plt.subplot(1,3,1)
    plt.plot([mnn_mean.min(),mnn_mean.max()],[mnn_mean.min(),mnn_mean.max()],color='gray')
    plt.plot(mnn_mean[0,:], snn_mean,'.')
    plt.axis('equal')
    plt.xlabel('MNN')
    plt.ylabel('SNN')
    plt.title('Mean firing rate')
    #plt.legend(['w={}'.format(w_array[i]) for i in range(4)])
    plt.subplot(1,3,2)
    plt.plot([mnn_ff.min(),mnn_ff.max()],[mnn_ff.min(),mnn_ff.max()],color='gray')
    plt.plot(mnn_ff[0,:],snn_ff, '.')  
    plt.axis('equal')     
    plt.xlabel('MNN')
    plt.ylabel('SNN')
    plt.title('Fano factor')
    plt.subplot(1,3,3)
    indx=np.triu_indices(mnn_corr.shape[1],k=1)
    plt.plot([-0.2, 0.2],[-0.2, 0.2],color='gray')
    plt.plot(mnn_corr[0,:,:][indx], snn_corr_coef[indx], '.')       
    plt.axis('equal')
    plt.xlabel('MNN')
    plt.ylabel('SNN')
    plt.title('Corr. coef.')
    plt.tight_layout()
    plt.savefig(path+'mean_ext_{}_var_ext_{}.pdf'.format(mean_ext,var_ext))

    return

# def para_sweep_rand_net_w(exp_id='para_sweep_rand_net_w', mean_ext=0.5, var_ext=9):

#     num_neurons = 32
#     m = 11
#     w_array = np.linspace(0,10,m)

#     M = np.zeros((num_neurons,m,2))
#     FF = np.zeros((num_neurons,m,2))
#     R = np.zeros((num_neurons,num_neurons,m,2))

#     for i in range(m):
#         print('Running... {}/{}'.format( i+1,m))
#         config = gen_config(N=num_neurons, w=w_array[i], mean_ext=mean_ext, var_ext=var_ext, device='cuda')
#         config['T_snn']=1e3
#         config['batchsize']= int(100e3)

#         W = gen_synaptic_weight(config, type='random')
#         
#         snn_mean, snn_ff, snn_corr_coef = get_stats_snn(config, W)
#         M[:,i,0] = snn_mean
#         FF[:,i,0] = snn_ff
#         R[:,:,i,0] = snn_corr_coef

#         mnn_mean, mnn_ff, mnn_corr = get_stats_mnn(config, W)
#         M[:,i,1] = mnn_mean[0,:]
#         FF[:,i,1] = mnn_ff[0,:]
#         R[:,:,i,1] = mnn_corr[0,:,:]
    
#     path = './projects/rmnn/runs/{}/'.format(exp_id)
#     os.makedirs(path, exist_ok=True)
#     filename = 'mean_ext_{}_var_ext_{}_mk2.npz'.format(mean_ext,var_ext)
#     np.savez(path+filename, M=M, FF=FF, R=R, w_array=w_array, mean_ext=mean_ext, var_ext=var_ext, config=config)
#     return

# def plot_rand_net_w(exp_id='para_sweep_rand_net_w', mean_ext=0.5, var_ext=9):
#     path = './projects/rmnn/runs/{}/'.format(exp_id)
#     filename = 'mean_ext_{}_var_ext_{}_mk2.npz'.format(mean_ext,var_ext)
#     dat = np.load(path+filename)
#     w_array = dat['w_array']
#     M, FF, R = dat['M'], dat['FF'], dat['R']
#     plt.figure(figsize=(3.5*3,3.5))
#     plt.suptitle('mean_ext={}, var_ext={}'.format(mean_ext,var_ext), fontsize=12, y=0.96)
#     plt.subplot(1,3,1)
#     for i in range(len(w_array)):
#         plt.plot(M[:,i,1], M[:,i,0],'.')
#         if i>3: break
        
#     plt.xlabel('MNN')
#     plt.ylabel('SNN')
#     plt.title('Mean firing rate')
#     plt.legend(['w={}'.format(w_array[i]) for i in range(4)])
#     plt.subplot(1,3,2)
#     for i in range(len(w_array)):
#         plt.plot(FF[:,i,1],FF[:,i,0], '.')
#         if i>3: break
        
#     plt.xlabel('MNN')
#     plt.ylabel('SNN')
#     plt.title('Fano factor')
#     plt.subplot(1,3,3)
#     indx=np.triu_indices(R.shape[0],k=1)
#     for i in range(len(w_array)):
#         plt.plot(R[:,:,i,1][indx],R[:,:,i,0][indx], '.')
#         if i>3: break
        
#     plt.xlabel('MNN')
#     plt.ylabel('SNN')
#     plt.title('Corr. coef.')
#     plt.tight_layout()
#     plt.savefig(path+'mean_ext_{}_var_ext_{}_mk2.pdf'.format(mean_ext,var_ext))

# def fixed_pt_eq(xy, ma, w, mean_ext, var_ext):
#     curr_mean = w*xy[0] + mean_ext
#     curr_std = np.sqrt(w*w*xy[1]+var_ext)
#     mean_out = ma.forward_fast_mean(curr_mean, curr_std)
#     std_out = ma.forward_fast_std(curr_mean,curr_std,mean_out) #<-- probably in correct.
#     var_out = std_out**2
#     return -xy[0]+mean_out, -xy[1]+var_out


def para_sweep_rec_rand_mean_var(exp_id='para_sweep_rec_rand_mean_var'):
    
    mean_array = np.linspace(-0.5,2,6)
    var_array = np.linspace(0.5,3,6)**2
    w = 5
    num_neurons=32

    arr_shape = (len(mean_array), len(var_array))
    m = np.prod(arr_shape)
    
    M = np.zeros((m,num_neurons,2))
    FF = np.zeros((m,num_neurons,2))
    R = np.zeros((m,num_neurons,num_neurons,2))

    for indx in range(m):
        print('Running... {}/{}'.format( indx+1,m))

        i,j = np.unravel_index(indx, arr_shape)

        config = gen_config(N=num_neurons, w=w, mean_ext=mean_array[i], var_ext=var_array[j], device='cuda')
        config['T_snn']=10e3
        config['batchsize']= int(100e3)
        W = gen_synaptic_weight(config, type='random')

        snn_mean, snn_ff, snn_corr_coef = get_stats_snn(config, W)
        
        M[indx,:,0] = snn_mean
        FF[indx,:,0] = snn_ff
        R[indx,:,:,0] = snn_corr_coef

        mnn_mean, mnn_ff, mnn_corr = get_stats_mnn(config, W)
        M[indx,:,1] = mnn_mean[0,:]
        FF[indx,:,1] = mnn_ff[0,:]
        R[indx,:,:,1] = mnn_corr[0,:,:]
    
    path = './projects/rmnn/runs/{}/'.format(exp_id)
    os.makedirs(path, exist_ok=True)
    filename = exp_id+'.npz'
    np.savez(path+filename, M=M, FF=FF, R=R, w=w, mean_array=mean_array, var_array=var_array)
    return

def plot_rec_rand_mean_var(exp_id='para_sweep_rec_rand_mean_var'):
    path = './projects/rmnn/runs/{}/'.format(exp_id)
    filename = exp_id+'.npz'
    dat = np.load(path+filename)
    #print(list(dat)) >> ['M', 'FF', 'R', 'w', 'mean_array', 'var_array']
    num_neurons = 32

    mean_array = dat['mean_array']
    var_array = dat['var_array']
    #w_array = dat['w_array']
    M, FF, R = dat['M'], dat['FF'], dat['R']
    
    M = M.reshape(len(mean_array), len(var_array), num_neurons , 2)
    FF=FF.reshape(len(mean_array), len(var_array), num_neurons , 2)
    R=R.reshape(len(mean_array), len(var_array), num_neurons , num_neurons, 2)
    shape = (len(mean_array), len(var_array))

    # plot mean
    panel = 0
    plt.figure(figsize=(12,10))
    plt.suptitle('Mean firing rate (sp/s)', fontsize=18, y=0.99)
    for i in range(shape[0]):
        for j in range(shape[1]):
            panel+=1
            plt.subplot(shape[0],shape[1], panel)
            x=1e3*M[i,j,:,1]
            y=1e3*M[i,j,:,0]
            plt.plot(x,y , '.k')
            plt.plot([x.min(), x.max()],[x.min(), x.max()], color='gray')
            if y.max()<0.1: # firing rate is too low
                plt.ylim([-0.01,0.1])
            # if i==5:
            #     plt.xlabel('MNN')
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
            x = FF[i,j,:,1]
            y = FF[i,j,:,0]
            plt.plot(x,y , '.k')
            plt.plot([x.min(), x.max()],[x.min(), x.max()], color='gray')
            plt.axis('equal')
            # if i==5:
            #     plt.xlabel('MNN')
            if i==0:
                plt.title('ext_var={}'.format(var_array[j]))
            if j==0:
                plt.ylabel('ext_mean={}'.format(mean_array[i]))
    plt.tight_layout()
    plt.savefig(path+'fano_factor_w_mean_var.pdf')
            

    panel = 0
    idx,idy = np.triu_indices(num_neurons, k=1)
    subsamp = np.random.choice(len(idx), 100, replace=False)
    idx,idy = idx[subsamp],idy[subsamp]
    plt.figure(figsize=(12,10))
    plt.suptitle('Correlation coefficient', fontsize=18, y=0.99)
    for i in range(shape[0]):
        for j in range(shape[1]):
            panel+=1
            plt.subplot(shape[0],shape[1], panel)
            y = R[i,j,:,:,0][idx,idy]
            x = R[i,j,:,:,1][idx,idy]
            plt.plot(x,y, '.k')
            plt.plot([x.min(), x.max()],[x.min(), x.max()], color='gray')
            plt.axis('equal')
            # if i==5:
            #     plt.xlabel('w')
            if i==0:
                plt.title('ext_var={}'.format(var_array[j]))
            if j==0:
                plt.ylabel('ext_mean={}'.format(mean_array[i]))
    plt.tight_layout()
    plt.savefig(path+'corr_coef_w_mean_var.pdf')

    return

def para_sweep_rec_rand_p(exp_id='para_sweep_rec_rand_p'):
    
    #mean_array = np.linspace(-0.5,2,6)
    #var_array = np.linspace(0.5,3,6)**2
    p_array = np.linspace(0,1,11)
    mean_ext = 0.5
    var_ext = 9.0
    w = 10
    num_neurons=1000

    #arr_shape = (len(mean_array), len(var_array))
    #m = np.prod(arr_shape)
    m = len(p_array)

    M = np.zeros((m,num_neurons,2))
    FF = np.zeros((m,num_neurons,2))
    R = np.zeros((m,num_neurons,num_neurons,2))

    for indx in range(m):
        print('Running... {}/{}'.format( indx+1,m))

        #i,j = np.unravel_index(indx, arr_shape)

        config = gen_config(N=num_neurons, w=w, mean_ext=mean_ext, var_ext=var_ext, device='cuda')
        config['T_snn']=10e3
        config['batchsize']= int(10e3)
        config['p'] = p_array[indx]

        W = gen_synaptic_weight(config, type='random')

        snn_mean, snn_ff, snn_corr_coef = get_stats_snn(config, W)
        
        M[indx,:,0] = snn_mean
        FF[indx,:,0] = snn_ff
        R[indx,:,:,0] = snn_corr_coef

        mnn_mean, mnn_ff, mnn_corr = get_stats_mnn(config, W)
        M[indx,:,1] = mnn_mean[0,:]
        FF[indx,:,1] = mnn_ff[0,:]
        R[indx,:,:,1] = mnn_corr[0,:,:]
    
    path = './projects/rmnn/runs/{}/'.format(exp_id)
    os.makedirs(path, exist_ok=True)
    filename = 'rec_rand_p_n{}_w{}'.format(num_neurons,w)+'.npz'
    np.savez(path+filename, M=M, FF=FF, R=R, w=w, p_array=p_array, config=config)
    return

def plot_rec_rand_p(exp_id='para_sweep_rec_rand_p', num_neurons=100, w=5):
    path = './projects/rmnn/runs/{}/'.format(exp_id)
    filename = 'rec_rand_p_n{}_w{}'.format(num_neurons,w)+'.npz'
    dat = np.load(path+filename, allow_pickle=True)
    #print(list(dat)) # >> ['M', 'FF', 'R', 'w', 'p_array']
    
    #mean_array = dat['mean_array']
    #var_array = dat['var_array']
    #w_array = dat['w_array']
    p_array = dat['p_array']
    M, FF, R = dat['M'], dat['FF'], dat['R']

    m = len(p_array)
    
    path = './projects/rmnn/runs/{}/plots_n{}_w{}/'.format(exp_id,num_neurons,w)
    os.makedirs(path, exist_ok=True)

    # plot mean
    plt.figure(figsize=(12,4))
    plt.suptitle('Mean firing rate (sp/s)', fontsize=18, y=0.99)
    for i in range(m):    
        plt.subplot(2,6, i+1)
        x=1e3*M[i,:,1]
        y=1e3*M[i,:,0]
        plt.plot(x,y , '.k')
        plt.plot([x.min(), x.max()],[x.min(), x.max()], color='gray')
        if y.max()<0.1: # firing rate is too low
            plt.ylim([-0.01,0.1])    
        plt.title('p={:.1f}'.format(p_array[i]))
    plt.tight_layout()
    plt.savefig(path+'mean_firing_rate_p.pdf')
    
    plt.figure(figsize=(12,4))
    plt.suptitle('Fano factor', fontsize=18, y=0.99)
    for i in range(m):
        plt.subplot(2,6, i+1)
        x = FF[i,:,1]
        y = FF[i,:,0]
        plt.plot(x,y , '.k')
        plt.plot([x.min(), x.max()],[x.min(), x.max()], color='gray')
        plt.axis('equal')
        # if i==5:
        #     plt.xlabel('MNN')
        plt.title('p={:.1f}'.format(p_array[i]))
    plt.tight_layout()
    plt.savefig(path+'fano_factor_p.pdf')
            
    idx,idy = np.triu_indices(num_neurons, k=1)
    subsamp = np.random.choice(len(idx), 100, replace=False)
    idx,idy = idx[subsamp],idy[subsamp]
    plt.figure(figsize=(12,4))
    plt.suptitle('Correlation coefficient', fontsize=18, y=0.99)
    for i in range(m):            
        plt.subplot(2,6, i+1)
        y = R[i,:,:,0][idx,idy]
        x = R[i,:,:,1][idx,idy]
        plt.plot(x,y, '.k')
        plt.plot([x.min(), x.max()],[x.min(), x.max()], color='gray')
        plt.axis('equal')
        # if i==5:
        #     plt.xlabel('w')
        plt.title('p={:.1f}'.format(p_array[i]))
    plt.tight_layout()
    plt.savefig(path+'corr_coef_p.pdf')

    # histogram of corr coef
    indx = np.triu_indices(num_neurons, k=1)
    plt.figure(figsize=(12,3))
    plt.suptitle('Histogram, correlation coefficient', fontsize=18, y=0.99)
    for i in range(m):            
        plt.subplot(2,6, i+1)        
        x = R[i,:,:,1][indx] #MNN
        plt.hist(x, np.linspace(-0.1,0.1,30), edgecolor=None, density=True)
        # if i==5:
        #     plt.xlabel('w')
        plt.title('p={:.1f}'.format(p_array[i]))
    plt.tight_layout()
    plt.savefig(path+'hist_corr_coef_p.pdf')

    # histogram of corr coef as line art
    indx = np.triu_indices(num_neurons, k=1)
    plt.figure(figsize=(3.5,3))
    red_gradient = [
    "#FFE5E5",  # Lightest (pinkish-white)
    "#FFCCCC",  # Soft pink-red
    "#FFB3B3",  # Warm pastel red
    "#FF9999",  # Peachy red
    "#FF8080",  # Light coral
    "#FF6666",  # Medium warm red
    "#FF4D4D",  # Vibrant tomato red
    "#FF3333",  # Bright pure red
    "#FF1A1A",  # Intense red
    "#FF0000",  # Classic #FF0000 (RGB red)
    "#CC0000"   # Deep crimson
]
    for i in range(m):
        if i==0: continue  # skip p=0 since corr is 0            
        x = R[i,:,:,1][indx] #MNN
        bin_edges = np.linspace(-0.1,0.1,50)        
        counts, _ = np.histogram(x, bin_edges , density=True)
        bincenter = bin_edges[1:] - 0.5*(bin_edges[1]-bin_edges[0])        
        plt.plot(bincenter, counts, color=red_gradient[i])
        # if i==5:
        #     plt.xlabel('w')
    plt.legend(['p={:.1f}'.format(p) for p in p_array[1:]])
    plt.xlabel('Noise correlation')
    plt.ylabel('PDF')
    plt.tight_layout()
    plt.savefig(path+'histline_corr_coef_p.pdf')

    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(path+'histlog_corr_coef_p.pdf')

    # average stats
    plt.figure(figsize=(3.5*3,3))
    plt.subplot(1,3,1)
    y = 1e3*M[:,:,1].mean(1)
    dy = (1e3*M[:,:,1]).std(1)
    plt.fill_between(p_array, y-dy, y+dy, color='#1f77b4', alpha=0.2, label='Error Range')
    plt.plot(p_array, y)
    plt.xlabel('Conn. prob.')
    plt.ylabel('Mean firing rate (sp/s)')
    plt.subplot(1,3,2)
    y = FF[:,:,1].mean(1)
    dy = FF[:,:,1].std(1)
    plt.fill_between(p_array, y-dy, y+dy, color='#1f77b4', alpha=0.2, label='Error Range')
    plt.plot(p_array, y)
    plt.xlabel('Conn. prob.')
    plt.ylabel('Fano factor)')
    plt.subplot(1,3,3)
    #y = np.array([R[i,:,:,1][indx].std() for i in range(m)])
    y = np.array([np.abs(R[i,:,:,1][indx]).mean() for i in range(m)])
    #dy = np.array([np.abs(R[i,:,:,1][indx]).std() for i in range(m)])
    #plt.fill_between(p_array, y-dy, y+dy, color='#1f77b4', alpha=0.2, label='Error Range')
    plt.plot(p_array, y)
    plt.xlabel('Conn. prob')
    plt.ylabel('Abs. corr. coef.')
    plt.tight_layout()
    plt.savefig(path+'stats_vs_p.pdf')

    return

if __name__=='__main__':
    torch.set_default_dtype(torch.float64)

    #para_sweep_stats_vs_w(mean_ext=2, var_ext=9)
    #plot_stats_vs_w(mean_ext=2, var_ext=9)
    #para_sweep_w_mean_var()

    #rmnn_debug()

    #para_sweep_rand_net_w()
    #plot_rand_net_w()

    #rand_rmnn_debug(mean_ext=1.04, var_ext=16, flush_snn=True)

    #para_sweep_rec_rand_mean_var('rec_rand_mean_var_w5_t10')
    #plot_rec_rand_mean_var('rec_rand_mean_var_w5_t10')

    #para_sweep_rec_rand_p()
    plot_rec_rand_p(exp_id='para_sweep_rec_rand_p', num_neurons=1000, w=10)

    # to run the script in background and print to log file: -u for instant flush
    #nohup python -u -m projects.rmnn.random_recurrent > output.log 2>&1 &

    

    