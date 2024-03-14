# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:59:18 2023

@author: dell
"""

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import os
import fnmatch
from projects.crit.rec_snn_simulator_torch import spk_time2csr, spk_time2count


def load_data(path, indx=None):    
    data_files = os.listdir(path)    
    if indx == None:
        meta_dat = np.load(path+'meta_data.npz', allow_pickle=True)
        return meta_dat        
    for f in data_files:
        if fnmatch.fnmatch( f, str(indx).zfill(3) + '*.npz'):
            dat = np.load(path+f, allow_pickle=True)
            return dat

path = './projects/crit/runs/snn_vary_ie_ratio_spont_bgrate_40/'  # with corr

meta_dat = load_data(path) #load meta data
print(meta_dat)


# uext = []
# ie_ratio = []
# N = []
# degree_hetero = []
# w = []

try:
    uext = meta_dat['uext_array']
    ie_ratio = meta_dat['ie_ratio_array']
    size = (len(uext), len(ie_ratio))
except:
    pass

# try:
#     uext = meta_dat['uext_array']
#     N = meta_dat['N_array']        
#     size = (len(uext), len(N))
# except:
#     pass

# try:
#     uext = meta_dat['uext_array']
#     degree_hetero = meta_dat['degree_hetero_array']            
#     size = (len(uext), len(degree_hetero))
# except:
#     pass

# try:
#     uext = meta_dat['uext_array']
#     w = meta_dat['w_array']
#     size = (len(uext), len(w))
# except:
#     pass

# # [:,:,0] is ext; [:,:,1] is inh
mean_pop_avg = np.zeros(size)
ff_pop_avg = np.zeros(size)
# corr_pop_avg = np.zeros(size+(3,))

mean_pop_std = np.zeros(size)
ff_pop_std = np.zeros(size)
# corr_pop_std = np.zeros(size+(3,))

# mean_quartiles = np.zeros(size+(2,))
# ff_quartiles = np.zeros(size+(2,))

# osc_amp = np.zeros(size)
# osc_amp_ff = np.zeros(size)
# osc_freq = np.zeros(size)

for i in range(size[0]):
    print('Processing... {}/{}'.format(i, size[0]))
    for j in range(size[1]):
        print(j)        
        indx = np.ravel_multi_index((i,j), size )
        dat = load_data(path, indx)
        
        if dat==None:
            print('No data found! Skipping...')
            continue
        
        config = dat['config'].item()  
        #print(config)          
        NE = config['NE']
        NI = config['NI']
        
        try:
            t = dat['t']
        except:
            print('t not found! Skipping...')
            continue        
        
        # spk history data: #spikes x 3 , each column is trial_id, neuron_id, time
        #print(dat['spk_history'])
        spk_history = dat['spk_history']#.item()
        
        nsteps = int(config['T_snn']/config['dt_snn'])
        nneurons = config['NE']+config['NI']    

        sparse_matrix = spk_time2csr(spk_history, nneurons, nsteps, sample_id = 0)
        
        spk_count = np.asarray(np.sum(sparse_matrix, axis=0)) #sum up over all neurons
        binwidth = 10
        spk_count = spk_count.reshape( int(spk_count.shape[1]/binwidth) , binwidth ).sum(axis=1)
        
        pop_firing_rate = spk_count/nneurons/0.1*1e3
        print('Avg pop firing rate (sp/s)', pop_firing_rate.mean())


        pop_firing_rate = spk_count/nneurons/(binwidth*config['dt_snn'])*1e3 # sp/s

        tt= np.linspace(0,t[-1], spk_count.shape[0])    
        
        # #!!! VERY BROKEN!!!!
        # #timewindow = 100 #ms
        # #spk_count = spk_time2count(spk_history, timewindow, config) #batch x neuron x time_bin
        
        # #calcualte stats over time windows
        # rate = spk_count.mean(axis=2)/timewindow # kHz
        # ff = np.var(spk_count, axis=2)/(1e-12+np.mean(spk_count,axis=2))

        # #avg over neurons AND trials
        # mean_pop_avg[i,j] = rate.mean()
        mean_pop_avg[i,j] = pop_firing_rate.mean()
        # ff_pop_avg[i,j] = ff.mean()
        # mean_pop_std[i,j] = rate.std()
        # ff_pop_std[i,j] = ff.std()

        # # event plot & pop rate
        plt.figure()
        plt.subplot(2,1,1)
        neuron_indices = spk_history[spk_history[:,0]==0, 1]
        time_indices = spk_history[spk_history[:,0]==0, 2]
        tmp_t = time_indices*config['dt_snn']
        plt.plot(tmp_t[tmp_t>900], neuron_indices[tmp_t>900], ',', color='black', markersize=0.1)
        #plt.eventplot(time_indices*dt/1e3, neuron_indices, color='black')#, lineoffsets=0.5, linelengths=0.9, linewidth=2)
        plt.ylabel('Neuron index')

        plt.subplot(2,1,2)
        plt.bar(tt[tt>900] , pop_firing_rate[tt>900], width = tt[1]-tt[0], color='black', edgecolor='none')
        #plt.xlim([t[0],t[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Pop. firing rate (sp/s)')

        plt.tight_layout()
        plt.savefig(path+'pop_firing_rate_{}_{}.png'.format(i,j))
        plt.close('all')



#tt = np.linspace(0,t[-1],num_bins)

plt.figure()

mean_pop_avg = np.squeeze(mean_pop_avg)
ff_pop_avg = np.squeeze(ff_pop_avg)
mean_pop_std = np.squeeze(mean_pop_std)
ff_pop_std = np.squeeze(ff_pop_std)

plt.figure()
plt.plot(ie_ratio,mean_pop_avg)
plt.fill_between(ie_ratio, mean_pop_avg - mean_pop_std, mean_pop_avg + mean_pop_std, alpha=0.3)
#plt.ylim([0,200])
plt.xlabel('IE weight ratio')
plt.ylabel('Mean firing rate (sp/s)')
plt.savefig(path+'mean_firing_rate_vs_IE_ratio.png')

plt.figure()
plt.plot(ie_ratio,ff_pop_avg)
#plt.fill_between(ie_ratio, ff_pop_avg - ff_pop_std, ff_pop_avg + ff_pop_std, alpha=0.3)
#plt.ylim([0,2])
plt.xlabel('IE weight ratio')
plt.ylabel('Fano factor')
plt.savefig(path+'fano_factor_vs_IE_ratio.png')
plt.close('all')

print('ie_ratio', ie_ratio)
print('avg pop firing rate', mean_pop_avg)
        
#         u = dat['mnn_mean']
#         s = dat['mnn_std']
#         ff = s**2/u
        
#         try:
#             rho_ee = dat['mnn_corr'][:NE,:NE]
#             rho_ei = dat['mnn_corr'][:NE,NE:]
#             rho_ii = dat['mnn_corr'][NE:,NE:]
            
#             rho_ee = rho_ee[np.triu_indices( NE ,k=1)] #upper triangle entries
#             corr_pop_avg[i,j,0] = np.mean(rho_ee)
#             corr_pop_std[i,j,0] = np.std(rho_ee)
            
#             rho_ii = rho_ii[np.triu_indices( NI ,k=1)] #upper triangle entries
#             corr_pop_avg[i,j,1] = np.mean(rho_ii)
#             corr_pop_std[i,j,1] = np.std(rho_ii)
            
#             corr_pop_avg[i,j,2] = np.mean(rho_ei)
#             corr_pop_std[i,j,2] = np.std(rho_ei)
            
#         except:
#             corr_pop_avg[i,j,:] = None
#             corr_pop_std[i,j,:] = None
        
#         # average over second half of simulation, to deal with oscillating solutions
#         cut_off = int(u.shape[1]/2) #discard the first half of time series
#         u_time_avg = np.mean(u[:, cut_off:], axis = 1)  #average over time
#         ff_time_avg = np.mean(ff[:, cut_off:], axis = 1) 
               
#         # population stats
#         mean_pop_avg[i,j,0] = np.mean(u_time_avg[:NE])
#         mean_pop_avg[i,j,1] = np.mean(u_time_avg[NE:])
#         ff_pop_avg[i,j,0] = np.mean(ff_time_avg[:NE])
#         ff_pop_avg[i,j,1] = np.mean(ff_time_avg[NE:])
        
#         mean_pop_std[i,j,0] = np.std(u_time_avg[:NE])
#         mean_pop_std[i,j,1] = np.std(u_time_avg[NE:])
#         ff_pop_std[i,j,0] = np.std(ff_time_avg[:NE])
#         ff_pop_std[i,j,1] = np.std(ff_time_avg[NE:])
        
#         mean_quartiles[i,j,:] = np.percentile(u_time_avg, [25,75])        
#         ff_quartiles[i,j,:] = np.percentile(ff_time_avg, [25,75])
        
#         # detect oscillation
#         tmp = np.mean(u[:, cut_off:], axis=0) #population average, no time average
#         tmp_ff = np.mean(ff[:, cut_off:], axis=0)
        
        
#         #if ie_ratio[j]>4: # no oscilation found for excitation dominant regime
#         osc_amp[i,j] = 0.5*(np.max(tmp)-np.min(tmp)) #rough estimate of oscillation amplitude
#         osc_amp_ff[i,j] = 0.5*(np.max(tmp_ff)-np.min(tmp_ff))
#         #if osc_amp[i,j]>1e-5:
#         psd = np.abs(np.fft.fft(tmp))
#         psd[0]=0
#         psd = psd[:int(len(psd)/2)] #discard mirrored result
#         osc_freq[i,j] = np.argmax(psd)/(config['T_mnn']/2*0.02)  # psd peak index * df, which is 1/simulation time (0.02 s is mem constant)
            

# dat = {'ie_ratio':ie_ratio,
# 'uext':uext,
# 'degree_hetero':degree_hetero,
# 'N':N,
# 'w':w,
# 'mean_pop_avg':mean_pop_avg,
# 'ff_pop_avg':ff_pop_avg,
# 'mean_pop_std':mean_pop_std,
# 'ff_pop_std':ff_pop_std,
# 'osc_amp':osc_amp,
# 'osc_freq':osc_freq,
# 'osc_amp_ff':osc_amp_ff,
# 'mean_quartiles':mean_quartiles,
# 'ff_quartiles':ff_quartiles,
# 'corr_pop_avg':corr_pop_avg,
# 'corr_pop_std':corr_pop_std,
# }

# np.savez(path+'post_analysis.npz', **dat)