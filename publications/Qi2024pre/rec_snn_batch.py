# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 01:06:29 2021
A helper function for running multiple runs with different config
@author: dell
"""

from projects.pre2023.rec_snn_simulator_torch import *
from projects.pre2023.rec_mnn_simulator import gen_synaptic_weight, gen_config
#from matplotlib import pyplot as plt
import numpy as np
import os, sys, time

#For PBS:
#INPUT: search_space a dictionary of lists
#       PBS_array index
#Wrapper: nested loop over the search_space
#Output the config dictionary

def run(config, record_ts = True ):
    
    W = gen_synaptic_weight(config) #doesn't take too much time with 1e4 neurons    
    #W = W.numpy() #convert to numpy
    input_gen = SNNInputGenerator(config).ind_poisson
    #mnn_model = RecurrentMNN(config, W, input_gen)
    snn_model = InteNFireRNN(config, W.T , input_gen)
    #snn_model = InteNFireRNN_filtered(config, W.T , input_gen)
    spk_count, spk_history, V, t, pop_spk_count = snn_model.run( config['T_snn'] ,show_message=True, device = config['device'])
    return spk_count.cpu().numpy(), spk_history, t, pop_spk_count.cpu().numpy()

def quick_plots(config, spk_history, path, indx, sample_id = 0):
    print('Plotting results...')
    nsteps = int(config['T_snn']/config['dt_snn'])
    nneurons = config['NE']+config['NI']    
    sparse_mat = spk_time2csr(spk_history, nneurons, nsteps, sample_id = sample_id)
    
    spk_count = np.asarray(np.sum(sparse_mat, axis=0))
    binwidth = 10 # dt = 0.01 ms; binwidth of 10 is equal to 0.1 ms window
    spk_count = spk_count.reshape( int(spk_count.shape[1]/binwidth) , binwidth ).sum(axis=1)
    
    pop_firing_rate = spk_count/nneurons/(binwidth*config['dt_snn'])*1e3
    print('Avg pop firing rate (sp/s)', pop_firing_rate.mean())

    #tt= np.linspace(0,t[-1], spk_count.shape[0])
    
    dt = config['dt_snn']
    dT = 100 #ms
    cut_off = (config['T_snn']-dT)/1e3

    plt.figure()    
    plt.subplot(2,1,1)
    neuron_indices, time_indices, _ = sp.sparse.find(sparse_mat) #plot first 100 ms
    tmp_t = time_indices*dt/1e3

    plt.plot(tmp_t[tmp_t>cut_off], neuron_indices[tmp_t>cut_off], ',', color='black', markersize=0.1)        
    plt.title('bg_rate={}, ie_ratio={}'.format(config['bg_rate'],config['ie_ratio']))
        
    plt.subplot(2,1,2)
    tmp_t = np.linspace(0,config['T_snn'], len(pop_firing_rate))/1e3
    plt.bar( tmp_t[tmp_t>cut_off], pop_firing_rate[tmp_t>cut_off], width=binwidth*dt/1e3, color='black', edgecolor='none')
    plt.xlabel('Time (s)')
    plt.ylabel('Pop. firing rate (sp/s)')
    plt.tight_layout()
    plt.savefig(path+'pop_firing_rate_{}.png'.format(str(indx).zfill(3)))
    plt.close('all')


    return 
    
if __name__ == "__main__":    
    
    indx = int(sys.argv[1]) #use this to pick a particular config
    exp_id = sys.argv[2] # id for the set of experiment
    
    #beta=1.1676406135865596 # calibration factor
    uext_array = np.array([20,40]) # only simulate specific slices
    ie_ratio_array = np.linspace(0, 8.0, 41) 
    delay_array = np.array([1.5]) #ms

    i,j,k = np.unravel_index(indx, [len(uext_array), len(ie_ratio_array), len(delay_array)] ) 
    # to get linear index back from subscripts, use: np.ravel_multi_index((i,j),[len(uext_array), len(ie_ratio_array)])
    
    config = gen_config(N=12500, w=0.1, ie_ratio=ie_ratio_array[j], bg_rate=uext_array[i])
    config = update_config_snn(config)
    config['device']=sys.argv[3]
    
    config['delay_snn'] = delay_array[k]

    config['Vth'] = 20.0 #mV, firing threshold, default 20
    config['Vres'] = 10.0 #mV reset potential; default 0
    config['Tref'] = 2.0
    config['T_snn'] = int(1e3)
    config['batchsize'] = 1000
    config['randseed'] = 0
    config['discard'] = 100 # ms
    config['transient_corr'] = None
    config['transient_duration'] = None #ms
    
    # temporary test override
    # config['T_snn'] = 1e3
    # config['transient_corr'] = 0.8
    # config['transient_duration'] = 100 #ms
    # config['discard'] = 200 # ms
    config['batchsize'] = 100
    config['dt_snn'] = 1e-3
    config['delay_snn'] = 0 #only care about async activity

    print(config)

    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    if not os.path.exists(path):
        os.makedirs(path)
    np.savez(path+'meta_data.npz', exp_id=exp_id, uext_array=uext_array, ie_ratio_array=ie_ratio_array, delay_array=delay_array)
    
    # MAIN SIMULATION
    spk_count, spk_history, t, pop_spk_count = run(config) 
    #
    
    file_name = str(indx).zfill(3) + '.npz'
    np.savez(path+file_name, config=config, t=t, spk_count=spk_count, pop_spk_count=pop_spk_count)
    #np.savez(path+file_name, config=config, t=t, spk_count=spk_count, spk_history=spk_history)
    
    print('Results saved to: '+path+file_name)
    quick_plots(config, spk_history, path, indx) # don't save spike history; takes too much space!

