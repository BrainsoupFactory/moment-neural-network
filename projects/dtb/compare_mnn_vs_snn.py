# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:01:36 2024

@author: qiyangku
"""
import numpy as np
from projects.dtb.ma_conductance import Moment_Activation_Cond
from matplotlib import pyplot as plt


def test(exp_id, indx = 0):
    path = './projects/dtb/runs/{}/'.format(exp_id)
    
    dat = np.load(path+str(indx).zfill(3)+'.npz', allow_pickle=True)
    config=dat['config'].item()
    print(list(dat))
    
    # ['spk_count',
    #  'config',
    #  'exc_input_rate',
    #  'inh_input_rate',
    #  'rho',
    #  'spk_count_history',
    #  't',
    #  'T_snn',
    #  'dt_snn']
    
    ma = Moment_Activation_Cond()
    #then update model parameters to be consistent with SNN
    ma.tau_L = 20 # membrane time constant
    ma.tau_E = 4 # excitatory synaptic time scale (ms)
    ma.tau_I = 10 # inhibitory synaptic time scale (ms)
    ma.VL = -70 # leak reverseal potential
    ma.VE = 0 # excitatory reversal potential
    ma.VI = -70 # inhibitory reversal potential
    ma.vol_th = -50 # firing threshold
    ma.vol_rest = -60 # reset potential
    ma.L = 1/ma.tau_L
            
    ma.sE = 1#0.125/ma.L # modifier to conductance (can be voltage dependent)
    ma.sI = 1#5.46/ma.L
    
    ma.t_ref = 2 # ms; NB inhibitory neuron has different value
    
    # TODO: horrible scalability; consider a list of g, input_mean, input_std
    # so it supports any number of channels

    if exp_id == 'vary_input_stats':        
        we=0.1*np.array([1,2,3])[indx]
        wi=0.4
        ma.tau_E *= 1.44490582
        ma.tau_I *= 2.14643191
    elif exp_id == 'vary_input_stats_poisson':
        we=np.linspace(0,1,21)[1:][indx]
        wi=0.4
        ma.tau_E *= 1.44490582
        ma.tau_I *= 2.14643191
    elif exp_id == 'vary_x_input_stats':
        we = 0.1 #np.linspace(0,1.0,21)[1:][indx]
        wi = 0.4 # 0.4, 1.0, 10
        ma.tau_E *= 1.44490582
        ma.tau_I *= 2.14643191
        x_input_rate = np.linspace(0,1,11)[indx]
        
    # add manual corrections
    #ma.tau_E *= 1.45
    #ma.tau_I *= 1.8
    #ma.tau_E *= 1.414
    #ma.tau_I*=1.414
    
    #ma.tau_E *= 1.414
    #ma.tau_I *= 1.75
    
    try:
        exc_rate = dat['exc_input_rate'] # firng rate in kHz
        inh_rate = dat['inh_input_rate']
        exc_FF = dat['inh_input_FF']
        inh_FF = dat['inh_input_FF']
        M1,FF1,M2,FF2 = np.meshgrid(exc_rate, exc_FF, inh_rate, inh_FF, indexing='ij')
        
    except:
        exc_rate = dat['exc_input_rate'] # firng rate in kHz
        inh_rate = dat['inh_input_rate']
        M1,M2 = np.meshgrid(exc_rate, inh_rate, indexing='xy')
        FF1 = 1
        FF2 = 1
        
    exc_input_mean = we*M1
    exc_input_std = we*np.sqrt(M1*FF1)
    inh_input_mean = wi*M2
    inh_input_std = wi*np.sqrt(M2*FF2)
    
    eff_input_mean, eff_input_std, tau_eff = ma.cond2curr(exc_input_mean,exc_input_std,inh_input_mean,inh_input_std)
    try:
        eff_input_mean = eff_input_mean + x_input_rate*ma.tau_E/ma.tau_L
        h0 = np.sqrt(x_input_rate)*ma.tau_E/ma.tau_L
        eff_input_std = np.sqrt(eff_input_std*eff_input_std + h0*h0*tau_eff/(tau_eff+ma.tauE) )
    except:
        pass
    mean_out = ma.forward_fast_mean( eff_input_mean, eff_input_std, tau_eff)
    std_out = ma.forward_fast_std( eff_input_mean, eff_input_std, tau_eff, mean_out)
    
    
    T = dat['T_snn']-config['discard'] # ms
    
    mean_firing_rate = dat['spk_count'].mean(0)/T
    firing_var = dat['spk_count'].var(0)/T
    
    plt.close('all')
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    plt.plot(mean_out.flatten(), mean_firing_rate,'.', markersize=1)
    plt.plot( [0,0.07] , [0, 0.07], color='gray')
    #plt.plot(mean_firing_rate,'--')
    #plt.plot(mean_out.flatten(), mean_firing_rate.flatten(),'.')
    
    #plt.plot(mean_out.T, color='gray')
    #plt.plot(mean_firing_rate.reshape(mean_out.shape).T,'--')
    
    
    #plt.xlabel(r'$\tau_E$ (ms)')
    plt.xlabel('Mean firing rate (sp/ms)')
    plt.ylabel('Mean firing rate (sp/ms)')

    plt.subplot(1,2,2)
    #plt.plot(std_out.flatten()*std_out.flatten())
    #plt.plot(firing_var,'--')
    plt.plot(std_out.flatten()*std_out.flatten(), firing_var, '.', markersize=1)
    plt.plot( [0,0.03] , [0, 0.03], color='gray')
    plt.xlabel('Firing variability (sp$^2$/ms)')
    plt.ylabel('Firing variability (sp$^2$/ms)')

    plt.tight_layout()
    
    plt.savefig(path+'mnn_vs_snn_indx_{}.png'.format(indx), dpi=300)
    plt.close('all')


    plt.figure(figsize=(16,2))
    plt.subplot(2,1,1)
    plt.plot(mean_out.flatten())#, lw=0.2)
    plt.plot(mean_firing_rate,'--')#, lw=0.2)
    
    #plt.xlabel(r'$\tau_E$ (ms)')
    #plt.xlabel('Mean firing rate (sp/ms)')
    #plt.ylabel('Mean firing rate (sp/ms)')

    plt.subplot(2,1,2)
    plt.plot(std_out.flatten()*std_out.flatten())#, lw=0.2)
    plt.plot(firing_var,'--')#, lw=0.2)
    #plt.xlabel('Firing variability (sp$^2$/ms)')
    #plt.ylabel('Firing variability (sp$^2$/ms)')

    plt.tight_layout()
    plt.savefig(path+'mnn_vs_snn_flat_indx_{}.png'.format(indx), dpi=300)
    #plt.savefig(path+'mnn_vs_snn_flat_indx_{}.pdf'.format(indx), format='pdf')
    
    plt.close('all')

    

def vary_tau_E(exp_id):
    path = './projects/dtb/runs/{}/'.format(exp_id)
    
    if exp_id=='vary_tau_E':
        tau_E = np.linspace(0,100,21)
    elif exp_id=='vary_tau_E_zoom_in':
        tau_E = np.linspace(0,10,21)
    
    snn_rate = np.zeros((len(tau_E),3))
    snn_var =  np.zeros((len(tau_E),3))
    mnn_rate = np.zeros((len(tau_E),3))
    mnn_var = np.zeros((len(tau_E),3))
    
    for indx in range(len(tau_E)):
        print(indx)
        dat = np.load(path+str(indx).zfill(3)+'.npz', allow_pickle=True)
        config=dat['config'].item()
        # ['spk_count',
        #  'config',
        #  'exc_input_rate',
        #  'inh_input_rate',
        #  'rho',
        #  'spk_count_history',
        #  't',
        #  'T_snn',
        #  'dt_snn']
        
        ma = Moment_Activation_Cond(config)
        ma.tau_E = tau_E[indx]
        #then update model parameters to be consistent with SNN
        
        #add manual correction; doesn't work - can't make sure it work for all tau_E
        #ma.tau_E *= 1.414
        #ma.tau_I *= 1.75
        
        we=0.1
        wi=0.4
        
        exc_input_mean = we*dat['exc_input_rate']
        exc_input_std = we*np.sqrt(dat['exc_input_rate'])
        inh_input_mean = wi*dat['inh_input_rate']
        inh_input_std = wi*np.sqrt(dat['inh_input_rate'])
        
        eff_input_mean, eff_input_std, tau_eff = ma.cond2curr(exc_input_mean,exc_input_std,inh_input_mean,inh_input_std)
        
        mean_out = ma.forward_fast_mean( eff_input_mean, eff_input_std, tau_eff)
        std_out = ma.forward_fast_std( eff_input_mean, eff_input_std, tau_eff, mean_out)
        
        T = dat['T_snn']-config['discard'] # ms
        snn_rate[indx,:] = dat['spk_count'].mean(0)/T
        snn_var[indx,:] = dat['spk_count'].var(0)/T
        mnn_rate[indx,:] = mean_out.flatten()
        mnn_var[indx,:] = std_out.flatten()**2
    
        
    gain = snn_rate/mnn_rate
    gain_mean = gain[5:,:].mean()
    gain_std = gain[5:,:].std()
    print('gain = {} +/- {}'.format(gain_mean, gain_std))
    
    
#    colors = ['#4169E1','#DC143C','#228B22'] # RBG
    colors = ['#145369', '#427a99', '#70a0c1'] #blues
    plt.close('all')
    plt.figure(figsize=(7,3))
    plt.subplot(1,2,1)
    for i in range(mnn_rate.shape[1]):
        plt.plot(tau_E, mnn_rate[:,i],color=colors[i])
        plt.plot(tau_E, snn_rate[:,i],'.', color=colors[i])
        #plt.plot(tau_E, mnn_rate[:,i]-snn_rate[:,i], color=colors[i])
    plt.xlabel(r'$\tau_E$ (ms)')
    plt.ylabel('Mean firing rate (sp/ms)')
    #plt.xlim(0,25)
    
    plt.subplot(1,2,2)
    for i in range(mnn_rate.shape[1]):    
        plt.plot(tau_E, mnn_var[:,i],color=colors[i])
        plt.plot(tau_E, snn_var[:,i],'.', color=colors[i])
    plt.xlabel(r'$\tau_E$ (ms)')
    plt.ylabel('Firing variability  (sp$^2$/ms)')
    #plt.xlim(0,100)
    plt.tight_layout()
    
    #plt.figure()
    #plt.plot(snn_rate/mnn_rate)
    
    return snn_rate, snn_var, mnn_rate, mnn_var
        
        
if __name__=='__main__':
    #test('test2')
    #for i in range(3):
    #test('vary_input_stats',  indx=0)
    #exp_id = 'vary_input_stats_poisson'
    exp_id = 'vary_x_input_stats'
    for i in range(20):
        test(exp_id,  indx=i)
    
    #vary_tau_E('vary_tau_E_zoom_in')
    #vary_tau_E('vary_tau_E')
    



