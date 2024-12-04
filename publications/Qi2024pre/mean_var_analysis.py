# analyzing corr

import numpy as np
from matplotlib import pyplot as plt
from mnn.mnn_core.mnn_pytorch import *
from projects.pre2023.vanilla_maf import MomentActivation
from pprint import pprint


def run_mnn(exp_id):
    indx = 0
    filename = str(indx).zfill(3) + '.npz'
    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    dat = np.load(path+filename, allow_pickle=True)
    #print(dat)
    
    # pytorch ver
    if exp_id == '2024_mar_30_mean_std':
        mean_grid = torch.tensor(dat['mean_grid'])
        std_grid = torch.tensor(dat['std_grid'])
    elif exp_id == '2024_Apr_2_w_rate':
        w = torch.tensor(dat['input_w_grid'])
        rate = torch.tensor(dat['input_rate_grid'])
        mean_grid = w*rate
        std_grid = w*np.sqrt(rate)
    elif exp_id == '2024_Apr_6_w_rate':
        w = torch.tensor(dat['w_grid'])
        exc_rate = torch.tensor(dat['exc_rate_grid'])
        inh_rate = torch.tensor(dat['inh_rate_grid'])
        mean_grid = w*(exc_rate-inh_rate)
        std_grid = w*np.sqrt(exc_rate+inh_rate)
    elif exp_id in ['2024_Apr_13_w_rate','2024_Apr_17_w_rate']:
        w = torch.tensor(dat['w_grid'])
        #exc_rate = torch.tensor(dat['exc_rate_grid'])
        #inh_rate = torch.tensor(dat['inh_rate_grid'])
        mean_grid = torch.tensor(dat['curr_mean_grid'])
        std_grid = torch.tensor(dat['curr_std_grid'])
    else:
        print('Nothing happened!!')

    mean_out, std_out = mnn_activate_no_rho(mean_grid, std_grid)
    mean_out = mean_out.numpy()
    std_out = std_out.numpy()
    # vanilla ver
    # mean_grid = dat['mean_grid']
    # std_grid = dat['std_grid']
    # maf=MomentActivation()
    # mean_out = maf.mean(mean_grid, std_grid)
    # std_out, FF_out = maf.std(mean_grid, std_grid)
    
    # TODO: this uses default neuronal parameters!

    return mean_out, std_out

def analyze_mean_var_old(exp_id, indx, savefile=False, savefig=False):
    #exp_id = '2024_mar_13'
    print('Processing data ', indx)
    filename = str(indx).zfill(3) + '.npz'
    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    dat = np.load(path+filename, allow_pickle=True)
    #print(dat)

    spk_count = dat['spk_count']
    mean_grid = dat['mean_grid']
    std_grid = dat['std_grid']
    config = dat['config'].item()
    rho = dat['rho']
    #print(config)

    T = config['T_snn'] - config['discard'] #ms

    
    print('Calculating moment activation...')
    mnn_mean, mnn_std, mnn_corr=run_mnn(exp_id, indx)
    mnn_FF = mnn_std.pow(2.0)/(1e-12+mnn_mean)    
    mnn_mean = mnn_mean.numpy().squeeze()
    mnn_std = mnn_std.numpy().squeeze()
    mnn_FF = mnn_FF.numpy().squeeze()
    
    T = config['T_snn'] - config['discard']
    print('Calculating spike count stats...')
    snn_mean = np.mean(spk_count, axis=0)/T
    snn_std = np.std(spk_count,axis=0)/np.sqrt(T)
    snn_fano_factor = np.var(spk_count,axis=0)/(1e-6+np.mean(spk_count,axis=0))
    
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(snn_mean*1e3)
    # plt.plot(mnn_mean.squeeze()*1e3,'--')
    # plt.ylabel('Firing rate (sp/s)')
    # plt.subplot(2,1,2)
    # plt.plot(snn_fano_factor)
    # plt.plot(mnn_FF.squeeze(),'--')
    # plt.ylabel('Fano factor')
    # plt.tight_layout()
    # plt.savefig(path+str(indx).zfill(3)+'_snn_rate_FF.png')
    
    size = (51,5) 
    snn_mean = snn_mean.reshape(size) #inner dim is std
    mnn_mean = mnn_mean.reshape(size)
    snn_fano_factor = snn_fano_factor.reshape(size)
    mnn_FF = mnn_FF.reshape(size)
    snn_std = snn_std.reshape(size)
    mnn_std = mnn_std.reshape(size)

    input_mean = mean_grid.reshape(size)[:,0]
    input_std = std_grid.reshape(size)[0,:]

    if savefig:
        legend = ['$\\bar{\\sigma}=$'+ str(s) for s in input_std.squeeze()]
        #colors = ['#3d7dbf', '#649ecb', '#8ccbdd', '#b4e4e9', '#daf4f6']
        colors = ['#2c6aae', '#4e88ba', '#70a6c6', '#91c4d0', '#b2e2db']
        #colors = ['#5277a6', '#6c8caa', '#869fb0', '#9db2b6', '#b5c5bc']
        ax_pos = (0.25,0.2,0.7,0.7) #left, bottom, width, height
        print(legend)
        plt.figure(figsize = (3.5,3))        
        for i in range(size[1]):            
            plt.plot(input_mean, mnn_mean[:,i], color=colors[i])
        for i in range(size[1]):
            plt.plot(input_mean, snn_mean[:,i], '.', markersize=4, color=colors[i])
        plt.ylabel('Firing rate $\\mu$ (sp/ms)')
        plt.xlabel('Input current mean $\\bar{\\mu}$ (mV/ms)')        
        plt.legend(legend, frameon=False)
        plt.gca().set_position(ax_pos)        
        plt.savefig(path+str(indx).zfill(3)+'_firing_rate.png')
        
        plt.figure(figsize = (3.5,3))
        for i in range(size[1]):
            plt.plot(input_mean, mnn_std[:,i], color=colors[i])
            #plt.plot(input_mean, snn_std[:,i], '.', markersize=4, color=colors[i])
            plt.plot(input_mean, snn_std[:,i], '--', color=colors[i])
        plt.ylabel('Firing variability $\\sigma$ (sp/ms$^{1/2}$)')  
        plt.xlabel('Input current std $\\bar{\\sigma}$ (mV/ms$^{1/2}$)')             
        plt.gca().set_position(ax_pos)
        plt.savefig(path+str(indx).zfill(3)+'_firing_variability.png')
        plt.figure(figsize = (3.5,3))
        for i in range(size[1]):
            plt.plot(input_mean, mnn_FF[:,i], color=colors[i])
            #plt.plot(input_mean, snn_fano_factor[:,i], '.', markersize=4, color=colors[i])
            plt.plot(input_mean, snn_fano_factor[:,i], '--', color=colors[i])
        plt.ylabel('Fano factor')        
        plt.xlabel('Input current mean $\\bar{\\mu}$ (mV/ms)')        
        plt.gca().set_position(ax_pos)
        plt.savefig(path+str(indx).zfill(3)+'_fano_factor.png')
        plt.close('all')
    return #mnn_mean, mnn_std, mnn_corr, snn_rate, snn_std, snn_corr, mean_grid, std_grid,rho

def analyze_mean_var(exp_id, savefile=False, savefig=False):
    #exp_id = '2024_mar_13'
    indx = 0 # dummy variable
    filename = str(indx).zfill(3) + '.npz'
    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    dat = np.load(path+filename, allow_pickle=True)
    print(list(dat))
    #['spk_count', 'config', 'mean_grid', 'std_grid', 'rho', 'indx', 'spk_count_history', 't']
    spk_count = dat['spk_count']
    mean_grid = dat['mean_grid']
    std_grid = dat['std_grid']
    config = dat['config'].item()
    rho = dat['rho']

    T = config['T_snn'] - config['discard']

    print('Calculating spike count stats...')
    snn_mean = np.mean(spk_count, axis=0)/T
    snn_std = np.std(spk_count, axis=0)/np.sqrt(T)
    snn_fano_factor = np.var(spk_count,axis=0)/(1e-6+np.mean(spk_count,axis=0))

    print('Calculating moment activation...')
    mnn_mean, mnn_std =run_mnn(exp_id)
    #mnn_FF = mnn_std.pow(2.0)/(1e-12+mnn_mean)    #pow only works for numpy
    
    #mnn_mean = mnn_mean.numpy().squeeze()
    #mnn_std = mnn_std.numpy().squeeze()
    #mnn_FF = mnn_FF.numpy().squeeze()

    print('Plotting results...')
    if exp_id=='2024_mar_29_mean_std':
        size = (21,20)
    elif exp_id=='2024_mar_30_mean_std':
        size = (51,50)
        
    input_mean = mean_grid.reshape(size).T
    input_std = std_grid.reshape(size).T
    snn_mean = snn_mean.reshape(size).T
    snn_std = snn_std.reshape(size).T

    mnn_mean = mnn_mean.reshape(size).T
    mnn_std = mnn_std.reshape(size).T

    extent = (input_mean[0,0], input_mean[0,-1], input_std[0,0], input_std[-1,0])
    plt.figure()
    plt.subplot(2,2,1)        
    plt.imshow(np.abs(snn_mean-mnn_mean), origin='lower', extent=extent)
    plt.title('abs mean error')
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(np.abs(snn_std-mnn_std), origin='lower', extent=extent)
    plt.title('abs std error')
    plt.colorbar()
    
    plt.subplot(2,2,3)        
    plt.imshow(np.abs(snn_mean-mnn_mean)/mnn_mean, origin='lower', vmax=0.1, extent=extent)
    plt.title('rel mean error')
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(np.abs(snn_std-mnn_std)/mnn_std, origin='lower', vmax=0.1, extent=extent)
    plt.title('rel std error')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(path+'mean_std_err.png', dpi=300)
    plt.close('all')

def analyze_fano_factor(exp_id, savefile=False, savefig=False):
    #exp_id = '2024_mar_13'
    indx = 0 # dummy variable
    filename = str(indx).zfill(3) + '.npz'
    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    dat = np.load(path+filename, allow_pickle=True)
    print(list(dat))
    #['spk_count', 'config', 'mean_grid', 'std_grid', 'rho', 'indx', 'spk_count_history', 't']
    spk_count = dat['spk_count']
    mean_grid = dat['mean_grid']
    std_grid = dat['std_grid']
    config = dat['config'].item()
    rho = dat['rho']
    spk_count_history = dat['spk_count_history']
    
    print(spk_count_history.shape) # batch x neuron (i.e. mean x std) x time
    
    dT = config['record_interval'] # ms
    T = config['T_snn'] - config['discard']

    print('Calculating spike count stats...')
    snn_fano_factor = np.var(spk_count_history,axis=0)/(1e-6+np.mean(spk_count_history,axis=0))

    print('Calculating moment activation...')
    mnn_mean, mnn_std =run_mnn(exp_id)
    mnn_fano_factor = mnn_std**2/(mnn_mean+1e-16)
    

    # TODO: let MA output exact FF...
    print('mnn_fano_factor.shape: ', mnn_fano_factor.shape)
    pprint(config)

    if exp_id=='2024_mar_30_mean_std':
        size = (51,50)
    mnn_fano_factor = mnn_fano_factor.reshape(size).T

    print('Plotting results...')
    plt.figure(figsize=(10,10))
    for i in range(100):
        plt.subplot(10,10,i+1)
        transient_FF = snn_fano_factor.squeeze()[:,i].reshape(size).T #
        transient_FF[transient_FF==0] = np.nan # replace with NAN
        if i==0:
            plt.imshow(mnn_fano_factor, cmap = 'plasma',vmin=0, vmax=1.0, origin='lower')
            plt.title('FF (MNN)')
        else:
            plt.imshow(transient_FF, cmap = 'plasma',vmin=0, vmax=1.0, origin='lower')
            plt.title('t={} s'.format( np.round(dT*1e-3*i,2) ))
        plt.xticks([])
        plt.yticks([])
        
    plt.tight_layout()
    plt.savefig(path+'FF_vs_time.png', dpi=300)

    plt.figure(figsize=(10,10))
    for i in range(100): #plot difference between analytical and empirical FF
        plt.subplot(10,10,i+1)
        transient_FF = snn_fano_factor.squeeze()[:,i].reshape(size).T #
        transient_FF[transient_FF==0] = np.nan # replace with NAN
        plt.imshow(np.abs(transient_FF-mnn_fano_factor), vmin=0,vmax=1e-2, cmap = 'plasma', origin='lower')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(path+'diff_FF_vs_time.png', dpi=300)
        #snn_fano_factor = snn_fano_factor.reshape(size).T
        #mnn_fano_factor = mnn_fano_factor.reshape(size).T
    plt.close('all')
    return 

def analyze_w_rate():
    exp_id = '2024_Apr_2_w_rate'
    indx = 0 # dummy variable
    filename = str(indx).zfill(3) + '.npz'
    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    dat = np.load(path+filename, allow_pickle=True)
    print(list(dat))
    #['spk_count', 'config', 'input_w', 'input_rate', 'input_w_grid', 'input_rate_grid', 'rho', 'spk_count_history', 't', 'T_snn', 'dt_snn']spk_count = dat['spk_count']
    #input_w_grid = dat['input_w_grid']
    #input_rate_grid = dat['input_rate_grid']
    input_w = dat['input_w']#torch.logspace(-2,1,31, device=device)
    input_rate = dat['input_rate']
    spk_count = dat['spk_count']
    config = dat['config'].item()
    rho = dat['rho']
    
    T = config['T_snn'] - config['discard']

    print('Calculating spike count stats...')
    snn_mean = np.mean(spk_count, axis=0)/T
    snn_std = np.std(spk_count, axis=0)/np.sqrt(T)
    
    print('Calculating moment activation...')
    mnn_mean, mnn_std =run_mnn(exp_id)

    # reshaping data
    size = (len(input_w), len(input_rate))
    snn_mean = snn_mean.squeeze().reshape(size)
    snn_std = snn_std.squeeze().reshape(size)
    mnn_mean = mnn_mean.squeeze().reshape(size)
    mnn_std = mnn_std.squeeze().reshape(size)
    extent = (input_rate[0],input_rate[-1],np.log10(input_w[0]), np.log10(input_w[-1]))

    plt.figure(figsize=(9,5))
    plt.subplot(2,3,1)
    plt.imshow(snn_mean, vmin=0, vmax=0.2, origin='lower', extent=extent, aspect = 'auto')
    plt.colorbar()
    plt.title('SNN mean')
    plt.subplot(2,3,2)
    plt.imshow(mnn_mean, vmin=0, vmax=0.2, origin='lower', extent=extent, aspect = 'auto')
    plt.colorbar()
    plt.title('MNN mean')
    plt.subplot(2,3,3)
    plt.imshow(np.abs(snn_mean-mnn_mean), origin='lower', extent=extent, aspect = 'auto')
    plt.colorbar()
    plt.title('Error')

    plt.subplot(2,3,4)
    plt.imshow(snn_std, vmin=0, vmax=0.08, origin='lower', extent=extent, aspect = 'auto')
    plt.colorbar()
    plt.title('SNN std')
    plt.subplot(2,3,5)
    plt.imshow(mnn_std, vmin=0, vmax=0.08, origin='lower', extent=extent, aspect = 'auto')
    plt.colorbar()
    plt.title('MNN std')
    plt.subplot(2,3,6)
    plt.imshow(np.abs(snn_std-mnn_std), origin='lower', extent=extent, aspect = 'auto')
    plt.colorbar()
    plt.title('Error')

    plt.tight_layout()
    plt.savefig(path+'snn_v_mnn.png', dpi=300)
    plt.close('all')


    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(mnn_mean.flatten(), snn_mean.flatten(), '.')
    plt.plot([0,0],[0.2, 0.2],color='gray')
    plt.xlabel('MNN mean')
    plt.ylabel('SNN mean')
    plt.subplot(2,2,2)
    plt.plot(mnn_std.flatten(), snn_std.flatten(), '.')
    plt.plot([0,0],[0.08, 0.08],color='gray')
    plt.xlabel('MNN std')
    plt.ylabel('SNN std')
    
    plt.subplot(2,2,3)
    plt.imshow(mnn_std**2/(1e-12+mnn_mean), origin='lower', extent=extent, aspect = 'auto')
    plt.colorbar()
    plt.xlabel('input rate (sp/ms)')
    plt.ylabel('Log w')
    
    plt.tight_layout()
    plt.savefig(path+'snn_v_mnn_scatter.png')
    plt.close('all')

    print('done!')
    return

def analyze_w_rate_EI(plot_ids=[0,1]):
    exp_id = '2024_Apr_17_w_rate'
    indx = 0 # dummy variable
    filename = str(indx).zfill(3) + '.npz'
    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    dat = np.load(path+filename, allow_pickle=True)
    print(list(dat))
    #['spk_count', 'config', 'exc_rate', 'inh_rate', 'input_w', 'exc_rate_grid', 'inh_rate_grid', 'w_grid', 'rho', 'spk_count_history', 't', 'T_snn', 'dt_snn']
    #input_w_grid = dat['input_w_grid']
    #input_rate_grid = dat['input_rate_grid']
    input_w = dat['input_w']#torch.logspace(-2,1,31, device=device)
    exc_rate = dat['exc_rate']
    inh_rate = dat['inh_rate']
    curr_mean = dat['curr_mean']
    curr_std = dat['curr_std']
    spk_count = dat['spk_count']
    config = dat['config'].item()
    rho = dat['rho']
    
    print(spk_count.sum())

    T = config['T_snn'] - config['discard']

    print('Calculating spike count stats...')
    snn_mean = np.mean(spk_count, axis=0)/T
    snn_std = np.std(spk_count, axis=0)/np.sqrt(T)
    
    print('Calculating moment activation...')
    mnn_mean, mnn_std =run_mnn(exp_id)
    # reshaping data
    
    try:
        size = (len(exc_rate), len(inh_rate), len(input_w))
        extent = (inh_rate[0],inh_rate[-1],exc_rate[0],exc_rate[-1])
    except:
        size = (len(curr_mean), len(curr_std), len(input_w))
        extent = (curr_std[0],curr_std[-1],curr_mean[0],curr_mean[-1])
        # exc_rate_grid = dat['exc_rate_grid']
        # inh_rate_grid = dat['inh_rate_grid']
    
    
    print(size)
    snn_mean = snn_mean.squeeze().reshape(size)
    snn_std = snn_std.squeeze().reshape(size)
    mnn_mean = mnn_mean.squeeze().reshape(size)
    mnn_std = mnn_std.squeeze().reshape(size)

    #exc_rate_grid = exc_rate_grid.squeeze().reshape(size)
    #inh_rate_grid = inh_rate_grid.squeeze().reshape(size)
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.imshow(exc_rate_grid[:,:,w_indx])
    # plt.colorbar()
    # plt.subplot(2,1,2)
    # plt.imshow(inh_rate_grid[:,:,w_indx])    
    # plt.colorbar()
    # plt.savefig('tmp.png')
    # plt.close('all')

    #print('snn_mean.mean: ', snn_mean.mean())
    #plt.plot(snn_mean.flatten(), '.', markersize=1.0)
    #plt.savefig('tmp3.png')

    if 0 in plot_ids:
        for w_indx in range(size[2]):
            plt.figure(figsize=(7,6))
            plt.subplot(2,2,1)
            for j in range(snn_std.shape[1]):
                plt.plot(snn_mean[:,j,w_indx])
            plt.title('SNN mean')
            
            plt.subplot(2,2,3)
            for j in range(snn_std.shape[1]):
                plt.plot(snn_std[:,j,w_indx])
            plt.title('SNN std')
            plt.tight_layout()
            plt.savefig(path+'snn_v_mnn_lineplot_{}.png'.format(w_indx), dpi=300)
            plt.close('all')
            

            plt.figure(figsize=(9,5))
            plt.subplot(2,3,1)
            plt.imshow(snn_mean[:,:,w_indx], vmin=0, vmax=mnn_mean.max(), origin='lower', extent=extent, aspect = 'auto')
            plt.colorbar()
            plt.title('SNN mean')
            plt.subplot(2,3,2)
            plt.imshow(mnn_mean[:,:,w_indx], vmin=0, vmax=mnn_mean.max(), origin='lower', extent=extent, aspect = 'auto')
            plt.colorbar()
            plt.title('MNN mean')
            plt.subplot(2,3,3)
            plt.imshow(np.abs(snn_mean[:,:,w_indx]-mnn_mean[:,:,w_indx]), origin='lower', extent=extent, aspect = 'auto')
            plt.colorbar()
            plt.title('Error')

            plt.subplot(2,3,4)
            plt.imshow(snn_std[:,:,w_indx], vmin=0, vmax=mnn_std.max(), origin='lower', extent=extent, aspect = 'auto')
            plt.colorbar()
            plt.title('SNN std')
            plt.subplot(2,3,5)
            plt.imshow(mnn_std[:,:,w_indx], vmin=0, vmax=mnn_std.max(), origin='lower', extent=extent, aspect = 'auto')
            plt.colorbar()
            plt.title('MNN std')
            plt.subplot(2,3,6)
            plt.imshow(np.abs(snn_std[:,:,w_indx]-mnn_std[:,:,w_indx]), origin='lower', extent=extent, aspect = 'auto')
            plt.colorbar()
            plt.title('Error')

            plt.tight_layout()
            plt.savefig(path+'snn_v_mnn_{}.png'.format(w_indx), dpi=300)
            plt.close('all')
    
    if 1 in plot_ids:
        extent = (curr_mean[0],curr_mean[-1],curr_std[0],curr_std[-1])
        vlim = 0.03
        plt.figure(figsize=(9,4))
        plt.subplot(2,5,1)
        plt.imshow(mnn_mean[:,:,0].T, extent=extent, origin='lower')
        for w_indx in range(size[2]):
            plt.subplot(2,5,w_indx+2)
            plt.imshow(snn_mean[:,:,w_indx].T-mnn_mean[:,:,w_indx].T, extent=extent, cmap='coolwarm',vmin=-vlim,vmax=vlim, origin='lower')
            plt.title('w = {}'.format(input_w[w_indx]))
        vlim = 0.06
        plt.subplot(2,5,6)
        plt.imshow(mnn_std[:,:,0].T, extent=extent, origin='lower')
        for w_indx in range(size[2]):
            plt.subplot(2,5,w_indx+7)
            plt.imshow(snn_std[:,:,w_indx].T-mnn_std[:,:,w_indx].T, extent=extent, cmap='coolwarm',vmin=-vlim,vmax=vlim, origin='lower')
            plt.title('w = {}'.format(input_w[w_indx]))
        plt.tight_layout()
        plt.savefig(path+'snn_vs_mnn_vary_w.png')
        plt.savefig(path+'snn_vs_mnn_vary_w.pdf', dpi=300, format='pdf')
        plt.close('all')

        # save colorbar as separate file
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(mnn_mean[:,:,-1])
        plt.colorbar()
        plt.subplot(2,2,2)
        vlim = 0.03
        plt.imshow(snn_mean[:,:,-1].T-mnn_mean[:,:,-1].T, vmin=-vlim, vmax=vlim, cmap='coolwarm')
        plt.colorbar()

        plt.subplot(2,2,3)
        plt.imshow(mnn_std[:,:,-1])
        plt.colorbar()
        plt.subplot(2,2,4)
        vlim = 0.06
        plt.imshow(snn_std[:,:,-1].T-mnn_std[:,:,-1].T, vmin=-vlim, vmax=vlim, cmap='coolwarm')
        plt.colorbar()
        plt.savefig(path+'snn_vs_mnn_vary_w_colorbars.png')
        plt.savefig(path+'snn_vs_mnn_vary_w_colorbars.pdf', dpi=300, format='pdf')
        plt.close('all')
    
    if 2 in plot_ids:
        plt.figure(figsize=(9,4))
        colors = iter(['#2c6aae', '#4e88ba', '#70a6c6', '#91c4d0', '#b2e2db']*10)
        for w_indx in range(size[2]):
            indxj = np.linspace(0,20,6, dtype='int')[1:]
            #indxj = np.linspace(0,10,6, dtype='int')[1:]
            input_std = np.linspace(0,5,21)
            print('input std: ', input_std[indxj])
            x_snn = np.linspace(-1,4,21)
            x_mnn = torch.linspace(-1,4,101)
            
            plt.subplot(2,4,w_indx+1)
            for j in indxj:
                mean_out, std_out = mnn_activate_no_rho(x_mnn, torch.ones(x_mnn.shape)*input_std[j])
                color = next(colors)
                plt.plot(x_mnn.numpy(), mean_out.numpy(), color=color)
                plt.plot(x_snn, snn_mean[:,j,w_indx],'.', markersize=2, color=color)
                plt.title('w = {}'.format(input_w[w_indx]))
            #if w_indx==0:
            #    plt.legend( ['$\sigma={}$'.format(s) for s in input_std[indxj]] )

                
            plt.subplot(2,4,w_indx+5)
            for j in indxj:
                mean_out, std_out = mnn_activate_no_rho(x_mnn, torch.ones(x_mnn.shape)*input_std[j])
                color = next(colors)
                plt.plot(x_mnn.numpy(), std_out.numpy(), color=color)
                plt.plot(x_snn, snn_std[:,j,w_indx],'.', markersize=2, color=color)
                #plt.title('w = {}'.format(input_w[w_indx]))
        plt.tight_layout()
        plt.savefig(path+'snn_v_mnn_lineplot_prod.png', dpi=300)
        plt.close('all')
    
    if 3 in plot_ids:
        plt.figure(figsize=(9,4))
        colors = iter(['#2c6aae', '#4e88ba', '#70a6c6', '#91c4d0', '#b2e2db']*10)
        w_indx = 1 # fix weight
        #indxj = np.linspace(0,20,6, dtype='int')[1:]
        indxj = np.linspace(0,10,6, dtype='int')[1:]
        input_std = np.linspace(0,5,21)
        print('input std: ', input_std[indxj])
        x_snn = np.linspace(-1,4,21)
        x_mnn = torch.linspace(-1,4,101)
        
        plt.figure(figsize=(7,3))
        plt.subplot(1,2,1)
        for j in indxj:
            mean_out, std_out = mnn_activate_no_rho(x_mnn, torch.ones(x_mnn.shape)*input_std[j])
            color = next(colors)
            plt.plot(x_mnn.numpy(), mean_out.numpy(), color=color)
        plt.legend( ['$\\bar{{\\sigma}}={}$'.format(s) for s in input_std[indxj]] , frameon=False)

        for j in indxj:
            color = next(colors)
            plt.plot(x_snn, snn_mean[:,j,w_indx],'.', markersize=3, color=color)
        
        plt.ylabel('Firing rate $\\mu$ (sp/ms)')
        plt.xlabel('Input current mean $\\bar{\\mu}$ (mV/ms)') 
            
        plt.subplot(1,2,2)
        for j in indxj:
            mean_out, std_out = mnn_activate_no_rho(x_mnn, torch.ones(x_mnn.shape)*input_std[j])
            color = next(colors)
            plt.plot(x_mnn.numpy(), std_out.numpy(), color=color)
            plt.plot(x_snn, snn_std[:,j,w_indx],'.', markersize=3, color=color)
            #plt.title('w = {}'.format(input_w[w_indx]))
        plt.ylabel('Firing variability $\\sigma$ (sp/ms$^{1/2}$)')  
        plt.xlabel('Input current std $\\bar{\\sigma}$ (mV/ms$^{1/2}$)') 
        
        
        plt.tight_layout()
        plt.savefig(path+'snn_v_mnn_lineplot_prod_w=0.1.png', dpi=300)
        plt.savefig(path+'snn_v_mnn_lineplot_prod_w=0.1.pdf', dpi=300, format='pdf')
        plt.close('all')

    return


def analyze_mean_var_combined(indx, savefile=False, savefig=True):
    ''' Production plot using 2024 Mar 19b and 2024 Apr 17'''
    indx = 0

    # load snn simulation with gaussian noise
    exp_id = '2024_mar_19b_smaller_dt'
    filename = str(indx).zfill(3) + '.npz'
    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    dat = np.load(path+filename, allow_pickle=True)
    
    spk_count = dat['spk_count']
    mean_grid = dat['mean_grid']
    std_grid = dat['std_grid']
    config = dat['config'].item()
    
    T = config['T_snn'] - config['discard']
    print('Calculating spike count stats...')
    snn_mean = np.mean(spk_count, axis=0)/T
    snn_std = np.std(spk_count,axis=0)/np.sqrt(T)
    snn_fano_factor = np.var(spk_count,axis=0)/(1e-6+np.mean(spk_count,axis=0))
    
    size = (51,5) 
    snn_mean_gaussian = snn_mean.reshape(size) #inner dim is std
    #snn_fano_factor = snn_fano_factor.reshape(size)
    snn_std_gaussian = snn_std.reshape(size)
    input_mean_gaussian = mean_grid.reshape(size)[:,0]
    input_std_gaussian = std_grid.reshape(size)[0,:]

    # load snn simulation with spiking input
    exp_id = '2024_Apr_17_w_rate'
    indx = 0 # dummy variable
    filename = str(indx).zfill(3) + '.npz'
    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    dat = np.load(path+filename, allow_pickle=True)
    
    input_w = dat['input_w']#torch.logspace(-2,1,31, device=device)
    exc_rate = dat['exc_rate']
    inh_rate = dat['inh_rate']
    curr_mean = dat['curr_mean']
    curr_std = dat['curr_std']
    spk_count = dat['spk_count']
    config = dat['config'].item()
    
    T = config['T_snn'] - config['discard']

    print('Calculating spike count stats...')
    snn_mean = np.mean(spk_count, axis=0)/T
    snn_std = np.std(spk_count, axis=0)/np.sqrt(T)
    
    try:
        size = (len(exc_rate), len(inh_rate), len(input_w))
        extent = (inh_rate[0],inh_rate[-1],exc_rate[0],exc_rate[-1])
    except:
        size = (len(curr_mean), len(curr_std), len(input_w))
        extent = (curr_std[0],curr_std[-1],curr_mean[0],curr_mean[-1])

    snn_mean_spk = snn_mean.squeeze().reshape(size)[:,:,1]
    snn_std_spk = snn_std.squeeze().reshape(size)[:,:,1]

    # PLOT routine
    colors = iter(['#2c6aae', '#4e88ba', '#70a6c6', '#91c4d0', '#b2e2db']*50)

    indxj = np.linspace(0,20,6, dtype='int')[1:]
    input_std_spk = np.linspace(0,5,21)
    print('input std: ', input_std_spk[indxj])
    x_snn = np.linspace(-1,4,21)
    x_mnn = torch.linspace(-1,4,101)
    
    plt.figure(figsize=(7,3))
    plt.subplot(1,2,1)
    for j in indxj:
        mean_out, std_out = mnn_activate_no_rho(x_mnn, torch.ones(x_mnn.shape)*input_std_spk[j])
        color = next(colors)
        plt.plot(x_mnn.numpy(), mean_out.numpy(), color=color)
    plt.legend( ['$\sigma={}$'.format(s) for s in input_std_spk[indxj]] )
    
    for j in indxj:
        color = next(colors)
        plt.plot(x_snn, snn_mean_spk[:,j],'.', markersize=2, color=color)

    for i in range(snn_mean_gaussian.shape[1]):
        color = next(colors)
        plt.plot(input_mean_gaussian, snn_mean_gaussian[:,i], '--', markersize=4, color=color)
    plt.ylabel('Firing rate $\\mu$ (sp/ms)')
    plt.xlabel('Input current mean $\\bar{\\mu}$ (mV/ms)') 

    plt.subplot(1,2,2)
    for j in indxj:
        mean_out, std_out = mnn_activate_no_rho(x_mnn, torch.ones(x_mnn.shape)*input_std_spk[j])
        color = next(colors)
        plt.plot(x_mnn.numpy(), std_out.numpy(), color=color)
        plt.plot(x_snn, snn_std_spk[:,j],'.', markersize=2, color=color)

    for i in range(snn_std_gaussian.shape[1]):
        color = next(colors)
        plt.plot(input_mean_gaussian, snn_std_gaussian[:,i], '--', color=color)
        plt.ylabel('Firing variability $\\sigma$ (sp/ms$^{1/2}$)')  
        plt.xlabel('Input current std $\\bar{\\sigma}$ (mV/ms$^{1/2}$)') 
    ###################
    plt.tight_layout()
    plt.savefig(path+'production_fig.png', dpi=300)
    plt.close('all')
    return 


if __name__=='__main__':
    torch.set_default_dtype(torch.float64)
    #exp_id = '2024_mar_19b'
    #exp_id = '2024_mar_21_debug_low_std_high_mean'
    #exp_id = '2024_mar_29_mean_std'

    # 1. plot mean-var results
    #exp_id = '2024_mar_30_mean_std'
    #analyze_mean_var(exp_id, savefile=False, savefig=False)
    
    # 2. plot fano factor results
    #exp_id = '2024_mar_30_mean_std'    
    #analyze_fano_factor(exp_id)

    # 3. plot w-rate results
    analyze_w_rate_EI(plot_ids=[1,3])
    #analyze_mean_var_combined(0)