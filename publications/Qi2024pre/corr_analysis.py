# analyzing corr

import numpy as np
from matplotlib import pyplot as plt
from mnn.mnn_core.mnn_pytorch import *
import os

def run_mnn(exp_id, indx):
    filename = str(indx).zfill(3) + '.npz'
    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    dat = np.load(path+filename, allow_pickle=True)
    #print(dat)

    #spk_count = dat['spk_count']
    mean_grid = torch.tensor(dat['mean_grid'])
    std_grid = torch.tensor(dat['std_grid'])
    config = dat['config'].item()    
    #print(config)
    N = config['NE']+config['NI']

    rho = dat['rho']
    input_corr = torch.ones(N,N)*rho
    input_corr.fill_diagonal_(1.0)
    input_corr = input_corr.unsqueeze(0) #add dummy batch dimension

    mean_out, std_out, corr_out = mnn_activate_trio(mean_grid, std_grid, input_corr)
    # TODO: this uses default neuronal parameters!

    return mean_out, std_out, corr_out

def analyze_corr(exp_id, indx, savefile=False, savefig=False):
    #exp_id = '2024_mar_13'
    print('Processing data ', indx)
    filename = str(indx).zfill(3) + '.npz'
    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    dat = np.load(path+filename, allow_pickle=True)
    #print(dat)

    n = int(dat['mean_grid'].shape[1]/2) #length of parameters
    print('length of parameters: ', n)

    spk_count = dat['spk_count']
     #remove duplicates
    mean_grid = dat['mean_grid'][:,n:]
    std_grid = dat['std_grid'][:,n:]
    config = dat['config'].item()
    rho = dat['rho']
    #print(config)

    T = config['T_snn'] - config['discard'] #ms

    print('Calculating spike count stats...')
    snn_rate = np.mean(spk_count[:,n:], axis=0)/T
    snn_std = np.std(spk_count[:,n:],axis=0)/np.sqrt(T)
    snn_fano_factor = np.var(spk_count[:,n:],axis=0)/(1e-12+np.mean(spk_count[:,n:],axis=0))
    snn_corr = np.corrcoef(spk_count.T, dtype = np.float128) # corr coef between columns
    snn_corr = snn_corr[:n,n:] # take off-diagonal block
    snn_corr = np.nan_to_num(snn_corr) # replace nan with zeros (e.g. no spikes)
    #np.fill_diagonal(snn_corr, 1.0) this is wrong
    
    print('Calculating moment activation...')
    mnn_mean, mnn_std, mnn_corr=run_mnn(exp_id, indx)
    mnn_mean = mnn_mean[:,n:] # remove duplicates
    mnn_std = mnn_std[:,n:]
    mnn_corr = mnn_corr[:,:n,n:] #take off-diagonal block

    mnn_FF = mnn_std.pow(2.0)/(1e-12+mnn_mean)    
    mnn_mean = mnn_mean.numpy()
    mnn_std = mnn_std.numpy()
    mnn_FF = mnn_FF.numpy()
    mnn_corr = mnn_corr.numpy()

    if savefig:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(snn_rate*1e3)
        plt.plot(mnn_mean.squeeze()*1e3,'--')
        plt.ylabel('Firing rate (sp/s)')
        plt.subplot(2,1,2)
        plt.plot(snn_fano_factor)
        plt.plot(mnn_FF.squeeze(),'--')
        plt.ylabel('Fano factor')
        plt.tight_layout()
        plt.savefig(path+str(indx).zfill(3)+'_snn_rate_FF.png')

        plt.figure()
        extent = (mean_grid[0,0], mean_grid[0,-1], mean_grid[0,-1], mean_grid[0,0])
        plt.imshow( snn_corr, vmin=-1,vmax=1, cmap='coolwarm', extent=extent) #snn_corr could have too many entries?
        plt.colorbar()
        plt.savefig(path+str(indx).zfill(3)+'_snn_corr.png')

        plt.close('all')
    return mnn_mean, mnn_std, mnn_corr, snn_rate, snn_std, snn_corr, mean_grid, std_grid,rho

if __name__=='__main__':
    #exp_id = '2024_mar_19'
    #num_files = 20
    #exp_id = '2024_mar_20_neg_corr'
    exp_id = '2024_mar_21_longer_T'
    num_files = 39

    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    savefig=True
    # mnn_mean, mnn_std, mnn_corr=run_mnn(exp_id, 10)
    # print(mnn_mean.numpy().squeeze())
    # plt.figure()
    # plt.plot(mnn_mean.numpy().squeeze())
    # plt.savefig('testtest.png')
    # plt.close('all')
    #print(np.max(mnn_mean.squeeze()))
    #analyze_corr(exp_id, indx, savefile=False, savefig=True)
    corr_err_L2 = 0
    corr_err_cum = 0
    corr_array = []

    mnn_corr_5d = np.zeros((110,110,num_files))
    snn_corr_5d = np.zeros((110,110,num_files))

    for indx in range(num_files):
        # try to load existing result, 
        #if False:
        if os.path.exists(path+str(indx).zfill(3)+'post_analysis.npz'):
            print('Existing analysis result found! Loading existing results...')
            dat = np.load(path+str(indx).zfill(3)+'post_analysis.npz')
            mnn_mean = dat['mnn_mean']
            mnn_std = dat['mnn_std']
            mnn_corr = dat['mnn_corr']
            snn_rate = dat['snn_rate']
            snn_std = dat['snn_std']
            snn_corr = dat['snn_corr']
            mean_grid = dat['mean_grid']
            std_grid = dat['std_grid']
            rho = dat['rho']
        else:
            mnn_mean, mnn_std, mnn_corr, snn_rate, snn_std, snn_corr, mean_grid, std_grid,rho = analyze_corr(exp_id, indx, savefile=False, savefig=savefig)
            dat_dict = {'mnn_mean':mnn_mean,
                    'mnn_std':mnn_std,
                    'mnn_corr':mnn_corr,
                    'snn_rate':snn_rate,
                    'snn_std':snn_std,
                    'snn_corr':snn_corr,
                    'mean_grid':mean_grid,
                    'std_grid':std_grid,
                    'rho':rho}
            np.savez(path+str(indx).zfill(3)+'post_analysis.npz', **dat_dict)
        
        # aggregate results into a single array for further analysis
        mnn_corr_5d[:,:,indx] = mnn_corr
        snn_corr_5d[:,:,indx] = snn_corr

        corr_err_L2 += np.power(mnn_corr-snn_corr, 2.0)
        corr_err_cum += snn_corr- mnn_corr
        corr_array.append(rho)

        if np.abs(rho)<1e-6: # calculate error in mean/std stats
            
            mean_err = mnn_mean-snn_rate
            std_err = mnn_std-snn_std

            plt.figure()
            mean_err = mean_err.reshape(11,10) #remember inner dim is std
            plt.imshow(mean_err, origin='lower')
            plt.colorbar()
            plt.xlabel('Current std')
            plt.ylabel('Current mean')
            plt.tight_layout()
            plt.savefig(path+'mean_err.png', dpi=300)
            #plt.savefig('corr_err_L2.pdf', dpi=300, format='pdf')

            plt.figure()
            std_err = std_err.reshape(11,10)
            plt.imshow(std_err, origin='lower')
            plt.colorbar()
            plt.xlabel('Current std')
            plt.ylabel('Current mean')
            plt.tight_layout()
            plt.savefig(path+'std_err.png', dpi=300)

            plt.figure()
            tmp_mnn_mean = mnn_mean.reshape(11,10)
            tmp_snn_rate = snn_rate.reshape(11,10)
            input_mean = mean_grid.reshape(11,10)[:,0]
            for i in range(tmp_mnn_mean.shape[1]): #iterate thru stds
                plt.plot(input_mean,tmp_mnn_mean[:,i])
                plt.plot(input_mean,tmp_snn_rate[:,i],'.')                
                plt.xlabel('Current mean')
                plt.ylabel('Mean firing rate (sp/ms)')
                plt.tight_layout()
                plt.savefig(path+'mean_firing_rate.png', dpi=300)
            
            plt.figure()
            tmp_mnn_std = mnn_std.reshape(11,10)
            tmp_snn_std = snn_std.reshape(11,10)      
            for i in range(tmp_mnn_std.shape[1]): #iterate thru stds
                plt.plot(input_mean,tmp_mnn_std[:,i])
                plt.plot(input_mean,tmp_snn_std[:,i],'.')                
                plt.xlabel('Current mean')
                plt.ylabel('Firing variability (sp/ms$^{1/2}$)')
                plt.tight_layout()
                plt.savefig(path+'firing_variability.png', dpi=300)
                


    corr_array = np.array(corr_array)
    dr = corr_array[1]-corr_array[0]
    corr_err_L2 = np.sqrt(corr_err_L2.squeeze()*dr) # calculate L2 norm
    corr_err_cum = corr_err_cum.squeeze()*dr

    #print(np.where(corr_err_L2[:20,:20]==corr_err_L2[:20,:20].max())) #print outlier
    corr_err_L2[4,13]=corr_err_L2[13,4] # replace outlier
    # save aggregated data for further analysis
    agg_dat = {        
        'mnn_corr_5d':mnn_corr_5d,
        'snn_corr_5d':snn_corr_5d,
        'mean_grid':mean_grid,
        'std_grid':std_grid,
        'corr_array':corr_array,
        'corr_err_L2':corr_err_L2,
        'corr_err_cum':corr_err_cum,
    }
    np.savez(path+'aggregated_data.npz', **agg_dat)
    # try swap inner and outer coordinates
    # a = np.arange(snn_std.shape[0])
    # a = a.reshape(11,10) #inner dim: std, outer dim: mean
    # a = a.T.flatten()
    # corr_err = corr_err[a,:][:,a] # permute column and rows
    # doesn't look as good
    
    '''Plot (std,std) x (mean,mean) '''
    plt.figure()
    # plot the axis separately
    #extent = (mean_grid[0,0], mean_grid[0,-1], mean_grid[0,-1], mean_grid[0,0])
    plt.imshow(corr_err_L2, vmin=0,vmax=0.25, origin='upper')#, extent=extent)    
    # plot boundaries
    x = 0
    y = [0, snn_std.shape[0]]
    while x<snn_std.shape[0]:
        x+=10        
        plt.plot([x,x],y,'white')
        plt.plot(y,[x,x],'white')
    plt.xlim([0,snn_std.shape[0]])
    plt.ylim([0,snn_std.shape[0]])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig(path+'corr_err_L2.png', dpi=300)
    plt.savefig(path+'corr_err_L2.pdf', dpi=300, format='pdf')
    plt.close('all')

    '''Plot (mean,std) x (mean,std) '''
    plt.figure()
    #if exp_id == '2024_mar_19':
    size = (11,10) # mean x std
    perm_indx = np.arange(110)
    perm_indx = np.reshape(perm_indx,size)
    perm_indx = perm_indx.T.flatten()
    print(perm_indx)
    corr_err_L2_permuted = corr_err_L2[perm_indx,:] #swap vertical inner/outer dim   
    plt.imshow(corr_err_L2_permuted, vmin=0,vmax=0.25, origin='upper')#, extent=extent)    
    # plot boundaries
    x = 0
    y = [0, snn_rate.shape[0]]
    while x<snn_rate.shape[0]:
        x+=11    
        plt.plot(y,[x,x],'white')
    plt.xlim([0,snn_rate.shape[0]])    
    x = 0
    y = [0, snn_std.shape[0]]
    while x<snn_std.shape[0]:
        x+=10        
        plt.plot([x,x],y,'white')        
    plt.ylim([0,snn_std.shape[0]])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig(path+'corr_err_L2_permuted.png', dpi=300)
    #plt.savefig(path+'corr_err_L2.pdf', dpi=300, format='pdf')
    plt.close('all')


    plt.figure()
    # plot the axis separately
    #extent = (mean_grid[0,0], mean_grid[0,-1], mean_grid[0,-1], mean_grid[0,0])
    plt.imshow(corr_err_cum, vmin=-0.2, vmax=0.2, origin='upper', cmap='coolwarm')#, extent=extent)    
    # plot boundaries
    x = 0
    y = [0, snn_std.shape[0]]
    while x<snn_std.shape[0]:
        x+=10        
        plt.plot([x,x],y,'white')
        plt.plot(y,[x,x],'white')
    plt.xlim([0,snn_std.shape[0]])
    plt.ylim([0,snn_std.shape[0]])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.savefig(path+'corr_err_cum.png', dpi=300)
    plt.close('all')
    
       
    
