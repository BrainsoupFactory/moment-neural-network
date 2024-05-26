import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from projects.pre2023.rec_snn_simulator_torch import *
import os

'''
Temporary code snipets for inspecting results from recurrent mnn with correlation
'''


def inspect_raw_data(exp_id, indx):
    ''' inspecting raw data and generate temporary pre-production plots
    '''

    path = './projects/pre2023/runs/{}/'.format(exp_id)

    dat = np.load(path + str(indx).zfill(3)+'.npz', allow_pickle=True)
    config = dat['config'].item()
    #['mnn_mean', 'mnn_std', 'mnn_corr', 'config', 'indx', 'exp_id', 'W', 
    #'ie_ratio_array', 'input_rate_array', 'input_mean', 'input_var']

    C = dat['mnn_corr']

    input_rate_array = dat['input_rate_array'].flatten()
    ie_ratio_array = dat['ie_ratio_array'].flatten()
    avg_corr = np.zeros((C.shape[0],1))
    max_corr = np.zeros((C.shape[0],1))
    min_corr = np.zeros((C.shape[0],1))

    for i in range(C.shape[0]):
        #np.triu(C[i,:,:].squeeze())
        cc = C[i,:,:].squeeze()
        U = np.triu_indices(C.shape[1], k=1)
        avg_corr[i] = cc[U].mean()
        max_corr[i] = cc[U].max()
        min_corr[i] = cc[U].min()

        if i==20: # plot a representative example of correlation matrix
            plt.figure(figsize=(3.5,3))
            plt.imshow(cc, cmap='coolwarm', vmin=-1,vmax=1)
            plt.title('rate={}, ie_ratio={}'.format(input_rate_array[i], ie_ratio_array[indx]))
            plt.colorbar()
            plt.tight_layout()
            plt.savefig('tmp_corr.png')
            
            plt.figure(figsize=(3.5,3))
            NE = config['NE']
            NI = config['NI']
            cc_EE = cc[:NE,:NE][np.triu_indices(NE, k=1)]
            cc_EI = cc[NE:,:NE].flatten()
            cc_II = cc[NE:,NE:][np.triu_indices(NI, k=1)]
            
            plt.hist(cc_EE, 50)
            plt.hist(cc_EI, 50)   
            plt.hist(cc_II, 50)   
            
            plt.xlabel('spike count corr.')
            plt.ylabel('#neural pairs')
            plt.title('rate={}, ie_ratio={}'.format(input_rate_array[i], ie_ratio_array[indx]))
            plt.tight_layout()
            plt.savefig('tmp_corr_hist.png')
            
            plt.close('all')

    plt.figure(figsize=(3.5,3))
    plt.plot(input_rate_array, avg_corr)
    plt.plot(input_rate_array, max_corr)
    plt.plot(input_rate_array, min_corr)
    plt.xlabel('input rate (sp/ms)')
    plt.ylabel('Spike count correlation')
    plt.title('IE ratio = {}'.format( ie_ratio_array.flatten()[indx] ))
    plt.tight_layout()
    plt.savefig('tmp.png')
    plt.close('all')

    return 


def plot_corr_hist_all(exp_id):
    ''' Plot correlation histogram for all parameters
    Take quite some time...
    '''
    path = './projects/pre2023/runs/{}/'.format(exp_id)
    
    plt.figure(figsize=(10,10))
    for indx in range(41):        
        dat = np.load(path + str(indx).zfill(3)+'.npz', allow_pickle=True)
        config = dat['config'].item()
        #['mnn_mean', 'mnn_std', 'mnn_corr', 'config', 'indx', 'exp_id', 'W', 
        #'ie_ratio_array', 'input_rate_array', 'input_mean', 'input_var']

        C = dat['mnn_corr']
        
        input_rate_array = dat['input_rate_array'].flatten()
        ie_ratio_array = dat['ie_ratio_array'].flatten()
        
        
        for i in range(C.shape[0]):
            print('Plotting for rate={}, ie_ratio={}...'.format(input_rate_array[i], ie_ratio_array[indx]))
            cc = C[i,:,:].squeeze()
            U = np.triu_indices(C.shape[1], k=1)
            
            plt.subplot(41,41,i*41+indx+1) # horizontal ie_ratio, vertical input rate
            plt.hist(cc[U], np.linspace(-1,1,20))
            plt.xticks([])
            plt.yticks([])            
            
    plt.tight_layout(pad=0.1)
    plt.savefig(path+'corr_hist_all.pdf', format='pdf')
    plt.close('all')


def plot_heatmaps(exp_id):
    ''' Plot 2d heatmaps of firing stats
    '''
    path = './projects/pre2023/runs/{}/'.format(exp_id)
    file = path+'slice_data.npz'
    if os.path.exists(file):
        print('Loading cached data...')
        dat = np.load(file, allow_pickle=True)
        avg_rate = dat['avg_rate']
        avg_FF = dat['avg_FF']
        avg_corr = dat['avg_corr']
        div_rate = dat['div_rate']
        div_FF = dat['div_FF'] 
        div_corr = dat['div_corr']
        input_rate_array = dat['input_rate_array']
        ie_ratio_array = dat['ie_ratio_array']
    else:
        print('No data found! Run plot_slices first!')
        return
            
    extent=(ie_ratio_array[0],ie_ratio_array[-1],input_rate_array[0],input_rate_array[-1])
    plt.figure(figsize=(10.5,6))
    plt.subplot(2,3,1)
    plt.imshow(avg_rate, origin='lower', extent=extent, aspect='auto')
    plt.xlabel('IE weight ratio')
    plt.ylabel('Total input rate (sp/ms)')
    plt.title('Pop. avg. firing rate')
    plt.colorbar()
    
    plt.subplot(2,3,2)
    avg_FF[np.isnan(avg_FF)]=1
    plt.imshow(avg_FF, origin='lower', extent=extent, cmap='plasma', aspect='auto')
    plt.xlabel('IE weight ratio')
    plt.ylabel('Total input rate (sp/ms)')
    plt.title('Pop. avg. Fano factor')
    plt.colorbar()
    
    plt.subplot(2,3,3)
    plt.imshow(avg_corr, origin='lower', vmin=-1,vmax=1, cmap='coolwarm', extent=extent, aspect='auto')
    plt.xlabel('IE weight ratio')
    plt.ylabel('Total input rate (sp/ms)')
    plt.title('Pop. avg. correlation')
    plt.colorbar()
    
    plt.subplot(2,3,4)
    plt.imshow(div_rate, origin='lower', extent=extent, aspect='auto')
    plt.xlabel('IE weight ratio')
    plt.ylabel('Total input rate (sp/ms)')
    plt.title('Firing rate div.')
    plt.colorbar()
    
    plt.subplot(2,3,5)
    div_FF[np.isnan(div_FF)]=0
    plt.imshow(div_FF, origin='lower', extent=extent, cmap='plasma', aspect='auto')
    plt.xlabel('IE weight ratio')
    plt.ylabel('Total input rate (sp/ms)')
    plt.title('Fano factor div.')
    plt.colorbar()

    plt.subplot(2,3,6)
    plt.imshow(div_corr, origin='lower', vmin=-0.2,vmax=0.2, cmap='coolwarm', extent=extent, aspect='auto')
    plt.xlabel('IE weight ratio')
    plt.ylabel('Total input rate (sp/ms)')
    plt.title('Correlation div.')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(path+'map_stats.pdf', format='pdf')
    plt.close('all')    

def plot_slices(exp_id, slice_i=10):
    ''' Plot results along specific slices'''
    path = './projects/pre2023/runs/{}/'.format(exp_id)
    file = path+'slice_data.npz'
    if os.path.exists(file):
        print('Loading cached data...')
        dat = np.load(file, allow_pickle=True)
        avg_rate = dat['avg_rate']
        avg_FF = dat['avg_FF']
        avg_corr = dat['avg_corr']
        div_rate = dat['div_rate']
        div_FF = dat['div_FF'] 
        div_corr = dat['div_corr']
        input_rate_array = dat['input_rate_array']
        ie_ratio_array = dat['ie_ratio_array']
    else:    
        avg_rate = np.zeros((41,41))
        avg_FF = np.zeros((41,41))
        avg_corr = np.zeros((41,41)) #input rate vs ie_ratio

        div_rate = avg_rate.copy() # diversity in firing stats
        div_FF = avg_rate.copy()
        div_corr = avg_rate.copy() 

        for indx in range(41):        
            dat = np.load(path + str(indx).zfill(3)+'.npz', allow_pickle=True)
            #config = dat['config'].item()
            #['mnn_mean', 'mnn_std', 'mnn_corr', 'config', 'indx', 'exp_id', 'W', 
            #'ie_ratio_array', 'input_rate_array', 'input_mean', 'input_var']

            C = dat['mnn_corr']
            
            avg_rate[:,indx] = dat['mnn_mean'].mean(axis=1)
            avg_FF[:,indx] = (dat['mnn_std']*dat['mnn_std']/dat['mnn_mean']).mean(axis=1)
            div_rate[:,indx] = dat['mnn_mean'].std(axis=1)
            div_FF[:,indx] = (dat['mnn_std']*dat['mnn_std']/dat['mnn_mean']).std(axis=1)

            input_rate_array = dat['input_rate_array'].flatten()
            ie_ratio_array = dat['ie_ratio_array'].flatten()
            
            for i in range(C.shape[0]):
                cc = C[i,:,:].squeeze()
                U = np.triu_indices(C.shape[1], k=1)
                avg_corr[i,indx] = cc[U].mean()
                div_corr[i,indx] = cc[U].std()

        dat = {
            'avg_rate' : avg_rate,
            'avg_FF' : avg_FF,
            'avg_corr' : avg_corr,
            'div_rate' : div_rate,
            'div_FF' : div_FF,
            'div_corr' : div_corr,
            'input_rate_array' : input_rate_array,
            'ie_ratio_array' : ie_ratio_array,
        }            
        np.savez(file,**dat)
    
    crit_pts = [3.4, 6.4]
    plt.figure(figsize=(9.5,3))    
    #
    plt.subplot(1,3,1)
    plt.fill_between(ie_ratio_array, avg_rate[slice_i, :]-div_rate[slice_i, :], avg_rate[slice_i, :]+div_rate[slice_i, :], alpha=0.3)
    plt.plot(ie_ratio_array, avg_rate[slice_i, :])
    plt.ylabel('Pop. avg. firing rate (sp/ms)')
    plt.xlabel('IE weight ratio')
    for p in crit_pts:
        plt.plot([p,p],[-0.1,0.2],'--', color='gray')
    plt.ylim([-0.02,0.2])
    plt.title('Input rate = {} (sp/ms)'.format(input_rate_array[slice_i]))
    #
    plt.subplot(1,3,2)
    plt.fill_between(ie_ratio_array, avg_FF[slice_i, :]-div_FF[slice_i, :], avg_FF[slice_i, :]+div_FF[slice_i, :], alpha=0.3)
    plt.plot(ie_ratio_array, avg_FF[slice_i, :])
    plt.ylabel('Pop. avg. Fano factor')
    plt.xlabel('IE weight ratio')
    for p in crit_pts:
        plt.plot([p,p],[-0.1,3],'--', color='gray')
    plt.ylim([-0.1,3])
    #
    plt.subplot(1,3,3)
    plt.fill_between(ie_ratio_array, avg_corr[slice_i, :]-div_corr[slice_i, :], avg_corr[slice_i, :]+div_corr[slice_i, :], alpha=0.3)
    plt.plot(ie_ratio_array, avg_corr[slice_i, :])
    plt.ylabel('Pop. avg. correlation')
    plt.xlabel('IE weight ratio')
    for p in crit_pts:
        plt.plot([p,p],[-0.1,3],'--', color='gray')
    plt.ylim([-0.02,1])

    #plt.title('Input rate = {}'.format(input_rate_array[slice_i]))
    plt.tight_layout()
    plt.savefig(path+'slice_{}.pdf'.format(slice_i),format='pdf')
    plt.close('all')



def plot_snn_vs_mnn(exp_id):
    ''' Plot results along specific slices'''
    
    print('Loading cached MNN results...')
    mnn_exp_id = '2024_May_15_rec_mnn'
    file = './projects/pre2023/runs/{}/slice_data.npz'.format(mnn_exp_id)    
    dat = np.load(file, allow_pickle=True)
    avg_rate = dat['avg_rate']
    avg_FF = dat['avg_FF']
    avg_corr = dat['avg_corr']
    div_rate = dat['div_rate']
    div_FF = dat['div_FF'] 
    div_corr = dat['div_corr']
    input_rate_array = dat['input_rate_array']
    ie_ratio_array = dat['ie_ratio_array']

    print('Analyzing SNN results...')
    path = './projects/pre2023/runs/{}/'.format(exp_id)
    meta = np.load(path+'meta_data.npz', allow_pickle=True)
    #['exp_id', 'uext_array', 'ie_ratio_array']
    snn_uext = meta['uext_array']
    snn_ie_ratio = meta['ie_ratio_array']
    
    shape = (len(snn_uext), len(snn_ie_ratio))

    batchsize = 1000
    num_neurons = 2500
    
    snn_rate = np.zeros((shape[0],shape[1], num_neurons))
    snn_var = snn_rate.copy()
    snn_corr = np.zeros(shape) 

    for indx in range(shape[0]*shape[1]):
        #print('processing file: ', indx)
        i,j = np.unravel_index(indx, shape)
        dat = np.load(path+str(indx).zfill(3)+'.npz', allow_pickle=True)
        config=dat['config'].item()
        #['config', 'spk_history', 't']
        spk_count = dat['spk_count'] # spk indx x 3; [trial id, neuron id, spk time]
        T = config['T_snn']-config['discard']
        snn_rate[i,j,:] = spk_count.mean(axis=0).flatten()/T
        snn_var[i,j,:] = spk_count.var(axis=0).flatten()/T
        rho = np.corrcoef(spk_count.T)  #spk_count shape = batch x neurons
        snn_corr[i,j] = rho[np.triu_indices( rho.shape[0] , k=1)].mean()
    print('rho shape:', rho.shape)

    # PLOT RESULTS
    # PLOT 1: mean firing rate
    # plot snn result
    plt.figure(figsize=(3.5,3))
    plt.plot(snn_ie_ratio, snn_rate.mean(axis=-1).T, color='gray')
    
    # plot mnn result
    for j in [10,20,30]:
        plt.plot(ie_ratio_array, avg_rate[j,:], '--k')
    plt.xlabel('IE weight ratio')
    plt.ylabel('Pop. avg. firing rate (sp/ms)')
    plt.tight_layout()
    plt.savefig(path+'slice_mnn_vs_snn.png')
    plt.close('all')

    # Plot 2: correlation coef
    plt.figure(figsize=(3.5,3))
    plt.plot(snn_ie_ratio, snn_corr.T, color='gray')
    
    # plot mnn result
    for j in [10,20,30]:
        plt.plot(ie_ratio_array, avg_corr[j,:], '--k')
    plt.xlabel('IE weight ratio')
    plt.ylabel('Corr coef')
    plt.tight_layout()
    plt.savefig(path+'slice_mnn_vs_snn_corr.png')
    plt.close('all')



def temp_inspect_raw_data():
    ''' check time course of spike data to make sure it works as intended'''
    exp_id = 'test'
    indx = 40
    path = './projects/pre2023/runs/{}/'.format(exp_id)

    dat = np.load(path + str(indx).zfill(3)+'.npz', allow_pickle=True)
    config = dat['config'].item()
    # ['config', 't', 'spk_count', 'spk_history']
    spk_history = dat['spk_history']
    #spk_count = spk_time2count(dat['spk_history'], 10, config)
    nneurons = config['NE']+config['NI']
    nsteps = int(config['T_snn']/config['dt_snn'])
    spk_mat = spk_time2csr(spk_history, nneurons, nsteps, sample_id = 0)
    print(spk_mat.shape)
    
    dT = 10
    binsize = int(dT/config['dt_snn'])
    nbins = int(nsteps/binsize)
    spk_count = np.zeros((nneurons, nbins))
    for i in range(nneurons):
        spk_count[i,:] = spk_mat[i,:].todense().reshape(nbins, binsize ).sum(axis=1).flatten()
    print(spk_count.shape)

    t = np.linspace(0,config['T_snn'], nbins)

    pop_rate = spk_count.mean(axis=0)/dT 
    print('Final rate is {} sp/ms'.format( pop_rate[t>1500].mean().round(4) ))
    
    plt.figure(figsize=(7,2.5))
    plt.plot(t, pop_rate)
    plt.ylabel('Pop avg firing rate (sp/ms)')
    plt.xlabel('Time (ms)')
    plt.tight_layout()
    plt.savefig(path+ str(indx).zfill(3)+'_spk_count_time_series.png')

    return
                    
def inspect_snn_oscillation(exp_id='2024_May_20_rec_snn_delay'):
    path = './projects/pre2023/runs/{}/'.format(exp_id)
    meta = np.load(path+'meta_data.npz', allow_pickle=True)
    print(list(meta))
    #['exp_id', 'uext_array', 'ie_ratio_array', 'delay_array']
    snn_uext = meta['uext_array']
    snn_ie_ratio = meta['ie_ratio_array']
    snn_delay = meta['delay_array']
    shape = (len(snn_uext), len(snn_ie_ratio),len(snn_delay))

    sum_data_file = path+'psd_analysis.npz'
    if os.path.exists(sum_data_file):
        print('Loading cached data...')
        sum_data = np.load(sum_data_file)
        PeakFreq = sum_data['PeakFreq']
        PeakPower = sum_data['PeakPower']
        snn_uext = sum_data['snn_uext']
        snn_ie_ratio = sum_data['snn_ie_ratio']
        snn_delay = sum_data['snn_delay']
    else:        
        print('Aggregating data and plotting psd for all trials...')   
        PeakFreq = np.zeros(shape)
        PeakPower = np.zeros(shape)

        plt.figure(figsize=(5,12))
        for indx in range(np.prod(shape)):        
            i,j,k = np.unravel_index(indx, shape )
            print('Processing data indx={}, i={}, j={}, k={}'.format(indx, i, j, k))

            dat = np.load(path+str(indx).zfill(3)+'.npz', allow_pickle=True)
            config = dat['config'].item()
            #print(list(dat))
            #['config', 't', 'spk_count', 'pop_spk_count']

            x = dat['pop_spk_count']    
            pts_per_segment = int( 1000/config['dt_snn'] ) # split to 1000 ms windows
            psd = 0
            for ii in range(x.shape[0]):
                freq, psd_tmp = sp.signal.welch(x[ii,:], fs=1/config['dt_snn'], nperseg= pts_per_segment)
                psd += psd_tmp
            psd = psd/x.shape[0]
            peak_indx = np.argmax(psd)
            peak_freq = freq[peak_indx].round(2)
            peak_pow = psd[peak_indx]
            
            PeakFreq[i,j,k] = peak_freq
            PeakPower[i,j,k] = peak_pow
            #down_sample = 1000 # downsample for plotting purpose
            #df = int(len(psd)/down_sample) #<-- this is bad; must do this on log bin        
            plt.subplot(shape[-1], 4, 4*k + 2*i+j+1) 
            plt.loglog(freq, psd)
            plt.ylim([1e-2,1e4]) # fix the range of power
            plt.title('fmax={} kHz'.format(peak_freq))
            #plt.xlabel('Osc. frequency (kHz)')
            #plt.ylabel('PSD')
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout(pad=0.1)    
        plt.savefig(path+'psd_all.png',dpi=200) #too many points. save to png only

        sum_data ={
            'PeakFreq':PeakFreq,
            'PeakPower':PeakPower,
            'snn_uext' :snn_uext,
            'snn_ie_ratio':snn_ie_ratio,
            'snn_delay':snn_delay,
        }
        np.savez(sum_data_file, **sum_data)
    
    # plot aggregated data
    

    plt.figure(figsize=(7,6))    
    for i in range(shape[0]):
        for j in range(shape[1]):
            snn_uext = sum_data['snn_uext'][i]
            snn_ie_ratio = sum_data['snn_ie_ratio'][j]
            plt.subplot(2,2, 2*i+j+1)
            plt.plot(sum_data['snn_delay'], PeakFreq[i,j,:] ,'.')
            plt.xlabel('Delay (ms)')
            plt.ylabel('Osc. frequency (kHz)')
            plt.title('input={}, ie_ratio={}'.format(snn_uext, snn_ie_ratio))
    plt.tight_layout()
    plt.savefig(path+'snn_freq_vs_delay.pdf', format='pdf')

    plt.figure(figsize=(7,6))    
    for i in range(shape[0]):
        for j in range(shape[1]):
            snn_uext = sum_data['snn_uext'][i]
            snn_ie_ratio = sum_data['snn_ie_ratio'][j]
            plt.subplot(2,2, 2*i+j+1)
            plt.semilogy(sum_data['snn_delay'], PeakPower[i,j,:] ,'.')
            plt.xlabel('Delay (ms)')
            plt.ylabel('Peak power density')
            plt.title('input={}, ie_ratio={}'.format(snn_uext, snn_ie_ratio))
    plt.tight_layout()
    plt.savefig(path+'snn_power_vs_delay.pdf', format='pdf')

    # finally fit the coefficients
    #kk = sum_data['snn_delay'] > 1.0 # keep only delay values larger then 1.0 ms?
    cut_off = 5 # below this freq vs 1/delay is quite nonlinear
    print('Omit delay < {} ms for fitting snn'.format(sum_data['snn_delay'][cut_off]) )
    plt.figure(figsize=(7,6))    
    for i in range(shape[0]):
        for j in range(shape[1]):
            snn_uext = sum_data['snn_uext'][i]
            snn_ie_ratio = sum_data['snn_ie_ratio'][j]
            x = 1/sum_data['snn_delay'][cut_off:]
            y = PeakFreq[i,j,:][cut_off:]
            #coefs = np.polyfit(x,y,1)   
            #print(coefs)         
            a, _, _, _ = np.linalg.lstsq(x[:,np.newaxis], y)
            plt.subplot(2,2, 2*i+j+1)
            #plt.plot(x, coefs[0]*x+coefs[1], color='gray')
            plt.plot(x, a*x, color='gray')
            plt.plot(x, y ,'.')            
            plt.legend(['slope={}'.format(np.round(a[0],4))])
            plt.xlabel('1/delay (ms)')
            plt.ylabel('Osc. frequency (kHz)')
            plt.title('input={}, ie_ratio={}'.format(snn_uext, snn_ie_ratio))
    plt.tight_layout()
    plt.savefig(path+'snn_fit_freq_delay.pdf', format='pdf')


    return

def inspect_mnn_oscillation(exp_id='2024_May_21_rec_mnn_longer'):
    path = './projects/pre2023/runs/{}/'.format(exp_id)
    
    
    if exp_id=='2024_May_21_rec_mnn':
        input_rate_array = np.array([20,40])    
        ie_ratio_array = np.array([6.0,8.0])   
        mnn_delay_array = np.linspace(0,1.5,16)
        X,Y = np.meshgrid(ie_ratio_array, mnn_delay_array)
        X=X.flatten()
        Y=Y.flatten()
        cut_off = 5
    elif exp_id == '2024_May_21_rec_mnn_longer':
        input_rate_array = np.array([20,40])    
        ie_ratio_array = np.array([6.0,8.0])   
        mnn_delay_array = np.linspace(0,3.0,16)
        X,Y = np.meshgrid(ie_ratio_array, mnn_delay_array)
        X=X.flatten()
        Y=Y.flatten()
        cut_off = 4
    

    shape = (len(input_rate_array), len(ie_ratio_array),len(mnn_delay_array))

    sum_data_file = path+'psd_analysis.npz'
    if os.path.exists(sum_data_file):
        print('Loading cached data...')
        sum_data = np.load(sum_data_file)
        PeakFreq = sum_data['PeakFreq']
        PeakPower = sum_data['PeakPower']
        input_rate_array = sum_data['input_rate_array']
        ie_ratio_array = sum_data['ie_ratio_array']
        mnn_delay_array = sum_data['mnn_delay_array']
    else:        
        print('Aggregating data and plotting psd for all trials...')   
        PeakFreq = np.zeros(shape)
        PeakPower = np.zeros(shape)

        plt.figure(figsize=(5,12))
        for indx in range( shape[1]*shape[2] ):
            j,k = np.unravel_index(indx, shape[1:] , order='F')
            print('Processing data indx={}, j={}, k={}'.format(indx, j, k))
            
            ie_ratio = ie_ratio_array[j]
            mnn_delay = mnn_delay_array[k]

            # check ordering is consistent...
            print('{} vs {}'.format(ie_ratio,  X[indx]))
            print('{} vs {}'.format(mnn_delay,  Y[indx]))
            
            
            dat = np.load(path+str(indx).zfill(3)+'.npz', allow_pickle=True)
            config = dat['config'].item()
            #print(list(dat))
            #['mnn_mean', 'mnn_std', 'mnn_mean_ts', 'mnn_std_ts', 'pop_mean', 'pop_std', 'config',
            # 'indx', 'exp_id', 'W', 'ie_ratio_array', 'input_rate_array', 'mnn_delay_array', 'input_mean', 'input_var']

            discard = int(10/config['dt'])
            x = dat['pop_mean'][:,discard:]
            
            pts_per_segment = x.shape[1]

            for i in range(x.shape[0]):
                #freq, psd = sp.signal.welch(x[i,:].flatten(), fs=1/config['dt'], nperseg= pts_per_segment)
                #<-- this method does some sort of smoothing... no good for deterministic data

                # from MATLAB:
                # N = length(x);
                # xdft = fft(x);
                # xdft = xdft(1:N/2+1);
                # psdx = (1/(fs*N)) * abs(xdft).^2;
                # psdx(2:end-1) = 2*psdx(2:end-1);
                # freq = 0:fs/length(x):fs/2;

                fs = 1/config['dt']
                N = x.shape[1]
                psd = np.fft.rfft(x[i,:].flatten())
                psd = psd[1:int(N/2)+1]

                psd = (1/(fs*N))*(np.abs(psd)**2)
                psd[1:-1] = 2*psd[1:-1]
                freq = np.linspace(0, fs/2, len(psd))
                
                peak_indx = np.argmax(psd)
                peak_freq = freq[peak_indx]
                peak_pow = psd[peak_indx]
            
                PeakFreq[i,j,k] = peak_freq
                PeakPower[i,j,k] = peak_pow
            
                plt.subplot(shape[-1], 4, 4*k + 2*i+j+1) 
                plt.loglog(freq, psd)
                plt.ylim([1e-14,1]) # fix the range of power
                #plt.title('fmax={} kHz'.format(peak_freq))
                #plt.xlabel('Osc. frequency (kHz)')
                #plt.ylabel('PSD')
                plt.xticks([])
                plt.yticks([])
        plt.tight_layout(pad=0.1)    
        plt.savefig(path+'psd_all.png',dpi=200) #too many points. save to png only

        sum_data ={
            'PeakFreq':PeakFreq,
            'PeakPower':PeakPower,
            'input_rate_array' :input_rate_array,
            'ie_ratio_array':ie_ratio_array,
            'mnn_delay_array':mnn_delay_array,
        }

        np.savez(sum_data_file, **sum_data)
    
    # plot aggregated data
    
    plt.figure(figsize=(7,6))    
    for i in range(shape[0]):
        for j in range(shape[1]):
            input_rate = sum_data['input_rate_array'][i]
            ie_ratio = sum_data['ie_ratio_array'][j]
            plt.subplot(2,2, 2*i+j+1)
            plt.plot(sum_data['mnn_delay_array'], PeakFreq[i,j,:] ,'.')
            plt.xlabel('Delay (a.u.)')
            plt.ylabel('Osc. frequency (a.u.)')
            plt.title('input={}, ie_ratio={}'.format(input_rate, ie_ratio))
    plt.tight_layout()
    plt.savefig(path+'mnn_freq_vs_delay.pdf', format='pdf')
    
    plt.figure(figsize=(7,6))    
    for i in range(shape[0]):
        for j in range(shape[1]):
            input_rate = sum_data['input_rate_array'][i]
            ie_ratio = sum_data['ie_ratio_array'][j]
            plt.subplot(2,2, 2*i+j+1)
            plt.semilogy(sum_data['mnn_delay_array'], PeakPower[i,j,:] ,'.')
            plt.xlabel('Delay (a.u.)')
            plt.ylabel('Peak power density')
            plt.title('input={}, ie_ratio={}'.format(input_rate, ie_ratio))
    plt.tight_layout()
    plt.savefig(path+'mnn_power_vs_delay.pdf', format='pdf')
    
    
    # finally fit the coefficients
    #kk = sum_data['mnn_delay_array'] > 0.5 # keep only delay values larger then 0.5
    
    print('Omit delay < {} a.u. for fitting mnn'.format(sum_data['mnn_delay_array'][cut_off]) )
    plt.figure(figsize=(7,6))    
    for i in range(shape[0]):
        for j in range(shape[1]):
            input_rate = sum_data['input_rate_array'][i]
            ie_ratio = sum_data['ie_ratio_array'][j]
            x = 1/sum_data['mnn_delay_array'][cut_off:]
            y = PeakFreq[i,j,:][cut_off:]
            #coefs = np.polyfit(x,y,1)   
            #print(coefs)         
            a, _, _, _ = np.linalg.lstsq(x[:,np.newaxis], y)
            plt.subplot(2,2, 2*i+j+1)
            #plt.plot(x, coefs[0]*x+coefs[1], color='gray')
            plt.plot(x, a*x, color='gray')
            plt.plot(x, y ,'.')            
            plt.legend(['slope={}'.format(np.round(a[0],4))])
            plt.xlabel('1/delay (a.u.)')
            plt.ylabel('Osc. frequency')
            plt.title('input={}, ie_ratio={}'.format(input_rate, ie_ratio))
    plt.tight_layout()
    plt.savefig(path+'mnn_fit_freq_delay.pdf', format='pdf')


    return


def plot_freq_delay_curve_snn_vs_mnn():
    
    mnn_path = './projects/pre2023/runs/2024_May_21_rec_mnn_longer/'
    sum_data_file = mnn_path+'psd_analysis.npz'
    
    if os.path.exists(sum_data_file):
        print('Loading cached data...')
        sum_data = np.load(sum_data_file)
        mnn_PeakFreq = sum_data['PeakFreq']
        mnn_PeakPower = sum_data['PeakPower']
        mnn_input_rate_array = sum_data['input_rate_array']
        mnn_ie_ratio_array = sum_data['ie_ratio_array']
        mnn_delay_array = sum_data['mnn_delay_array']
    else:        
        print('psd_analysis.npz not found in '+mnn_path)

    snn_path = './projects/pre2023/runs/2024_May_20_rec_snn_delay/'
    sum_data_file = snn_path+'psd_analysis.npz'
    
    if os.path.exists(sum_data_file):
        print('Loading cached data...')
        sum_data = np.load(sum_data_file)
        snn_PeakFreq = sum_data['PeakFreq']
        snn_PeakPower = sum_data['PeakPower']
        snn_input_rate_array = sum_data['snn_uext']
        snn_ie_ratio_array = sum_data['snn_ie_ratio']
        snn_delay_array = sum_data['snn_delay']
    else:        
        print('psd_analysis.npz not found in '+snn_path)

    #beta = 1.0262 # calibration factor for mnn
    #beta = 0.2748/0.2599 #mnn/snn
    beta = 0.3197/0.2738 # mnn/snn, ie_ratio = 6, input rate=40 # works quite well for ie_ratio =8 too!
    print('calibration factor beta=', beta)
    snn_cut_off = 4
    mnn_cut_off = 5

    
    plt.figure(figsize=(7,6))
    for i in range(2):
        for j in range(2):
            input_rate = snn_input_rate_array[i] # these should be the same in mnn and snn
            ie_ratio = snn_ie_ratio_array[j]
            plt.subplot(2,2, 2*i+j+1)
            #plt.plot(1/mnn_delay_array/beta, mnn_PeakFreq[i,j,:]/beta/beta)
            #plt.plot(1/snn_delay_array, snn_PeakFreq[i,j,:] ,'.')
            plt.plot(mnn_delay_array*beta, mnn_PeakFreq[i,j,:]/beta/beta)
            plt.plot(snn_delay_array, snn_PeakFreq[i,j,:] ,'.')
            plt.xlabel('Delay (ms)')
            plt.ylabel('Osc. frequency (kHz)')
            plt.title('input={}, ie_ratio={}'.format(input_rate, ie_ratio))
    plt.tight_layout()
    plt.savefig(snn_path+'snn_vs_mnn_freq_delay.pdf', format='pdf')

    plt.figure(figsize=(7,6))
    for i in range(2):
        for j in range(2):
            input_rate = snn_input_rate_array[i] # these should be the same in mnn and snn
            ie_ratio = snn_ie_ratio_array[j]
            plt.subplot(2,2, 2*i+j+1)
            plt.plot(1/mnn_delay_array[mnn_cut_off:]/beta, mnn_PeakFreq[i,j,mnn_cut_off:]/beta/beta)
            plt.plot(1/snn_delay_array[snn_cut_off:], snn_PeakFreq[i,j,snn_cut_off:] ,'.')
            plt.xlabel('1/delay (ms)')
            plt.ylabel('Osc. frequency (kHz)')
            plt.title('input={}, ie_ratio={}'.format(input_rate, ie_ratio))
    plt.tight_layout()
    plt.savefig(snn_path+'snn_vs_mnn_freq_delay_inv.pdf', format='pdf')
    plt.close('all')
    print('Fig saved to:' + snn_path +'snn_vs_mnnfreq_delay_inv.pdf')

    # production quality figure
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
 '#7f7f7f', '#bcbd22', '#17becf'] # matplotlib default colors
    snn_cut_off = 3
    mnn_cut_off = 3
    i = 1 # fix input rate = 40
    j = 0 # ie_ratio = 6 #for j in range(2):
    input_rate = snn_input_rate_array[i] # these should be the same in mnn and snn
    ie_ratio = snn_ie_ratio_array[j] 
    print('critical delay for snn: ', 0.5*(snn_delay_array[snn_cut_off]+snn_delay_array[snn_cut_off-1]))
    # frequency delay curve
    plt.figure(figsize=(3.5,3))       
    plt.plot(mnn_delay_array[mnn_cut_off:]*beta, mnn_PeakFreq[i,j,mnn_cut_off:]/beta/beta, color='gray')
    plt.plot(mnn_delay_array[[0,mnn_cut_off]]*beta, [0,0], color='gray')
    plt.plot(snn_delay_array[snn_cut_off:], snn_PeakFreq[i,j,snn_cut_off:] ,'.', color=colors[0])
    plt.plot(snn_delay_array[:snn_cut_off], np.zeros(snn_cut_off),'.', color=colors[0])
    plt.xlabel('Delay (ms)')
    plt.ylabel('Osc. frequency (kHz)')
    plt.xlim([0,3])
    #plt.title('input={}, ie_ratio={}'.format(input_rate, ie_ratio))
    plt.tight_layout()
    plt.savefig(snn_path+'snn_vs_mnnfreq_delay_production.pdf', format='pdf')

    # power delay curve
    plt.figure(figsize=(3.5,3))       
    plt.plot(mnn_delay_array*beta, mnn_PeakPower[i,j,:], color=colors[0])
    #plt.plot(mnn_delay_array[[0,mnn_cut_off-1]]*beta, [0,0], color=colors[0])
    N=12500
    plt.semilogy(snn_delay_array, snn_PeakPower[i,j,:] ,'.', color=colors[1])
    #plt.plot(snn_delay_array[:snn_cut_off], np.zeros(snn_cut_off),'.', color=colors[1])
    plt.xlabel('Delay (ms)')
    plt.ylabel('Peak power (/kHz)')
    plt.xlim([0,3])
    #plt.title('input={}, ie_ratio={}'.format(input_rate, ie_ratio))
    plt.tight_layout()
    plt.savefig(snn_path+'snn_vs_mnn_power_delay_production.pdf', format='pdf')
    
    
    
    
    plt.close('all')
    print('Fig saved to:' + snn_path +'snn_vs_mnnfreq_delay_production.pdf')



    return



if __name__=='__main__':
    #exp_id = '2024_Apr_28_rec_mnn'
    #exp_id = '2024_May_15_rec_mnn'
    #indx = 30
    #inspect_raw_data(exp_id, indx)
    #plot_corr_hist_all(exp_id)
    
    #for i in [10,20,30]:
    #    plot_slices(exp_id, slice_i=i)
    #plot_heatmaps(exp_id)

    #exp_id = '2024_May_16_rec_snn'
    #exp_id = '2024_May_18_rec_snn'
    #exp_id = '2024_May_19_rec_snn'
    #plot_snn_vs_mnn(exp_id)

    #temp_inspect_raw_data()

    #inspect_snn_oscillation()
    #inspect_mnn_oscillation()
    plot_freq_delay_curve_snn_vs_mnn()

