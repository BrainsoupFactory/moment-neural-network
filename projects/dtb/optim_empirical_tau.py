from scipy.optimize import minimize, basinhopping
import numpy as np
from projects.dtb.ma_conductance import Moment_Activation_Cond
from matplotlib import pyplot as plt

# # Define the function f(x, y, param1, param2)
# def func(xy, param1, param2):
#     ''' Demo toy function'''
#     x, y = xy
#     param1_squared = param1**2
#     param2_cubed = param2**3
#     # Define the equation
#     eq1 = x**2 + y**2 - param1_squared  # Example equation: x^2 + y^2 - param1^2 = 0
#     eq2 = x + y - param2_cubed          # Example equation: x + y - param2^3 = 0
#     return [eq1, eq2]


def load_empirical_data(exp_id, indx = 0):
    path = './projects/dtb/runs/{}/'.format(exp_id)
    
    dat = np.load(path+str(indx).zfill(3)+'.npz', allow_pickle=True)
    config=dat['config'].item()
    #print(list(dat))
    
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

    if exp_id == 'vary_input_stats_poisson':            
        we=np.linspace(0,1,21)[1:][indx]
        wi=0.4
        
        exc_rate = dat['exc_input_rate'] # firng rate in kHz
        inh_rate = dat['inh_input_rate']
        X, Y = np.meshgrid(exc_rate, inh_rate, indexing='xy')
        
        exc_input_mean = we*X
        exc_input_std = we*np.sqrt(X)
        inh_input_mean = wi*Y
        inh_input_std = wi*np.sqrt(Y)
    elif exp_id == 'vary_input_stats_we_wi':
        we = np.linspace(0,1.0,21)[1:]
        wi = np.linspace(0,1.0,21)
        ii, jj = np.unravel_index(indx, ( len(we) , len(wi)) ) #, order='C')
        we=we[ii]
        wi=wi[jj]
        
        exc_rate = dat['exc_input_rate'] # firng rate in kHz
        inh_rate = dat['inh_input_rate']
        X, Y = np.meshgrid(exc_rate, inh_rate, indexing='xy')
        
        exc_input_mean = we*X
        exc_input_std = we*np.sqrt(X)
        inh_input_mean = wi*Y
        inh_input_std = wi*np.sqrt(Y)
        
    
    
    
    T = dat['T_snn']-config['discard'] # ms
    
    mean_firing_rate = dat['spk_count'].mean(0)/T
    firing_var = dat['spk_count'].var(0)/T
    
    return ma, mean_firing_rate, firing_var, exc_input_mean.flatten(), exc_input_std.flatten(), inh_input_mean.flatten(), inh_input_std.flatten()
    

def func(X, ma, mean_firing_rate, firing_var, exc_input_mean, exc_input_std, inh_input_mean, inh_input_std):
    
    ma.tau_E = 4*X[0]
    ma.tau_I = 10*X[1]
    
    eff_input_mean, eff_input_std, tau_eff = ma.cond2curr(exc_input_mean,exc_input_std,inh_input_mean,inh_input_std)
    mean_out = ma.forward_fast_mean( eff_input_mean, eff_input_std, tau_eff)
    std_out = ma.forward_fast_std( eff_input_mean, eff_input_std, tau_eff, mean_out)
    
    L1 = np.power(mean_out - mean_firing_rate, 2).sum()
    L2 = np.power(std_out*std_out - firing_var,2).sum()
    
    return L1+L2
    

def param_sweep(exp_id, indx, tau_E, tau_I):
    ''' brute force parameter sweep'''
    
    params = load_empirical_data(exp_id, indx = indx)
    
    L = np.zeros((len(tau_E), len(tau_I)))
    for i in range(len(tau_E)):
        print(i)
        for j in range(len(tau_I)):
            X = [tau_E[i], tau_I[j]]
            L[i,j] = func(X, params[0],params[1],params[2],params[3],params[4],params[5],params[6])#ma, mean_firing_rate, firing_var, exc_input_mean, exc_input_std, inh_input_mean, inh_input_std):

    # Plot optimal solution 
    extent = (tau_I[0], tau_I[-1], tau_E[0], tau_E[-1])
    
    path =  './projects/dtb/runs/{}/'.format( exp_id )
    plt.close('all')
    plt.figure(figsize=(3.5,3))
    plt.imshow(np.log10(L), origin='lower', extent=extent)
    plt.plot(res.x[1], res.x[0], 'xr')
    plt.colorbar()
    plt.xlabel(r'$\tau_I$ (ms)')
    plt.ylabel(r'$\tau_E$ (ms)')
    plt.tight_layout()
    plt.savefig(path+'potential_well_{}.png'.format(indx))
    plt.close('all')

    return L

def batch_post_process(exp_id, cache=False):
    '''batch processing SNN simulation results'''
    x0 = [1.0, 1.0]
    #exp_id='vary_input_stats_poisson'
    path =  './projects/dtb/runs/{}/'.format( exp_id )
    #method = 'BFGS'
    method = 'Nelder-Mead'
    
    if exp_id=='vary_input_stats_we_wi':
        corr_factor = np.zeros((420,2))
    elif exp_id =='vary_input_stats_poisson':
        corr_factor = np.zeros((20,2))

    #indx = 1
    if cache:
        # load cached data
        print('Loading cache data...')
        corr_factor = np.load(path+'correction_factor.npy')
    else:
        for indx in range(corr_factor.shape[0]):
            params = load_empirical_data(exp_id, indx = indx)
            res = minimize(func, x0, args=params, method=method, tol=1e-16)
            corr_factor[indx,:] = res.x
            print(res.x)
        with open(path+'correction_factor.npy', 'wb') as f:
            np.save(f, corr_factor)
        
    if exp_id=='vary_input_stats_we_wi':
        #plt.plot(corr_factor[:,0])
        #plt.plot(corr_factor[:,1])
        A = corr_factor[:,0].reshape(20,21)
        B = corr_factor[:,1].reshape(20,21)
        extent = (0,1,0,1)
        plt.figure(figsize=(8,3))
        plt.subplot(1,2,1)
        plt.imshow(A, origin='lower', cmap = 'coolwarm', vmin=0,vmax=2, extent=extent)
        plt.title(r'$\tau_E$ correction')
        plt.colorbar()
        plt.xlabel('$w_I$')
        plt.ylabel('$w_E$')
        plt.subplot(1,2,2)
        plt.imshow(B, origin='lower', cmap = 'coolwarm', vmin=-1,vmax=3, extent=extent)   
        plt.title(r'$\tau_I$ correction')
        plt.xlabel('$w_I$')
        plt.ylabel('$w_E$')  
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path+'correction_factor_vs_we_wi.png')
        plt.close('all')
    elif exp_id =='vary_input_stats_poisson':
        we = np.linspace(0,1,21)[1:]
        plt.figure(figsize=(3.5,3))
        plt.plot(we,corr_factor[:,0])
        plt.plot(we,corr_factor[:,1])
        plt.xlabel('w_E')
        plt.ylabel('correction factor')
        plt.tight_layout()
        plt.savefig(path+'correction_factor_vs_we.png')
        plt.close('all')
    
    return corr_factor

if __name__=='__main__':
    # Find optimal scaling factor with optimization
    #exp_id='vary_input_stats_poisson'
    exp_id='vary_input_stats_we_wi'
    batch_post_process(exp_id, cache=True)

    
    



