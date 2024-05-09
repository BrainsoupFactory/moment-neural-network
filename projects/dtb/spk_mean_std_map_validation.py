import torch
import matplotlib.pyplot as plt
import sys, os
from projects.dtb.snn_simulator_torch import CondInteNFire, gen_config
import numpy as np

class SNNInputGenerator():
    def __init__(self, config, input_mean=None, input_std=None, input_corr=None, rho=None, w=None, ei_spk_stats=None, exc_rate=None, inh_rate=None, device='cpu'):
        self.NE = config['NE']
        self.NI = config['NI']
        self.N = config['NE']+config['NI']
        self.batchsize = config['batchsize']
        self.device = device
        
        self.input_mean = input_mean
        self.input_std = input_std

        self.w = w
        self.exc_rate = exc_rate
        self.inh_rate = inh_rate

        self.ei_spk_stats = ei_spk_stats  # [exc_spk_mean, exc_spk_var, inh_spk_mean, inh_spk_var]

        self.exc_t2spk = torch.zeros(self.batchsize, self.N, device=device, dtype=torch.int64)
        self.inh_t2spk = torch.zeros(self.batchsize, self.N, device=device, dtype=torch.int64)

        try:
            if not input_corr:    
                input_corr = torch.ones(self.N,self.N, device=device)*rho
                input_corr.fill_diagonal_(1.0)
            
            self.input_corr = input_corr        
            cov = input_corr*input_std*input_std.T
            # Perform Cholesky decomposition
            if np.abs(rho)>1e-6:
                self.chol = torch.linalg.cholesky(cov, upper=True)#.mH # upper cholesky
            else:
                self.chol = None
        except:
            pass            

        return
    
    def gen_corr_normal_rv(self, dt, device='cpu'): #generate correlated gaussian noise
        '''
        OUTPUT: correlated normal random variables (sample size x num elements)
        '''  
        dt = torch.tensor(dt).to(device)
        # NB: assume self.input_mean has size (1,N)
        standard_normal = torch.randn( self.batchsize, self.N, device=device)
        correlated_samples = self.input_mean*dt + torch.sqrt(dt)*torch.matmul(standard_normal, self.chol)
        return correlated_samples
    
    def gen_uncorr_normal_rv(self, dt, device='cpu'):
        dt = torch.tensor(dt).to(device)
        standard_normal = torch.randn( self.batchsize, self.N, device=device)
        correlated_samples = self.input_mean*dt + torch.sqrt(dt)*standard_normal*self.input_std
        return correlated_samples
    
    def gen_uncorr_spk(self, dt, device='cpu'):
        # w, rate dimension: 1 x #neurons
        pe = dt*self.exc_rate
        pi = dt*self.inh_rate        
        # use poisson instead
        spk  = (torch.poisson(pe) - torch.poisson(pi))*self.w
        #spk_e = torch.rand(self.batchsize, self.N, device=device)<pe
        #spk_i = torch.rand(self.batchsize, self.N, device=device)<pi
        #spk = (spk_e.float()-spk_i.float())*self.w
        return spk

    @staticmethod
    def gen_gamma_spk(dt, spk_mean, spk_var, time2nextspike):
        ''' Generate spikes with gamma distributed ISI
            isi_mean = shape*scale (unit: kHz)
            isi_var = shape*scale^2 (unit: kHz)
        '''
        #TODO: creating new instances of Gamma objects can be very slow!

        # calculate ISI parameters given spk count stats
        isi_mean = 1 / (spk_mean+1e-10)
        isi_var = torch.pow(isi_mean, 3) * spk_var        
        
        beta = isi_mean/isi_var
        alpha = isi_mean * isi_mean / isi_var
        
        is_spike = time2nextspike==0      # emit a spike if time-to-next-spike reaches zero
        isi = torch.distributions.Gamma(alpha[is_spike], beta[is_spike]).sample() #sample new ISI
        indx = isi_var[is_spike]==0
        isi[indx] = isi_mean[is_spike][indx] #replace isi with its mean if isi_var is zero.
        time2nextspike[is_spike] = (isi/dt).to(torch.int64) # update time-to-next-spike
        time2nextspike[~is_spike] += -1 # count down time
        
        return is_spike
    
    def gen_gamma_spk_EI(self,dt, device='cpu'):
        #ei_spk_stats  # [exc_spk_mean, exc_spk_var, inh_spk_mean, inh_spk_var]
        exc_spk = self.gen_gamma_spk(dt, self.ei_spk_stats[0], self.ei_spk_stats[1], self.exc_t2spk)
        inh_spk = self.gen_gamma_spk(dt, self.ei_spk_stats[2], self.ei_spk_stats[3], self.inh_t2spk)
        total_current  = (exc_spk.float() - inh_spk.float())*self.w
        #print(total_current[:,-1].mean()/dt) #-1 should correspond to ubar=4, sbar=5;
        #print(total_current[:,-1].std()/np.sqrt(dt))
        # debug
        #dt = torch.tensor(dt).to(device)
        #standard_normal = torch.randn( self.batchsize, self.N, device=device)
        #total_current = 2.0*dt + torch.sqrt(dt)*standard_normal*3.0
        
        return total_current

    def gen_uncorr_spk_one_channel(self, dt, device='cpu'):
        pe = dt*self.input_mean
        pe = pe.expand(self.batchsize, -1) # expand view no new copy
        spk  = self.w*torch.poisson(pe)
        return spk
    




def run(exp_id, indx=0, T=1e3, dt_snn=1e-2, device = 'cuda', savefile=False, savefig=False):
    path =  './projects/dtb/runs/{}/'.format( exp_id )
    if not os.path.exists(path):
        os.makedirs(path)
    
    print('Setting up parameters to be swept...')
    
    batchsize = 1000 # number of independent samples (make this large to avoid no spikes)     
    device= device    

    #indx = 0 # dummy variable, no use

    rho = 0 # for mean, std mapping need independent samples
    exc_input_rate = torch.linspace(0,1,21, device=device)
    inh_input_rate = torch.linspace(0,0.25,11, device=device)
    
    #KE=400 # excitatory in-degree?
    #KI=100
    #input_rate = torch.tensor([5e-3, 10e-3, 20e-3], device=device).unsqueeze(0) # sp/ms, same for both exc and inh inputs
    we=0.1*np.array([1,2,3])[indx] # 0.5
    wi = 0.4 # 0.4, 1.0, 10
       

    X, Y = torch.meshgrid(exc_input_rate, inh_input_rate, indexing='xy')
    # inner dim: std, outer dim: mean
    X = X.flatten().unsqueeze(0)
    Y = Y.flatten().unsqueeze(0)

    print('Setting up SNN model...')
    num_neurons = X.shape[1]
    
    config = gen_config(batchsize, num_neurons, T, device=device, dt_snn = dt_snn )
    config['record_interval'] = 100 # ms. record spike count every x ms
    
    print('rho = ', rho)    
    print('Using uncorrelated input.')
    exc_input_gen = SNNInputGenerator(config, input_mean=X, w=we, device=device).gen_uncorr_spk_one_channel
    inh_input_gen = SNNInputGenerator(config, input_mean=Y, w=wi, device=device).gen_uncorr_spk_one_channel

    snn_model = CondInteNFire(config, [exc_input_gen,inh_input_gen])

    print('Simulating SNN...')
    spk_count, V, t, spk_count_history = snn_model.run( config['T_snn'] , record_interval=config['record_interval'], show_message=True, device = device) # ms

    
    data_dict = {'spk_count': spk_count.cpu().numpy(),
    'config':config,
    'exc_input_rate':exc_input_rate.cpu().numpy(),
    'inh_input_rate':inh_input_rate.cpu().numpy(),
    'rho':rho,    
    'spk_count_history':spk_count_history,
    't':t,
    'T_snn': T,
    'dt_snn': dt_snn,
    }
    
    if savefile:
        #filename = str(indx).zfill(3) +'_'+str(int(time.time())) + '.npz'
        filename = str(indx).zfill(3) +'.npz'
        np.savez(path+filename, **data_dict)
        print('Results saved to: ', path+filename)
    
    if savefig:
        print('Calculating spike count stats...')
        T = config['T_snn']-config['discard']
        snn_rate = torch.mean(spk_count, dim=0)/T
        snn_std = torch.std(spk_count,dim=0)/np.sqrt(T)
        #snn_fano_factor = torch.var(spk_count,dim=0)/torch.mean(spk_count,dim=0)
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(snn_rate.cpu().numpy())
        plt.ylabel('Firing rate (sp/ms)')
        plt.subplot(2,1,2)
        plt.plot(snn_std.cpu().numpy())
        plt.ylabel('Firing variability (sp/ms$^{1/2}$)')
        plt.tight_layout()
        plt.savefig('test_snn_rate_var.png')
        
        plt.close('all')



def spike_gen_debugger(T=100, dt_snn=1e-2, device = 'cuda'):
    '''check gamma spike generator is correct'''
    batchsize = 1000 # number of independent samples (make this large to avoid no spikes)     
    num_neurons = 5
    
    config = gen_config(batchsize, num_neurons, T, device=device, dt_snn = dt_snn )
    input_gen = SNNInputGenerator(config, device=device).gen_gamma_spk
    
     # move these to initialization block
    spk_mean = torch.rand(1, num_neurons, device=device) 
    spk_var = torch.rand(1, num_neurons, device=device)
    print('theoretical mean:', spk_mean)
    print('theoretical var:', spk_var)
    #
    spk_mean = spk_mean*torch.ones(batchsize,1, device=device)
    spk_var = spk_var*torch.ones(batchsize,1, device=device)
    time2nextspike = torch.zeros(batchsize, num_neurons, device=device, dtype=torch.int64)

    nsteps = int(T/dt_snn)
    spk_count = 0.0
    for i in range(nsteps):
        #print(i)
        is_spike = input_gen(dt_snn, spk_mean, spk_var, time2nextspike)
        spk_count += is_spike
    
    emp_mean = spk_count.mean(dim=0)/T
    emp_var = spk_count.var(dim=0)/T
    print('emp mean:', emp_mean)
    print('emp_var:', emp_var)
    
    


def run_vary_tau_E(exp_id, indx, T=1e3, dt_snn=1e-2, device = 'cuda', savefile=False, savefig=False):
    path =  './projects/dtb/runs/{}/'.format( exp_id )
    if not os.path.exists(path):
        os.makedirs(path)
    
    print('Setting up parameters to be swept...')
    
    batchsize = 1000 # number of independent samples (make this large to avoid no spikes)     
    device= device    

    #indx = 0 # dummy variable, no use
    
    rho = 0 # for mean, std mapping need independent samples
    
    KE=400 # excitatory in-degree?
    KI=100
    input_rate = torch.tensor([5e-3, 10e-3, 20e-3], device=device).unsqueeze(0) # sp/ms, same for both exc and inh inputs
    we = 0.1 # 0.5
    wi = 0.4 # 0.4, 1.0, 10
    
    exc_input_rate = input_rate*KE
    inh_input_rate = input_rate*KI
   
    print('Setting up SNN model...')
    num_neurons = exc_input_rate.shape[1]
    
    config = gen_config(batchsize, num_neurons, T, device=device, dt_snn = dt_snn )
    config['record_interval'] = 100 # ms. record spike count every x ms
    
    if exp_id=='vary_tau_E':
        tau_E = np.linspace(0,100,21)
    elif exp_id=='vary_tau_E_zoom_in':
        tau_E = np.linspace(0,10,21)
    config['tau_E']=tau_E[indx]
    
    print('rho = ', rho)    
    print('Using uncorrelated input.')
    exc_input_gen = SNNInputGenerator(config, input_mean=exc_input_rate, w=we, device=device).gen_uncorr_spk_one_channel
    inh_input_gen = SNNInputGenerator(config, input_mean=inh_input_rate, w=wi, device=device).gen_uncorr_spk_one_channel

    snn_model = CondInteNFire(config, [exc_input_gen,inh_input_gen])

    print('Simulating SNN...')
    spk_count, V, t, spk_count_history = snn_model.run( config['T_snn'] , record_interval=config['record_interval'], show_message=True, device = device) # ms

    
    data_dict = {'spk_count': spk_count.cpu().numpy(),
    'config':config,
    'exc_input_rate':exc_input_rate.cpu().numpy(),
    'inh_input_rate':inh_input_rate.cpu().numpy(),
    'rho':rho,    
    'spk_count_history':spk_count_history,
    't':t,
    'T_snn': T,
    'dt_snn': dt_snn,
    'tau_E': tau_E,
    'indx':indx,
    }
    
    if savefile:
        #filename = str(indx).zfill(3) +'_'+str(int(time.time())) + '.npz'
        filename = str(indx).zfill(3) +'.npz'
        np.savez(path+filename, **data_dict)
        print('Results saved to: ', path+filename)
    
    if savefig:
        print('Calculating spike count stats...')
        snn_rate = torch.mean(spk_count, dim=0)/config['T_snn']
        snn_std = torch.std(spk_count,dim=0)/np.sqrt(config['T_snn'])
        #snn_fano_factor = torch.var(spk_count,dim=0)/torch.mean(spk_count,dim=0)
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(snn_rate.cpu().numpy())
        plt.ylabel('Firing rate (sp/ms)')
        plt.subplot(2,1,2)
        plt.plot(snn_std.cpu().numpy())
        plt.ylabel('Firing variability (sp/ms$^{1/2}$)')
        plt.tight_layout()
        plt.savefig('test_snn_rate_var.png')
        
        plt.close('all')


def run_gaussian_input(exp_id, indx=0, T=1e3, dt_snn=1e-2, device = 'cuda', savefile=False, savefig=False):
    path =  './projects/dtb/runs/{}/'.format( exp_id )
    if not os.path.exists(path):
        os.makedirs(path)
    
    print('Setting up parameters to be swept...')
    
    batchsize = 1000 # number of independent samples (make this large to avoid no spikes)     
    device= device    

    #indx = 0 # dummy variable, no use

    rho = 0 # for mean, std mapping need independent samples
    exc_input_rate = torch.linspace(0,1,11, device=device)
    exc_input_FF = torch.linspace(0,2,11, device=device)
    inh_input_rate = torch.linspace(0,0.25,11, device=device)
    inh_input_FF = torch.linspace(0,2,11, device=device)
    
    we=0.1*np.array([1,2,3])[indx] # 0.5
    wi = 0.4 # 0.4, 1.0, 10
       
    M1, FF1, M2, FF2 = torch.meshgrid(exc_input_rate, exc_input_FF, inh_input_rate, inh_input_FF, indexing='ij')
    
    # inner dim: std, outer dim: mean
    M1 = M1.flatten().unsqueeze(0)
    FF1 = FF1.flatten().unsqueeze(0)
    M2 = M2.flatten().unsqueeze(0)
    FF2 = FF2.flatten().unsqueeze(0)
    
    # calculate weighted input mean/std 
    exc_input_mean = M1*we
    exc_input_std = we*torch.sqrt(M1*FF1)
    inh_input_mean = M2*wi
    inh_input_std = wi*torch.sqrt(M2*FF2)
    

    print('Setting up SNN model...')
    num_neurons = M1.shape[1]
    
    config = gen_config(batchsize, num_neurons, T, device=device, dt_snn = dt_snn )
    config['record_interval'] = 100 # ms. record spike count every x ms
    
    print('rho = ', rho)    
    print('Using uncorrelated input.')
    print('batchsize=', batchsize)
    print(' #neurons=', num_neurons)
    exc_input_gen = SNNInputGenerator(config, input_mean=exc_input_mean, input_std=exc_input_std,  device=device).gen_uncorr_normal_rv
    inh_input_gen = SNNInputGenerator(config, input_mean=inh_input_mean, input_std=inh_input_std,  device=device).gen_uncorr_normal_rv

    snn_model = CondInteNFire(config, [exc_input_gen,inh_input_gen])

    print('Simulating SNN...')
    spk_count, V, t, spk_count_history = snn_model.run( config['T_snn'] , record_interval=config['record_interval'], show_message=True, device = device) # ms

    
    data_dict = {'spk_count': spk_count.cpu().numpy(),
    'config':config,
    'exc_input_rate':exc_input_rate.cpu().numpy(),
    'inh_input_rate':inh_input_rate.cpu().numpy(),
    'exc_input_FF': exc_input_FF.cpu().numpy(),
    'inh_input_FF': inh_input_FF.cpu().numpy(),
    'rho':rho,    
    'spk_count_history':spk_count_history,
    't':t,
    'T_snn': T,
    'dt_snn': dt_snn,
    }
    
    if savefile:
        #filename = str(indx).zfill(3) +'_'+str(int(time.time())) + '.npz'
        filename = str(indx).zfill(3) +'.npz'
        np.savez(path+filename, **data_dict)
        print('Results saved to: ', path+filename)


if __name__=='__main__':
    torch.set_default_dtype(torch.float64) #for accurate corrcoef estimate
    
    device = 'cpu'
    #device = 'cuda'
    exp_id = 'vary_input_stats'
    for i in range(3):
        run_gaussian_input(exp_id, indx=i, T=1e3, dt_snn=1e-2, device = device, savefile=True, savefig=True)
    
    #exp_id = 'vary_tau_E'
    #exp_id = 'vary_tau_E_zoom_in' #sys.argv[1] #'2024_mar_30_mean_std'
    
    # print('Running experiment {}'.format(exp_id))
    # for indx in range(21):
    #     print('Running trial:', indx)
    #     run_vary_tau_E(exp_id, indx, T = 10e3, dt_snn= 1e-2, device = device, savefile=True)
    
    # notes on performance
    # batchsize = 1000, T=1e3
    # 0.07 min on cpu
    # 0.12 min on gpu
    # => gpu no good on small scale
    
    # try 1e4 batchsize
    # cpu 0.31 min
    # gpu 0.13 min
    # 
            
