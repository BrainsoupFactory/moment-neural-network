import torch
import matplotlib.pyplot as plt
import sys, os
from projects.pre2023.snn_simulator_torch import *


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
    




def run(exp_id, T=1e3, dt_snn=1e-2, device = 'cuda', savefile=False, savefig=False):
    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    if not os.path.exists(path):
        os.makedirs(path)
    
    print('Setting up parameters to be swept...')
    
    batchsize = 10000 # number of independent samples (make this large to avoid no spikes)     
    device= device    

    indx = 0 # dummy variable, no use

    rho = 0 # for mean, std mapping need independent samples
    input_mean = torch.linspace(-1,4,51, device=device)
    input_std = torch.linspace(0,5,51, device=device)[1:] # no noise when std=0

    mean_grid, std_grid = torch.meshgrid(input_mean, input_std, indexing='ij')
    # inner dim: std, outer dim: mean
    mean_grid = mean_grid.flatten().unsqueeze(0)
    std_grid = std_grid.flatten().unsqueeze(0)

    print('Setting up SNN model...')
    num_neurons = mean_grid.shape[1]
    
    config = gen_config(batchsize, num_neurons, T, device=device, dt_snn = dt_snn )
    config['record_interval'] = 100 # ms. record spike count every x ms
    
    print('rho = ', rho)    
    print('Using uncorrelated input.')
    input_gen = SNNInputGenerator(config, input_mean=mean_grid, input_std=std_grid, rho=rho, device=device).gen_uncorr_normal_rv
    
    snn_model = InteNFire(config, input_gen)

    print('Simulating SNN...')
    spk_count, V, t, spk_count_history = snn_model.run( config['T_snn'] , record_interval=config['record_interval'], show_message=True, device = device) # ms

    
    data_dict = {'spk_count': spk_count.cpu().numpy(),
    'config':config,
    'mean_grid':mean_grid.cpu().numpy(),
    'std_grid':std_grid.cpu().numpy(),
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

def run_w_rate_validation(exp_id, T=1e3, dt_snn=1e-2, device = 'cuda', savefile=False):
    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    if not os.path.exists(path):
        os.makedirs(path)
    
    print('Setting up parameters to be swept...')
    
    batchsize = 10000 # number of independent samples (make this large to avoid no spikes)     
    device= device    

    indx = 0 # dummy variable, no use

    rho = 0 # for mean, std mapping need independent samples
    #input_mean = torch.linspace(0,4,41, device=device)
    input_w = torch.logspace(-2,1,31, device=device)
    input_rate = torch.linspace(0,40,41, device=device)

    input_w_grid, input_rate_grid = torch.meshgrid(input_w, input_rate, indexing='ij')
    # inner dim: std, outer dim: mean
    input_w_grid = input_w_grid.flatten().unsqueeze(0)
    input_rate_grid = input_rate_grid.flatten().unsqueeze(0)

    print('Setting up SNN model...')
    num_neurons = input_rate_grid.shape[1]
    
    config = gen_config(batchsize, num_neurons, T, device=device, dt_snn = dt_snn )
    config['record_interval'] = None # ms. record spike count every x ms
    
    print('rho = ', rho)    
    print('Using uncorrelated input.')
    input_gen = SNNInputGenerator(config, w=input_w_grid, rate=input_rate_grid, rho=rho, device=device).gen_uncorr_spk
    
    snn_model = InteNFire(config, input_gen)

    print('Simulating SNN...')
    spk_count, V, t, spk_count_history = snn_model.run( config['T_snn'] , record_interval=config['record_interval'], show_message=True, device = device) # ms

    
    data_dict = {'spk_count': spk_count.cpu().numpy(),
    'config':config,
    'input_w': input_w.cpu().numpy(),
    'input_rate': input_rate.cpu().numpy(),
    'input_w_grid':input_w_grid.cpu().numpy(),
    'input_rate_grid':input_rate_grid.cpu().numpy(),
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



def run_w_rate_validation_EI(exp_id, T=1e3, dt_snn=1e-2, device = 'cuda', savefile=False):
    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    if not os.path.exists(path):
        os.makedirs(path)
    
    print('Setting up parameters to be swept...')
    
    batchsize = 1000 # number of independent samples (make this large to avoid no spikes)     
    device= device    

    indx = 0 # dummy variable, no use

    rho = 0 # for mean, std mapping need independent samples
    #input_mean = torch.linspace(0,4,41, device=device)
    # only consider we = wi. The point here is to test MA for w of different scales.
    input_w = torch.tensor([1e-2, 1e-1, 1, 10], device=device) #torch.logspace(-2,1,31, device=device)

    # Calculate Poisson rate given input curr stats    
    curr_mean = torch.linspace(-1,4,21, device=device)
    curr_std = torch.linspace(0,5,21, device=device)
    curr_mean_grid, curr_std_grid, w_grid = torch.meshgrid(curr_mean, curr_std, input_w, indexing='ij')
    curr_mean_grid = curr_mean_grid.flatten().unsqueeze(0)
    curr_std_grid = curr_std_grid.flatten().unsqueeze(0)
    w_grid = w_grid.flatten().unsqueeze(0)

    # A = curr_mean_grid/w_grid
    # B = (curr_std_grid/w_grid).pow(2.0)
    # exc_rate_grid = 0.5*(B+A)
    # inh_rate_grid = 0.5*(B-A)

    # print( torch.sum(exc_rate_grid<0 ))
    # print( torch.sum(inh_rate_grid<0 ))
    
    # exc_rate_grid = torch.relu(exc_rate_grid)
    # inh_rate_grid = torch.relu(inh_rate_grid)

    # if exc_rate_grid.max()*dt_snn >1:
    #     print('Warning: insufficient time resolution!')
    # print('Setting up SNN model...')
    # num_neurons = exc_rate_grid.shape[1]

    exc_spk_mean = (curr_mean_grid+1)/w_grid
    inh_spk_mean = torch.ones(curr_std_grid.shape, device=device)/w_grid
    exc_spk_var = 0.5*(curr_std_grid/w_grid).pow(2.0)
    inh_spk_var = exc_spk_var.clone()

    # #debug
    # # check the inputs are correct
    # plt.subplot(4,1,1)
    # plt.plot( (exc_spk_mean*w_grid).squeeze().cpu().numpy(),'.',markersize=1)
    # plt.subplot(4,1,2)
    # plt.plot( (inh_spk_mean*w_grid).squeeze().cpu().numpy(),'.',markersize=1)
    # plt.subplot(4,1,3)
    # plt.plot( (torch.sqrt(exc_spk_var)*w_grid).squeeze().cpu().numpy(),'.',markersize=1)
    # plt.subplot(4,1,4)
    # plt.plot( (torch.sqrt(inh_spk_var)*w_grid).squeeze().cpu().numpy(),'.',markersize=1)
    # plt.savefig('tmp.png')
    # plt.close('all')
    # #debug


    num_neurons = exc_spk_mean.shape[1]
    ei_spk_stats = [exc_spk_mean.repeat(batchsize, 1), exc_spk_var.repeat(batchsize, 1), inh_spk_mean.repeat(batchsize, 1), inh_spk_var.repeat(batchsize, 1)]
    
    config = gen_config(batchsize, num_neurons, T, device=device, dt_snn = dt_snn )
    config['record_interval'] = None # ms. record spike count every x ms
    
    
    print('rho = ', rho)    
    print('Using uncorrelated input.')
    #input_gen = SNNInputGenerator(config, w=w_grid, exc_rate=exc_rate_grid, inh_rate=inh_rate_grid, rho=rho, device=device).gen_uncorr_spk
    #input_gen = SNNInputGenerator(config, w=w_grid, input_mean=curr_mean_grid, input_std=curr_std_grid, rho=rho, device=device).gen_uncorr_normal_rv
    input_gen = SNNInputGenerator(config, w=w_grid, ei_spk_stats=ei_spk_stats,  device=device).gen_gamma_spk_EI

    snn_model = InteNFire(config, input_gen)

    print('Simulating SNN...')
    spk_count, V, t, spk_count_history = snn_model.run( config['T_snn'] , record_interval=config['record_interval'], show_message=True, device = device) # ms

    print(spk_count.sum())
    plt.figure()
    plt.plot(spk_count.mean(dim=0).squeeze().cpu().numpy(),'.',markersize=1)
    plt.savefig('tmp2.png')

    data_dict = {'spk_count': spk_count.cpu().numpy(),
    'config':config,
    'exc_rate': None, #exc_rate.cpu().numpy(),
    'inh_rate': None, #inh_rate.cpu().numpy(),
    'curr_mean':curr_mean.cpu().numpy(),
    'curr_std':curr_std.cpu().numpy(),
    'input_w':input_w.cpu().numpy(),
    #'exc_rate_grid':exc_rate_grid.cpu().numpy(),
    #'inh_rate_grid':inh_rate_grid.cpu().numpy(),
    'curr_mean_grid':curr_mean_grid.cpu().numpy(),
    'curr_std_grid': curr_std_grid.cpu().numpy(),
    'w_grid':w_grid.cpu().numpy(),
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
    
    

if __name__=='__main__':
    torch.set_default_dtype(torch.float64) #for accurate corrcoef estimate
    
    exp_id = sys.argv[1] #'2024_mar_30_mean_std'
    #device = 'cpu'
    device = 'cuda'
    
    print('Running experiment {}'.format(exp_id))
    #run(exp_id, T = 10e3, dt_snn= 1e-2, device = 'cuda', savefile=True)
    #run_w_rate_validation(exp_id, T = 10e3, dt_snn= 1e-2, device = device, savefile=True)
    run_w_rate_validation_EI(exp_id, T=1e3, dt_snn=1e-3, device = device, savefile=True)
    #spike_gen_debugger()

    #exp_id = '2024_mar_21_debug_low_std_high_mean'
    #exp_id = 'test_run'
    #run(exp_id, 0, savefile=True)
    
