import torch
import matplotlib.pyplot as plt
import sys, os
from projects.pre2023.snn_simulator_torch import *


class SNNInputGenerator():
    def __init__(self, config, input_mean=None, input_std=None, input_corr=None, rho=None, device='cpu'):
        self.NE = config['NE']
        self.NI = config['NI']
        self.N = config['NE']+config['NI']
        self.batchsize = config['batchsize']
        self.device = device
        
        self.input_mean = input_mean
        self.input_std = input_std
        
        # TODO: issue I can't just let rho to be the same everywhere? can't guarantee positive definiteness


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

def run(exp_id, indx, savefile=False, savefig=False):
    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    if not os.path.exists(path):
        os.makedirs(path)
    
    print('Setting up parameters to be swept...')
    
    batchsize = 10000 # number of independent samples (make this large to avoid no spikes) 
    T = 1e3 # ms
    device= 'cuda'
    dt_snn = 1e-2 

    #indx = 0

    rho = 0 # for mean, std mapping need independent samples
    input_mean = torch.linspace(-1,4,21, device=device)
    input_std = torch.linspace(0,5,21, device=device)[1:] # no noise when std=0

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
    'indx':indx,
    'spk_count_history':spk_count_history,
    't':t,
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

if __name__=='__main__':
    torch.set_default_dtype(torch.float64) #for accurate corrcoef estimate
    
    #exp_id = '2024_mar_21_debug_low_std_high_mean'
    exp_id = 'test_run'
    run(exp_id, 0, savefile=True)
    
