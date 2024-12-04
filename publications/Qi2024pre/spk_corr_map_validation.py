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

        if not input_corr:    
            if rho>0:
                input_corr = torch.ones(self.N,self.N, device=device)*rho                
            else: # attempt to deal with negative corr
                input_corr = torch.ones(self.N,self.N, device=device)*rho
                m = int(self.N/2)
                input_corr[:m,:m] *= -1.0
                input_corr[m:,m:] *= -1.0
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

def run(indx, exp_id, T = 1e3, device = 'cuda', savefile=False, savefig=False):
    path =  './projects/pre2023/runs/{}/'.format( exp_id )
    if not os.path.exists(path):
        os.makedirs(path)
    
    print('Setting up parameters to be swept...')
    
    batchsize = 10000 # number of independent samples (make this large to avoid no spikes)         
    dt_snn = 0.01

    #indx = int(sys.argv[1])

    rho = np.linspace(-1,1,41)[1:][indx] #0.3 # scalar rho (for this tricl, negative rho doesn't work) 
    input_mean = torch.linspace(-1,4,11, device=device)
    input_std = torch.linspace(0,5,11, device=device)[1:] # no noise when std=0

    mean_grid, std_grid = torch.meshgrid(input_mean, input_std, indexing='ij')
    
    # duplicate inputs
    #input_mean = torch.cat((input_mean, input_mean.clone()), dim=0)
    #input_std = torch.cat((input_std, input_std.clone()), dim=0)
    
    # inner dim: std, outer dim: mean
    mean_grid = mean_grid.flatten().unsqueeze(0)
    std_grid = std_grid.flatten().unsqueeze(0)

    # duplicating input stats
    mean_grid = torch.cat((mean_grid, mean_grid.clone()), dim=1)
    std_grid = torch.cat((std_grid, std_grid.clone()), dim=1)
    

    print('Setting up SNN model...')
    num_neurons = mean_grid.shape[1]
    
    config = gen_config(batchsize, num_neurons, T, device=device, dt_snn = dt_snn )
    print('rho = ', rho)
    if np.abs(rho)>1e-6:
        input_gen = SNNInputGenerator(config, input_mean=mean_grid, input_std=std_grid, rho=rho, device=device).gen_corr_normal_rv
    else:
        print('Using uncorrelated input.')
        input_gen = SNNInputGenerator(config, input_mean=mean_grid, input_std=std_grid, rho=rho, device=device).gen_uncorr_normal_rv
    
    #''' debug. plot input current'''
    #input_current = self.input_gen(self.dt, device=device)

    snn_model = InteNFire(config, input_gen)

    print('Simulating SNN...')
    spk_count, V, t = snn_model.run( config['T_snn'] , show_message=True, device = device) # ms

    
    data_dict = {'spk_count': spk_count.cpu().numpy(),
    'config':config,
    'mean_grid':mean_grid.cpu().numpy(),
    'std_grid':std_grid.cpu().numpy(),
    'input_mean':input_mean.cpu().numpy(),
    'input_std':input_std.cpu().numpy(),
    'rho':rho,
    'indx':indx,
    'T':T,
    'dt_snn':dt_snn,
    'batchsize':batchsize,
    }
    
    if savefile:
        #filename = str(indx).zfill(3) +'_'+str(int(time.time())) + '.npz'
        filename = str(indx).zfill(3) +'.npz'
        np.savez(path+filename, **data_dict)
        print('Results saved to: ', path+filename)
    
    if savefig:
        print('Calculating spike count stats...')
        snn_rate = torch.mean(spk_count, dim=0)/config['T_snn']*1e3
        snn_fano_factor = torch.var(spk_count,dim=0)/torch.mean(spk_count,dim=0)
        snn_corr = torch.corrcoef(spk_count.T) # corr coef between columns        
        #plt.figure()
        #plt.imshow(spk_count) # this should have shape: batchsize (100) x neurons (420)
        #plt.colorbar()
        #plt.savefig('tmp_snn_spk_count.png')
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(snn_rate.cpu().numpy())
        plt.ylabel('Firing rate (sp/s)')
        plt.subplot(2,1,2)
        plt.plot(snn_fano_factor.cpu().numpy())
        plt.ylabel('Fano factor')
        plt.tight_layout()
        plt.savefig('test_snn_rate_FF.png')

        plt.figure()
        plt.imshow( snn_corr.cpu().numpy(), vmin=-1,vmax=1, cmap='coolwarm') #snn_corr could have too many entries?
        plt.colorbar()
        plt.savefig('test_snn_corr.png')
        
        plt.close('all')

if __name__=='__main__':
    torch.set_default_dtype(torch.float64) #for accurate corrcoef estimate
    
    if len(sys.argv) > 1: # if additional input arguments are provideds
        exp_id = sys.argv[2] #'2024_mar_31_longer_T'
        indx = int(sys.argv[1])
        print('Running experiment {}, indx {}'.format(exp_id, indx))        
        run(indx, exp_id, T = 10e3, device = 'cuda', savefile=True)
    else:
        exp_id = '2024_mar_21_longer_T'
        
        for indx in range(39):
            print('Running  ', indx)
            run(indx, exp_id, savefile=True)
    
