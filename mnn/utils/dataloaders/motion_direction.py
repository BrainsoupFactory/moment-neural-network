# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset
from mnn import mnn_core, utils

def random_loc_unit_disc(N, loc_generator=None, random_seed: int = 2):
     # Define input neuron locations (uniformly scattered on a disc)
    if loc_generator is None:
        loc_generator = torch.Generator()
    loc_generator.manual_seed(random_seed) #fix the seed for random neuron location
    
    th = 2*np.pi*torch.rand(N, generator=loc_generator) #random angles
    r = torch.rand(N, generator=loc_generator) + torch.rand(N, generator=loc_generator) #radius
    r[r>1] = 2 - r[r>1]
    x = r*torch.cos(th)
    y = r*torch.sin(th)
    return x, y

def hex_grid_unit_disc(N):
    N = N*1.53
    ratio = np.sqrt(3)/2 # cos(60°)        
    N_X = int(np.sqrt(N/ratio))
    N_Y = N // N_X    
    #STEP 1: generate square grid
    xv, yv = np.meshgrid(np.arange(N_X), np.arange(N_Y), sparse=False, indexing='xy')    
    #STEP 2: stretch x-coordinates
    xv = xv * ratio    
    #STEP 3: shift x-coordinates
    xv[::2, :] += ratio/2    
    #STEP 4: crop to unit circle
    xv = 2*xv/N_X/ratio/ratio - 1
    yv = 2*yv/N_Y - 1    
    r = np.sqrt(xv*xv+yv*yv)        
    xv = xv[r<1]
    yv = yv[r<1]    
    return xv, yv

class VisualOrientationDataset(Dataset):
    def __init__(self, sample_size, input_dim, frac=0.5, omega=1, c0=1, c_gain=5, cmax=0.16, k=5*np.pi, alpha=1, beta=1,
    transform=None, train=True, input_grid='random', contrast_scale='linear', dtype=torch.float, direct_decoding=False,
    target_scaling=1, dt=1e-2, duration=None, rate_policy='continuous', time_window=1) -> None:
        super().__init__()
        """
        Args:
            sample_size: the size of the dataset
            fix_rho: input correlation coefficient
            input_dim: 
            output_dim:
            frac: fraction of intensity detectors
            omega: angular frequency
            c0: global illuminance
            c_gain: contrast gain
            cmax: maximum contrast 0<cmax<1
            k:
            generator: pytorch random generator
            random_seed: 
        """
        self.sample_size = sample_size
        self.input_dim = input_dim
        self.frac = frac
        self.omega = omega
        self.c0 = c0
        self.c_gain = c_gain
        self.cmax = cmax
        self.k = k
        self.transform = transform
        self.train = train
        self.direct_decoding = direct_decoding
        self.target_scaling = target_scaling
        self.alpha = alpha
        self.beta = beta

        if input_grid == 'hexagonal':
            x, y = hex_grid_unit_disc(self.input_dim / 2)
            self.x_input = torch.from_numpy(np.concatenate((x, x)))
            self.y_input = torch.from_numpy(np.concatenate((y, y)))
            assert self.x_input.shape == self.y_input.shape
            self.input_dim = len(self.x_input)
        else:
            # fix neuron postion
            self.x_input, self.y_input = random_loc_unit_disc(self.input_dim)
        
        self.x_input = self.x_input.to(dtype=dtype)
        self.y_input = self.y_input.to(dtype=dtype)
        self.dx = self.x_input.view(1,self.input_dim,1) - self.x_input.view(1,1,self.input_dim)
        self.dy = self.y_input.view(1,self.input_dim,1) - self.y_input.view(1,1,self.input_dim)

        self.num_i_detector = int(self.input_dim * self.frac)

        n = int(np.sqrt(self.sample_size))
        if contrast_scale == 'linear':
            self.c = torch.linspace(0, self.cmax, n+1)[1:]
        else:
            # log
            self.c = torch.logspace(-1, 0, n) * self.cmax
        # current circshift function only works for theta on (-pi,pi)
        self.theta = torch.linspace(-np.pi, np.pi, n+1)[1:]
        
        self.dt = dt
        if duration is None:
            self.duration = int(2* np.pi / omega * 1)
        else:
            self.duration = duration
        self.time_steps = int(self.duration / self.dt)
        if rate_policy == 'discrete':
            # fix rate within the observe time window
            running_time = torch.arange(0, self.duration, time_window)
            self.running_time = running_time.unsqueeze(-1).repeat(1, int(time_window / dt)).flatten()
        else:
            # compute the rate at each time step
            self.running_time = torch.linspace(0, self.duration, self.time_steps)
        self.rate_policy = rate_policy
        self.time_window = time_window
    
    def gen_snn_input_rate(self, theta, c):
        # since the rate tensor may be too big for gpu, only generate rate and the spikes will generate on the fly.
        kx = self.k * torch.cos(theta)
        ky = self.k * torch.sin(theta)

        phase = kx * self.x_input.flatten() + ky * self.y_input.flatten()
        intensity_phase = phase[:self.num_i_detector] # shape (# intensity neurons)
        change_phase = phase[self.num_i_detector: ]

        intensity_rate = (self.c0 + self.c0  * self.c_gain * c * torch.cos(intensity_phase.unsqueeze(0) - self.omega * self.running_time.unsqueeze(-1))) * self.alpha
        change_rate = (self.c0 - self.c0 * self.c_gain * c * self.omega * torch.sin(change_phase.unsqueeze(0) - self.omega * self.running_time.unsqueeze(-1))) * self.beta
        rate = torch.cat([intensity_rate, change_rate], dim=-1).unsqueeze(-2)
        assert rate.size() == (self.time_steps, 1, self.input_dim) #  to meet the default shape（time, trial, neuron）
        return rate
    
    def gen_sample(self, batch_size, idx, theta=None, c=None): # num_timesteps, ext_input_type

        # STEP 1: generate random orientations and contrast
        if theta is None or c is None:
            if self.train:
                theta = torch.rand(batch_size, 1, 1) * 2 * np.pi - np.pi
                c = torch.rand(batch_size, 1) * self.cmax
            else:
                i,j = np.unravel_index(idx, (len(self.c),len(self.theta)) )
                theta = self.theta[j].view(batch_size, 1, 1)
                c = self.c[i].view(batch_size,1)
        else:
            theta = theta.reshape(batch_size, 1, 1)
            c = c.reshape(batch_size, 1)
        
        # STEP 2: input encoding
        input_mean = torch.ones(batch_size, self.input_dim) * self.c0
        input_mean[:, :self.num_i_detector] *= self.alpha
        input_mean[:, self.num_i_detector] *= self.beta
        signal_std = self.c0 / np.sqrt(2) * c * torch.ones(1, self.input_dim) * self.c_gain * np.sqrt(self.time_window)
        signal_std[:, :self.num_i_detector] *= self.alpha
        signal_std[:, self.num_i_detector:] = signal_std[:, self.num_i_detector:] * self.omega * self.beta

        input_std = torch.sqrt(signal_std * signal_std + input_mean)# add spontaneous variability
        kx = self.k * torch.cos(theta)
        ky = self.k * torch.sin(theta)

        phase = kx * self.dx + ky * self.dy
        input_corr = torch.cos(phase)
        input_corr[:,self.num_i_detector:,:self.num_i_detector] = -torch.sin(phase[:,self.num_i_detector:,:self.num_i_detector])
        input_corr[:,:self.num_i_detector,self.num_i_detector:] = torch.sin(phase[:,:self.num_i_detector,self.num_i_detector:])

        scaling = signal_std/input_std
        input_corr = input_corr*scaling.view(batch_size,1,self.input_dim)*scaling.view(batch_size,self.input_dim,1)
        torch.diagonal(input_corr, dim1=-1, dim2=-2).data.fill_(1.0)

        # STEP 3: target encoding
        if self.direct_decoding:
            target_mean = theta.squeeze(-1)
        else:
            target_mean = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        input_cov = mnn_core.nn.functional.compute_covariance(input_std, input_corr)
        return (input_mean.squeeze(), input_cov.squeeze()), target_mean.squeeze() * self.target_scaling


    def __len__(self):
        if self.train:
            return self.sample_size
        else:
            return self.c.numel() * self.theta.numel()
    
    def __getitem__(self, index):
        if isinstance(index, int):
            batch_size = 1
        else:
            index = index.numpy()
            batch_size = len(index)
        
        data, target = self.gen_sample(batch_size, index)
        return data, target

def motion_direction_dataloader(args):
    train_set = utils.datasets.VisualOrientationDataset(train=True, **args.DATASET)
    test_set = utils.datasets.VisualOrientationDataset(train=False, **args.DATASET)
    train_loader = utils.training_tools.make_dataloader(train_set, args, True)
    test_loader =  utils.training_tools.make_dataloader(test_set, args, False)
    return train_loader, test_loader
    


