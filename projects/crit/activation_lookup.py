# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:29:22 2023

@author: Yang Qi

Generate or load direct look-up table with interpolation
"""
#from mnn_core.maf import MomentActivation
#from scipy.interpolate import RegularGridInterpolator
import numpy as np
from mnn.mnn_core import mnn_activate_no_rho
from projects.crit.util_interp import sub2value, bilinear_interpolation
import torch
from torch import nn
from matplotlib import pyplot as plt
import time

class MomentActivationLookup(nn.Module):    
    def __init__(self, device = 'cpu'):
        super(MomentActivationLookup, self).__init__()
        '''
        Generate or load direct look-up table with interpolation
        '''
        self.file = 'ma_lookup_tab.pt'
        self.num_pts = 200 #number of points (along one dimension)
        self.input_mean_grid = torch.linspace(-10, 10, self.num_pts, device=device)
        self.input_std_grid = torch.linspace(0, 20, self.num_pts, device=device)
        self.ma =  mnn_activate_no_rho
        # TODO: add support for rho later - if the speed up is significant
        try:
            print('Loading table...')
            U, S = self.load_table()
        except:
            print('Loading failed! Generating new look-up table...')
            U, S = self.gen_table() #only works on cpu...
            
        self.groundtruth_mean = U.to(device)
        self.groundtruth_std = S.to(device)
        # Interpolating Jacobian in backward pass is inefficient. Just use autograd. 
        # self.groundtruth_dudu 
        # self.groundtruth_duds
        # self.groundtruth_dsdu
        # self.groundtruth_dsds
        
        #self.groundtruth_linear_res = X
        #self.interp_mean = RegularGridInterpolator( (self.input_mean , self.input_std), U.T)
        #self.interp_std = RegularGridInterpolator( (self.input_mean , self.input_std), S.T)
        #self.interp_chi = RegularGridInterpolator( (self.input_mean , self.input_std), X.T)

    def gen_table(self):
        # TODO: double check order
        X, Y = torch.meshgrid(self.input_mean_grid, self.input_std_grid)
        groundtruth_mean, groundtruth_std = mnn_activate_no_rho(X,Y)
        # Jacobian can be recovered by passing ones into the backward pass; but inefficient this way. 
        # du/du, du/ds = backward(dL/du=1, dL/ds=0)
        # ds/du, ds/ds = backward(dL/du=0, dL/ds=1)
        # _,_ = mnn_activate_no_rho.backward()

        data_dict = {
            'groundtruth_mean': groundtruth_mean,
            'groundtruth_std': groundtruth_std
        }
        torch.save( data_dict, self.file)
        return groundtruth_mean, groundtruth_std


    def load_table(self):        
        dat = torch.load(self.file)
        U = dat['groundtruth_mean']
        S = dat['groundtruth_std']
        #X = dat['X']
        return U, S#, X
        
    def forward(self,input_mean,input_std):
        output_mean = bilinear_interpolation(self.input_mean_grid, self.input_std_grid, self.groundtruth_mean, input_mean, input_std)
        output_std = bilinear_interpolation(self.input_mean_grid, self.input_std_grid, self.groundtruth_std, input_mean, input_std)
        return output_mean, output_std

        

if __name__=='__main__':
    torch.cuda.set_device(0)

    ma = MomentActivationLookup()
    
#    input_mean = torch.linspace(-10,10,1000).unsqueeze(0).repeat(1000,1)
    input_mean = torch.linspace(-10,10,50).unsqueeze(0)
    input_std = torch.ones(input_mean.shape)*5

    t0 = time.perf_counter()
    output_mean, output_std = ma(input_mean, input_std)
    print('Time for interpolation: ', time.perf_counter()-t0)

    t0 = time.perf_counter()
    groundtruth_mean, groundtruth_std = mnn_activate_no_rho(input_mean,input_std)
    print('Time for computing ground truth: ', time.perf_counter()-t0)

    print('Average abs. error for mean: ', torch.mean( torch.abs(groundtruth_mean- output_mean) ))

    # plt.plot(input_mean.flatten(), output_mean.flatten())
    # plt.plot(input_mean.flatten(), groundtruth_mean.flatten())    
    # plt.plot(ma.input_mean_grid.flatten(), ma.groundtruth_mean[:,50].flatten(),'--')    
    # plt.savefig('temp.png')

    # test interp function with some random function

    f = lambda x,y: torch.cos(x)+torch.sin(y)
    x0 = torch.linspace(-1,1,100) 
    y0 = torch.linspace(-1,1,100)
    X0,Y0 = torch.meshgrid(x0,y0)
    Z0 = f(X0,Y0)

    x = torch.linspace(-1,1,30).unsqueeze(0)
    y = torch.ones(x.shape)
    z = bilinear_interpolation(x0,y0,Z0,x,y)

    print('Average abs. error: ', torch.mean( torch.abs(z-f(x,y)) ))
    # plt.figure()
    # plt.plot(x.flatten(), z.flatten())    
    # plt.plot(x.flatten(), f(x,y).flatten())    
    # plt.savefig('temp2.png')

    





    
    
    
