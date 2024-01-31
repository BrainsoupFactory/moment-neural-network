import numpy as np
import torch
import time
from mnn.mnn_core.nn.linear import LinearNoRho
from mnn.mnn_core.nn.batch_norm import BatchNorm1dNoRho
from projects.crit.static_recurrent_layer import gen_config, gen_synaptic_weight, StaticRecurrentLayer
import logging
from projects.crit.activation import MnnActivationNoRho

#torch.set_default_tensor_type(torch.DoubleTensor)
#seed = 5
#torch.manual_seed(seed)

# model definition
class MLP_static_recurrent(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, cache = False):
        super(MLP_static_recurrent, self).__init__()
        
        feedforward = LinearNoRho(input_size, hidden_layer_size)
        # TODO: compare removing connection to inhibitory neurons

        # TODO: compare adding batchnorm
        logging.info('Initializing batchnorm to feedforward layer...')
        bn = BatchNorm1dNoRho(hidden_layer_size)
        bn.bn_mean.weight.data.fill_(1.0)
        bn.bn_mean.bias.data.fill_(0.0) #set external current to 0 since we already have a background input

        logging.info('Initializing static recurrent layers...')
        config = gen_config(N=hidden_layer_size, ie_ratio=4.0, bg_rate=15.0)
        config['conn_prob'] = 0.2 # double default value
        W = gen_synaptic_weight(config)
        logging.debug('W shape = {}'.format( W.shape))        
        static_recurrent = StaticRecurrentLayer(W, config)
        
        # TODO: compare removing readout from inhibitory neurons 
        read_out = LinearNoRho(hidden_layer_size, output_size)
        
        self.layers = torch.nn.ModuleList([feedforward, bn, static_recurrent, read_out])
        
#        for i in range(len(self.layers)-1): #initialization for batchnorm
#            self.layers[i].bn.bn_mean.weight.data.fill_(1.0)
#            self.layers[i].bn.bn_mean.bias.data.fill_(0.0)
#        self.layers[0].bn.bn_mean.eps = 1.0 #set eps in the first layer to 1 (as the batch variance is 0 for mu)
        
        self.cache = cache
        self.cache_u = []
        self.cache_s = []
        self.cache_rho = []
            #self.layers[i].bn.ext_std.data.fill_(0) #set initial values all to 0
            #init.uniform_(self.layers[i].bn.ext_std, 1, 3) #actually default was pretty good
        return

    def forward(self, u, s):
        for i in range(len(self.layers)):
            logging.debug('Forward pass for layer {}'.format(i))
            logging.debug('Input dimensions {}'.format(u.shape))
            
            u, s = self.layers[i].forward(u, s)
            
            logging.debug('Output dimensions {}'.format(u.shape))            
            
            if self.cache: # don't enable this during training otherwise will cause memory overflow
                self.cache_u.append(u)
                self.cache_s.append(s)
                #self.cache_rho.append(rho)
        return u, s

if __name__=='__main__':
    # testing
    torch.cuda.set_device(0)
    logging.basicConfig(level=logging.DEBUG) #this prints debug messages
    

    input_size = 100
    hidden_size = 1250
    output_size = 10

    batchsize = 100
    u = torch.rand( batchsize, input_size)
    s = torch.rand( batchsize, input_size)

    model = MLP_static_recurrent(100,1250,10)
    model.forward(u,s)

