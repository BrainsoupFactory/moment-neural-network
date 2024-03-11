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
    def __init__(self, input_size, hidden_layer_size, output_size, cache = False, config = None):
        super(MLP_static_recurrent, self).__init__()
        
        feedforward = LinearNoRho(input_size, hidden_layer_size)
        # TODO: compare removing connection to inhibitory neurons

        logging.info('Initializing batchnorm to feedforward layer...')
        bn = BatchNorm1dNoRho(hidden_layer_size)
        bn.bn_mean.weight.data.fill_(1.0)
        bn.bn_mean.bias.data.fill_(0.0) #set external current to 0 since we already have a background input

        logging.info('Initializing static recurrent layers...')
        
        if config==None:
            logging.info('No config specified for static_reurrent_layer. Generating default config...')
            config = gen_config(N=hidden_layer_size, ie_ratio=4.0, bg_rate=20.0)
            
        W = gen_synaptic_weight(config)
        logging.debug('W shape = {}'.format( W.shape))
        static_recurrent = StaticRecurrentLayer(W, config)
        
        # TODO: compare removing readout from inhibitory neurons 
        read_out = LinearNoRho(hidden_layer_size, output_size)
        
        self.layers = torch.nn.ModuleList([feedforward, bn, static_recurrent, read_out])

        self.cache = cache
        self.cache_u = []
        self.cache_s = []
        self.cache_rho = []

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
    hidden_size = 12500
    output_size = 10

    batchsize = 100
    device = 'cuda'
    u = torch.rand( batchsize, input_size, device=device)
    s = torch.rand( batchsize, input_size, device=device)

    #model = MLP_static_recurrent(input_size,hidden_size,output_size)
    config = gen_config(N=hidden_size, ie_ratio=4.0, bg_rate=20.0, device=device)
 #   config = gen_config(hidden_size, 4.0 , 20)    
    model = MLP_static_recurrent(input_size, hidden_size, output_size, config=config).to(device)
    
    model.forward(u,s)
