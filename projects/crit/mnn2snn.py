
import torch
import json, os, fnmatch
import numpy as np

class MNN2SNN():
    def __init__(self, folder_name):
        self.path = './projects/crit/runs/{}/'.format(folder_name)
        self.file_list = os.listdir(self.path)
        self.meta_data = np.load(self.path+'meta_data.npz')     
        return
        
    def load_result(self,indx):
        '''Load trained mnn model, given the trial id'''
        if isinstance(indx, int):
            indx = str(indx).zfill(3)        
        
        #load config file
        print('Loading config file...')
        config_file = fnmatch.filter(self.file_list, indx+'*_config.json')[0]        
        with open(self.path +'/'+config_file) as f:
            trainer_config = json.load(f)
        
        #load model
        print('Loading model...')
        result = fnmatch.filter(self.file_list, indx+'*.pt')[0]
        checkpoint = torch.load(self.path +'/'+result)
        model_state_dict = checkpoint['model_state_dict']
        
        #input_size = config['input_size']
        #output_size = config['output_size']
        #hidden_layer_size = config['hidden_layer_config']['NE']+config['hidden_layer_config']['NI']
        
        #model = MLP_static_recurrent(input_size,hidden_layer_size,output_size, config=config['hidden_layer_config'])
        #model.load_state_dict(checkpoint['model_state_dict'])        
        #model.eval() #set to evaluation mode
        
        return trainer_config, model_state_dict
        
    @torch.no_grad()
    def merge_bn_param(self, model_state_dict, eps=1e-6):
        #keys: ['layers.0.weight', 'layers.1.bn_mean.weight', 'layers.1.bn_mean.bias', 
        #'layers.1.bn_mean.running_mean', 'layers.1.bn_mean.running_var', 
        #'layers.1.bn_mean.num_batches_tracked', 'layers.3.weight']
        print('Merging batchnorm parameters into feedforward weight and current...')
        Wff = model_state_dict['layers.0.weight']  # (hidden size x input_size)
        bn_weight = model_state_dict['layers.1.bn_mean.weight'] #(hidden size)
        bn_bias = model_state_dict['layers.1.bn_mean.bias'] #(hidden size)

        running_mean = model_state_dict['layers.1.bn_mean.running_mean'] #(hidden size)
        running_var = model_state_dict['layers.1.bn_mean.running_var'] #(hidden size)

        scaling_factor = bn_weight/torch.sqrt(running_var+eps) #(hidden size)

        Wff = Wff*scaling_factor.unsqueeze(-1)
        current_ff = bn_bias - running_mean*scaling_factor
        
        W_rec = None
        W_out = model_state_dict['layers.3.weight']

        return Wff, current_ff, W_rec, W_out 
    
    @staticmethod
    def update_config_snn(trainer_config): #update config file for snn
        ''' Use base configs from MNN and add SNN specific configs'''
        # NB config contains base mnn config, and hidden_layer_config is included as a sub dict.
        # The hidden_layer_config contains the base SNN config.
        trainer_config['hidden_layer_config']['dt_snn'] = 0.01 #ms
        trainer_config['hidden_layer_config']['T_snn'] = int(1e3)
        trainer_config['hidden_layer_config']['delay_snn'] = 1.5 #ms
        
        return trainer_config


if __name__=='__main__':
    ''' Quick demo: '''
    m2s = MNN2SNN('mnn_vary_ie_ratio_5trials_fix_validation')
    trainer_config, model_state_dict = m2s.load_result(0)
    Wff, current_ff, W_rec  = m2s.merge_bn_param(model_state_dict)
    trainer_config = m2s.update_config_snn(trainer_config)

    print('config for mnn:', trainer_config)
    print('model state dict keys:', list(model_state_dict))

    print('Shape of merged weight matrix: ', Wff.shape)
    print('Shape of merged bias: ', current_ff.shape)
    


