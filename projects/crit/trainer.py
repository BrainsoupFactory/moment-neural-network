import torch
import time, json, logging, os
import numpy as np
from mnn.utils.dataloaders.mnist_loader import classic_mnist_loader
from mnn.mnn_core.nn.criterion import CrossEntropyOnMean
from projects.crit.model import MLP_static_recurrent
from projects.crit.static_recurrent_layer import gen_config

#from projects.crit.model_ann import MLP_static_recurrent
import torch.nn as nn
from torchvision import transforms

# def merge_bn_param(model, eps=1e-6):
#     ''' Calculate batchnorm-adjusted average feedforward current'''
#     bn_weight = model.layers[1].bn_mean.weight  #(hidden size)
#     bn_bias = model.layers[1].bn_mean.bias #(hidden size)
#     running_mean = model.layers[1].bn_mean.running_mean #(hidden size)
#     running_var = model.layers[1].bn_mean.running_var #(hidden size)
#     scaling_factor = bn_weight/torch.sqrt(running_var+eps) #(hidden size)
#     current_ff = bn_bias - running_mean*scaling_factor
#     return current_ff 

# one experiment one class, avoid excessive wrappers.
class trainer_MLP_static_recurrent_mnist():
    '''
        MLP with 1 hidden layer, replace hidden layer with a static recurrent layer
        Trained on mnist 
    '''
    @staticmethod
    def train(config):        
        if config['seed'] is None:            
            torch.manual_seed(int(time.time())) #use current time as seed
        else:
            torch.manual_seed(config['seed'])
        
        #sample_size = config['sample_size']        
        batch_size = config['batch_size'] #64
        #num_batches = int(sample_size/batch_size)
        num_epoch = config['num_epoch'] #50#1000
        lr = config['lr']#0.01
        momentum = config['momentum'] #0.9
        optimizer_name = config['optimizer_name']
        device = config['hidden_layer_config']['device']
        
        input_size = config['input_size']
        output_size = config['output_size']
        hidden_layer_size = config['hidden_layer_config']['NE']+config['hidden_layer_config']['NI']
        
        logging.info('Initializing model...')
        model = MLP_static_recurrent(input_size,hidden_layer_size,output_size, config=config['hidden_layer_config']).to(device)
        
        logging.info('Initializing dataloader...')
        train_dataloader, validation_dataloader = classic_mnist_loader(data_dir = './datasets/', train_batch=batch_size, test_batch=batch_size,  \
        transform_train = transforms.Compose([transforms.RandomCrop(28, padding=2), transforms.ToTensor()]), \
        transform_test = transforms.Compose([transforms.ToTensor()])   )

        logging.info('Initializing criterion...')
        criterion = CrossEntropyOnMean().to(device)

        params = model.parameters()
        
        if optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params, lr = lr, amsgrad = True) #recommended lr: 0.1 (Adam requires a much smaller learning rate than SGD otherwise won't converge)
        elif optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params, lr= lr, momentum= momentum) #recommended lr: 2
        elif optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr = lr, amsgrad = True, weight_decay=config['weight_decay']) #recommended lr: 0.1 (Adam requires a much smaller learning rate than SGD otherwise won't converge)
        else:
            print('Invalid optimizer name!')
            return

        # TODO: configure checkpoitns    
        model.checkpoint = {
            'epoch': [],
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'loss': [],
            'accuracy': [],
            'train_loss': [],
            'model_state_history':[],
            'optimizer_state_history':[]
            }
        
        t0 = time.perf_counter()
        batch_count = 0 #count the number of minibatches

        for epoch in range(num_epoch):            
            model.train()
            logging.info('Running epoch {}'.format(epoch))
            
            for i_batch, (data, target) in enumerate(train_dataloader):
                #logging.info( "--Time Elapsed: {}".format( int(time.perf_counter()-t0) ) )
                #logging.info('--Running batch {}, epoch {}'.format(i_batch, epoch))
                batch_count+=1
                
                optimizer.zero_grad()

                data = data.view(data.shape[0], -1).to(device) #flatten the image
                target = target.to(device)

                # stimulus transduction to firing stats (Poisson stats)
                input_mean = data*config['stim_trans'] 
                input_var = data*config['stim_trans']

                #input_mean = data*1.0                 # use constant current, no variance
                #input_var = torch.zeros(input_mean.shape)

                output_mean, output_var = model.forward(input_mean, input_var)
                #output_mean = model.forward(input_mean)
                
                loss = criterion(output_mean, target)

                # Calculate the regularization term (L2 regularization)
                #regularization_term = sum(torch.norm(param)**2 for param in model.parameters())
                if config['reg_factor']:
                    #ff_current = merge_bn_param(model)
                    #regularization_term = torch.sum(ff_current.pow(2.0))
                    bn_bias = model.layers[1].bn_mean.bias
                    bn_weight = model.layers[1].bn_mean.weight
                    regularization_term = torch.sum(bn_bias.pow(2.0) + bn_weight.pow(2.0))
                    # Add the regularization term to the loss
                    loss_reg = config['reg_factor']*regularization_term
                    #print('vanilla loss', loss.item())
                    #print('regularizer loss', loss_reg.item())
                    
                    loss = loss + loss_reg
                
                loss.backward()
                
                # apply gradient clipping
                if config['max_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])

                model.checkpoint['train_loss'].append(loss.item())
                
                if config['debug']:
                    model.checkpoint['model_state_history'].append(model.state_dict())
                    model.checkpoint['optimizer_state_history'].append(optimizer.state_dict())
                
                # weight clipping
                if config['max_grad_norm']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = config['max_grad_norm'])
                elif config['max_grad_value']:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value = config['max_grad_value'])
                
                optimizer.step()
                
            if epoch % 1 == 0:
                logging.debug('Running validation epoch {}/{}'.format(epoch,num_epoch))
                
                correct_predictions = 0
                total_predictions = 0
                
                with torch.no_grad():
                    model.eval()
                    for i_batch, (data, target) in enumerate(validation_dataloader):

                        data = data.view(data.shape[0], -1).to(device) #flatten the image
                        target = target.to(device)

                        input_mean = data*config['stim_trans'] 
                        input_var = data*config['stim_trans']
                        #input_mean = data*1.0
                        #input_var = torch.zeros(input_mean.shape)

                        output_mean, output_var = model.forward(input_mean, input_var)
                        #output_mean = model.forward(input_mean)
                
                        loss = criterion(output_mean, target)
                        
                        # record validation accuracy
                        _, predicted = torch.max(output_mean, 1)
                        correct_predictions += (predicted == target).sum().item()
                        total_predictions += target.size(0)

                    test_acc = correct_predictions / total_predictions
                    logging.info( 'Validation accuracy: {}'.format(test_acc) )                
                    model.checkpoint['accuracy'].append(test_acc)
                    model.checkpoint['loss'].append(loss.item())
                    model.checkpoint['epoch'].append(epoch)
                    
                    logging.info( "Loss: {}".format(loss.item()) )
                    logging.info( "Time Elapsed: {}".format( int(time.perf_counter()-t0) ) )

        #print("Number of batches: ", num_batches)
        print("Batch size: ",batch_size)
        print("Learning rate: ", lr)
        print("Momentum: ", momentum)
        print("Time Elapsed: ", time.perf_counter()-t0)
        print("===============================")
        
        model.checkpoint['model_state_dict'] =  model.state_dict()
        model.checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        return model

if __name__ == "__main__":    
    
    hidden_layer_config = gen_config(N=1250, ie_ratio=5.0, bg_rate=40.0, device='cpu')

    trainer_config = {'sample_size': None,
              'batch_size': 100,
              'num_epoch': 1,
              'lr': 0.001,
              'momentum': 0.9,
              'optimizer_name': 'AdamW',
              'num_hidden_layers': None,
              'input_size': 784,
              'output_size': 10,
              'trial_id': int(time.time()),
              'save_dir': './projects/crit/runs/test/',
              'dataset_name': None,              
              'seed': None,
              'debug': False, # cache all intermediate outputs & weights. WARNING: consumes large memory
              'max_grad_norm': None, # gradient clipping. set to None to turn off
              'max_grad_value': None, #may better log it and see if there is any outliers
              'hidden_layer_config': hidden_layer_config, 
              'reg_factor': 1e-3, # regularization factor in the loss function
              'stim_trans':0.1, 
        }
    
    torch.cuda.set_device(0)
    logging.basicConfig(level=logging.INFO) #this prints debug messages

    model = trainer_MLP_static_recurrent_mnist.train(trainer_config)    
    file_name = 'N{}_ie_ratio{}_bg_rate{}_{}'.format(hidden_layer_config['NE']+hidden_layer_config['NI'], hidden_layer_config['ie_ratio'], hidden_layer_config['bg_rate'], trainer_config['trial_id'])
    
    if not os.path.exists(trainer_config['save_dir']):
        os.makedirs(trainer_config['save_dir'])
    torch.save(model.checkpoint, trainer_config['save_dir']+file_name+'.pt') 
    with open(trainer_config['save_dir']+'{}_config.json'.format(file_name),'w') as f:
        json.dump(trainer_config,f)
    print('Results saved to '+trainer_config['save_dir']+file_name+'.pt')

# TODO: [priority] train one network for sufficient number of epochs e.g. 15 and verify its properties
# 1. how much does bias current stats change after training? (this determines if the resting state stays critical)
# 2. how much is the feedforward input current, relative to background current?
# 3. how close or far away the network is to criticality during tasks?
# TODO: clean up junk code during debugging 
# TODO: centralize config file for reproducibility. dictionary of dictionary allowed.
# TODO: test code with small network before systematic dispatch
