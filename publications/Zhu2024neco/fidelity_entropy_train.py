from mnn import utils
from tqdm import tqdm
import torch

def train_datasets(args):
    criterion = {'name': 'FidelityLoss', 'args': {'alpha': 0.8, 'use_full': True}, 'source': 'mnn_core'}
    args.CRITERION = criterion
    datasets = ['mnist', 'fashion_mnist', 'cifar10']
    for data in datasets:
        if data == 'mnist':
            args.dataset = 'mnist'
            args.MODEL['mnn_mlp']['structure'] = [784, 1000]
            args.DATAAUG_TRAIN['RandomCrop']['size'] = 28
        elif data == 'fashion_mnist':
            args.dataset = 'FashionMNIST'
            args.MODEL['mnn_mlp']['structure'] = [784, 1000]
            args.DATAAUG_TRAIN['RandomCrop']['size'] = 28
        else:
            args.dataset = 'CIFAR10'
            args.MODEL['mnn_mlp']['structure'] = [3072] + [1000] * 3
            args.DATAAUG_TRAIN['RandomCrop']['size'] = 32
        
        for i in tqdm(range(2)):
            args.save_name = '{}_alpha={}'.format(data, i)
            args.CRITERION['args']['alpha'] = i           
            utils.training_tools.general_train.general_train_pipeline(args)
        
        i = 1
        trainable = True
        args.MODEL['mnn_mlp']['params_for_criterion'] =  trainable
        args.save_name = '{}_alpha={}_trainable'.format(data, i)
        args.CRITERION['args']['trainable_weight_loss'] = True 
        args.CRITERION['args']['alpha'] = i 
        utils.training_tools.general_train.general_train_pipeline(args)
        
        trainable = False
        args.MODEL['mnn_mlp']['params_for_criterion'] =  trainable
        args.save_name = '{}_alpha={}_uniform'.format(data, i)
        args.CRITERION['args']['trainable_weight_loss'] = False
        args.CRITERION['args']['fidelity_weight'] = (torch.ones(9) / 9).tolist()
        args.CRITERION['args']['alpha'] = i 
        utils.training_tools.general_train.general_train_pipeline(args)
        
        

if __name__ == '__main__':
    config = utils.training_tools.deploy_config()
    train_datasets(config)