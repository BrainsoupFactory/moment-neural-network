import argparse
import os
import random

import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn
import torchvision
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import warnings

try:
    from torch.distributed.optim import ZeroRedundancyOptimizer
    zero_redundancy_optimizer_available = True
except ImportError:
    warnings.warn('Try to import ZeroRedundancyOptimizer for distributed training but failed')
    zero_redundancy_optimizer_available = False


from ... import mnn_core
from ... import models
from .. import dataloaders

_SPECIAL_ARGS = ['LR_SCHEDULER', 'OPTIMIZER', 'DATASET', 'DATALOADER']


class PrepareMethods:
    @staticmethod
    def prepare_optimizer(net, optimizer, *args, **kwargs):
        return optimizer(filter(lambda p: p.requires_grad, net.parameters()), *args, **kwargs)

    @staticmethod
    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True

    @staticmethod
    def device_prepare(is_cpu: bool = False):
        if is_cpu:
            device = 'cpu'
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                cudnn.benchmark = True
        return device


def make_model(model_args: dict):
    meta_info: dict = model_args['meta']
    if meta_info.get('arch', '') == 'mnn_mlp':
        model = make_mnn_mlp_model(model_args[meta_info['mlp_type']], meta_info['mlp_type'])
    else:
        model = None
    return model

def make_ann_model(model_args):
    meta_info = model_args['meta']
    cnn_args = model_args.get('backbone', None)
    model = getattr(torchvision.models, meta_info['cnn_type'])(pretrained=cnn_args['pretrained'])
    if cnn_args['frozen']:
        freeze_layers(model, cnn_args.get('freeze_layer', None))
    return model

def make_mnn_mlp_model(model_args, model_type='mnn_mlp'):

    if model_type == 'mnn_mlp':
        mlp_net = models.MnnMlp(**model_args)
    elif model_type == 'snn_mlp':
        mlp_net = models.SnnMlp(**model_args)
    else:
        mlp_net = None
    return mlp_net

def freeze_parameters(modules):
    for p in modules.parameters():
        p.requires_grad = False
    return modules

def freeze_layers(modules, layer_depth=None):
    if layer_depth is None:
        modules = freeze_parameters(modules)
    else:
        for i in range(layer_depth):
            modules[i] = freeze_parameters(modules[i])
    return modules

def make_transforms_compose(params):
    order = params['aug_order']
    transform_list = []
    for key in order:
        if key == 'ToTensor':
            transform_list.append(torchvision.transforms.ToTensor())
        else:
            transform_list.append(getattr(torchvision.transforms, key)(**params[key]))
    transform = torchvision.transforms.Compose(transform_list)
    return transform

def make_dataloader(dataset, args, train=True):
    if args.distributed:
        if train:
            sampler = DistributedSampler(dataset)
        else:
            sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None
    if train:
        data_loader = DataLoader(dataset, batch_size=args.bs, shuffle=sampler is None, sampler=sampler,
                                 pin_memory=args.pin_mem, num_workers=args.workers)
    else:
        data_loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, pin_memory=args.pin_mem,
                                 num_workers=args.workers, sampler=sampler)
    return data_loader


def make_image_fold_dataset(args):
    transform_train = make_transforms_compose(args.DATAAUG_TRAIN)
    transform_val = make_transforms_compose(args.DATAAUG_VAL)
    train_dir = os.path.join(args.data_dir, args.train_split)
    val_dir = os.path.join(args.data_dir, args.val_split)
    train_set = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    val_set = torchvision.datasets.ImageFolder(val_dir, transform=transform_val)
    return train_set, val_set

def make_torchvision_dataset(args, data_dir='./data/'):
    transform_train = make_transforms_compose(args.DATAAUG_TRAIN)
    transform_val = make_transforms_compose(args.DATAAUG_VAL)
    train_set = getattr(torchvision.datasets, args.dataset)(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = getattr(torchvision.datasets, args.dataset)(root=data_dir, train=False, download=True, transform=transform_val)
    return train_set, test_set
    
def prepare_dataloader(args, data_dir='./data/'):
    if args.dataset == 'mnist':
        transform_train = make_transforms_compose(args.DATAAUG_TRAIN)
        transform_val = make_transforms_compose(args.DATAAUG_VAL)
        train_loader, test_loader = dataloaders.classic_mnist_loader(data_dir, args.bs, transform_train=transform_train,
                                                                        transform_test=transform_val)
    else:
        train_set, test_set = make_torchvision_dataset(args, data_dir)
        
        train_loader = make_dataloader(train_set, args, True)
        test_loader = make_dataloader(test_set, args, False)
    return train_loader, test_loader

def make_optimizer(params_group, args):
    if args.distributed and zero_redundancy_optimizer_available:
        optimizer = ZeroRedundancyOptimizer(params_group, getattr(torch.optim, args.OPTIMIZER['name']),
                                            **args.OPTIMIZER['args'])
    else:
        optimizer = getattr(torch.optim, args.OPTIMIZER['name'])(params_group, **args.OPTIMIZER['args'])
    return optimizer

def make_schedule(optimizer, args):
    scheduler = getattr(torch.optim.lr_scheduler, args.LR_SCHEDULER['name'])(optimizer=optimizer,  **args.LR_SCHEDULER['name'])
    return scheduler

def prepare_optimizer_scheduler(params_group, args):
    if args.OPTIMIZER is None:
        optimizer = torch.optim.AdamW(params_group, lr=args.lr)
    else:
        optimizer = make_optimizer(params_group, args)

    if args.LR_SCHEDULER is None:
        scheduler = None
    else:
        scheduler = make_schedule(optimizer, args)
    return optimizer, scheduler

def make_criterion(criterion_args: dict):
    args = criterion_args.get('args', None)
    if criterion_args['source'] == 'pytorch':
        criterion = getattr(torch.nn, criterion_args['name'])
    else:
        criterion = getattr(mnn_core.nn, criterion_args['name'])
    if args is None:
        criterion = criterion()
    else:
        criterion = criterion(**args)
    return criterion

def prepare_criterion(args):
    try:
        criterion = make_criterion(args.CRITERION)
    except AttributeError:
        criterion = mnn_core.nn.CrossEntropyOnMean()
    return criterion

def prepare_data_augmentation(args):
    train = make_transforms_compose(args.DATAAUG_TRAIN)
    test = make_transforms_compose(args.DATAAUG_VAL)
    return train, test

def read_yaml_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def set_config2args(args):
    cfg = read_yaml_config(args.config)
    args = reset_special_args(args)
    for key, value in cfg.items():
        setattr(args, key, value)
    return args

def reset_special_args(args):
    for key in _SPECIAL_ARGS:
        setattr(args, key, None)
    return args

def prepare_args(parser):
    args = parser.parse_args()
    args.save_path = getattr(args, 'dump_path', './checkpoint/') + args.dir + '/'
    if args.config is not None:
        args = set_config2args(args)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.world_size = torch.cuda.device_count()
    args.distributed = args.world_size > 1 and args.multiprocessing_distributed
    args.local_rank = 0
    if torch.cuda.is_available() and not args.cpu:
        args.use_cuda = True
    else:
        args.use_cuda = False
    return args

class TempArgs:
    def __init__(self) -> None:
        pass
    
def model_generator(checkpoint_path, save_name, to_cuda=False, resume_model=True, local_rank=0, resume_best=False, make_func=None):
    if make_func is None:
        config = read_yaml_config('{}{}_config.yaml'.format(checkpoint_path, save_name))
        model = make_model(config['MODEL'])
    else:
        args = TempArgs()
        args.config = '{}{}_config.yaml'.format(checkpoint_path, save_name)
        set_config2args(args)
        model = make_func.make_model(args)
        
    if resume_model:
        if resume_best:
            ckpt = torch.load('{}{}_best_model.pth'.format(checkpoint_path, save_name), map_location='cpu')
        else:
            ckpt = torch.load('{}{}.pth'.format(checkpoint_path, save_name), map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
    if to_cuda:
        model.cuda(local_rank)
    model.eval()
    return model

def config_mnn_activation(activation_config: dict):
    for key, value in activation_config.items():
        mnn_core.set_core_attr(key, value)
        
def deploy_config(is_parse: bool = True):
    parser = argparse.ArgumentParser(description='Pytorch Mnn Training Template')
    # my custom parameters
    parser.add_argument('-c', '--config', default=None, type=str, metavar='FILE', help='YAML config file')
    parser.add_argument('--bs', default=50, type=int, help='batch size')
    parser.add_argument('--resume', action='store_true', default=False, help='resume from checkpoint')
    parser.add_argument('-p', '--print-freq', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--dir', default='mnist', type=str, help='dir path that used to save checkpoint')
    parser.add_argument('--cpu', action='store_true', default=False, help='Use CPU only or not')
    parser.add_argument('--trials', default=1, type=int, help='number of trials to run')
    parser.add_argument('--dataset_type', default='poisson', type=str, help='methods used in data loaders')
    parser.add_argument('--dataset', default='mnist', type=str, help='type of dataset')
    parser.add_argument('--data_dir', default='./data/', type=str, help='type of dataset')
    parser.add_argument('--eps', default=0.1, type=float, help='eps')
    parser.add_argument('--gpu', default=None, type=str, help='specify gpu idx if use gpu')
    parser.add_argument('--which_run', default='main', type=str, help='specify the model to run')
    parser.add_argument('--save_name', default='mnn_net', type=str, help='alias to save net')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    # Distributed Data Parallel setting
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--workers', default=1, type=int,
                        help='num workers for dataloader. ')
    parser.add_argument('--multiprocessing_distributed', action='store_true', default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')

    # Learning rate schedule parameters
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 25)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    if is_parse:
        args = prepare_args(parser)
        return args
    else:
        return parser
