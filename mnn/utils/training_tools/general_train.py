import os
import shutil
import tempfile
import time
from enum import Enum

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from . import general_prepare
from . import functional as func

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


def _get_batch_fmtstr(num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = _get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        info = '\t'.join(entries)
        print(info)
        return info

    def display_summary(self):
        entries = [self.prefix + " *"]
        entries += [meter.summary() for meter in self.meters]
        info = ' '.join(entries)
        print(info)
        return info


def to_cuda(data, local_rank):
    if isinstance(data, torch.Tensor):
        temp = data.cuda(local_rank, non_blocking=True)
    else:
        temp = list()
        for i in data:
            temp.append(i.cuda(local_rank, non_blocking=True))
    return temp


class TrainProcessCollections:
    
    def __init__(self):
        pass

    def set_random_seed(self, seed):
        general_prepare.PrepareMethods.seed_everything(seed)

    def make_model(self, args):
        model = general_prepare.make_model(model_args=args.MODEL)
        return model

    def prepare_dataloader(self, args):
        train_loader, test_loader = general_prepare.prepare_dataloader(args, getattr(args, 'data_dir', './data/'))
        return train_loader, test_loader

    def prepare_optimizer_scheduler(self, params_group, args):
        optimizer, scheduler = general_prepare.prepare_optimizer_scheduler(params_group, args)
        return optimizer, scheduler

    def prepare_criterion(self, args):
        try:
            criterion = general_prepare.make_criterion(args.CRITERION)
        except AttributeError:
            criterion = general_prepare.mnn_core.nn.CrossEntropyOnMean()
        return criterion

    def params_clip(self, model, MIN=-1, MAX=1):
        for p in model.parameters():
            p.data.clamp_(MIN, MAX)
        return model
    
    def params_frozen(self, model):
        for p in model.parameters():
            p.requires_grad = False
        return model

    def specify_params_group(self, model):
        return filter(lambda p: p.requires_grad, model.parameters())
    
    def input_preprocessing(self, data, args):
        if getattr(args, 'flatten_input', True):
            data = torch.flatten(data, start_dim=1)
        data = data * getattr(args, 'scale_factor', 1.)
        if getattr(args, 'input_prepare', None) == 'flatten_poisson':
            cov = torch.diag_embed(torch.abs(data))
        elif getattr(args, 'input_prepare', None) == 'poisson_no_rho':
            cov = torch.abs(data)
        elif getattr(args, 'input_prepare', None) == 'cov_embed':
            cov = torch.sqrt(torch.abs(data))
            cov = torch.einsum('b i, b j -> b i j', cov, cov)
        else:
            cov = None
            
        if getattr(args, 'background_noise', None) is not None and cov is not None:
            if data.size() == cov.size():
                cov = cov + torch.ones_like(cov, device=data.device) * getattr(args, 'background_noise')
            else:
                cov = cov + torch.eye(data.size(-1), device=data.device) * getattr(args, 'background_noise')
        if getattr(args, 'unsqueeze_input', None) is not None:
            data = data.unsqueeze(args.unsqueeze_input)
            if cov is not None:
                cov = cov.unsqueeze(args.unsqueeze_input)
        return data, cov
        
    def data2device(self, data, target, args):
        if args.use_cuda:
            data = data.cuda(args.local_rank, non_blocking=True)
            if isinstance(target, torch.Tensor):
                target = target.cuda(args.local_rank, non_blocking=True)
        
        data, cov = self.input_preprocessing(data, args)
        if cov is not None:
            return (data, cov), target
        else:
            return data, target

    def clip_model_params(self, model, args):
        clip_args: dict = getattr(args, 'CLIP_PARAMS', None)
        if clip_args is not None:
            for key in clip_args.keys():
                module_keys = key.split('.')
                sub_model = model
                for module_key in module_keys:
                    sub_model = getattr(sub_model, module_key)
                _ = self.params_clip(sub_model, MIN=clip_args[key]['min'], MAX=clip_args[key]['max'])

    def metric_init(self, data_loader, epoch, prefix='Epoch: [{}]'):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        #top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(data_loader),
            [batch_time, data_time, losses, top1],
            prefix=prefix.format(epoch))
        return batch_time, data_time, losses, top1, progress

    def reduce_distributed_info(self, args,  *metrics):
        temp = []
        for i in metrics:
            temp.append(func.DistributedOps.reduce_mean(i, args.nprocs))
        #loss = func.DistributedOps.reduce_mean(loss, args.nprocs)
        #acc1 = func.DistributedOps.reduce_mean(acc1, args.nprocs)
        #acc5 = func.DistributedOps.reduce_mean(acc5, args.nprocs)
        return temp

    def compute_model_output(self, model, inputs, args=None):
        # args used for custom operation, will pass the args through train and val
        output = model(inputs)
        return output
    
    def compute_loss(self, output, target, criterion, model=None, args=None, inputs=None):
        # will pass the model and args by default in train and val process to support custom operation
        loss = criterion(output, target)
        return loss

    def train_one_epoch(self, train_loader, model, criterion, optimizer, epoch, args,):
        batch_time, data_time, losses, top1, progress = self.metric_init(train_loader, epoch)

        # switch to train mode
        model.train()

        end = time.time()
        num_updates = epoch * len(train_loader)
        if getattr(args, 'save_epoch_state', False):
            if args.local_rank == 0:
                save_state = {
                    'epoch': epoch,
                    'arch': args.save_name,
                    'state_dict': model.state_dict(),
                }
                save_path = getattr(args, 'dump_path', './checkpoint/') + args.dir + '/'
                self.save_checkpoint(save_state, is_best=False, save_path=save_path,
                                     save_name=args.save_name + '_epoch_{}'.format(epoch))
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            self.clip_model_params(model, args)

            images, target = self.data2device(images, target, args)

            # compute output
            output = self.compute_model_output(model, images, args)
            loss = self.compute_loss(output=output, target=target, criterion=criterion, model=model, args=args, inputs=images)
            if isinstance(loss, tuple):
                loss, pred = loss
            else:
                pred = None

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            if getattr(args, 'max_grad_norm', None) is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            if getattr(args, 'max_grad_value', None) is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=args.max_grad_value)

            optimizer.step()

            # measure elapsed time
            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            num_updates += 1
            
            if isinstance(images, tuple):
                images = images[0]

            if getattr(args, 'task_type', 'classification') == 'classification':
                acc1, = self.score_function(output, target, topk=(1, ), pred_prob=pred)
                if args.distributed:
                    loss, acc1 = self.reduce_distributed_info(args, loss, acc1)
                top1.update(acc1.item(), images.size(0))
            # measure accuracy and record loss
            
            losses.update(loss.item(), images.size(0))
            #top5.update(acc5.item(), images.size(0))

            end = time.time()
            if i % args.print_freq == 0 and args.local_rank == 0:
                info = progress.display(i) + '\n'
                func.RecordMethods.writing_log(args.log_path, info)
        if args.local_rank == 0 and getattr(args, 'log_path', None) is not None:
            info = 'Training result: ' + progress.display_summary() + '\n'
            func.RecordMethods.writing_log(args.log_path, info)

    def validate(self, val_loader, model, criterion, args, epoch=0):
        batch_time, data_time, losses, top1, progress = self.metric_init(val_loader, epoch, 'Test [Epoch:{}]: ')
        """
        For regression task, one can simply modify the attribute 'task_type' in args to 'regression'
        """

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                images, target = self.data2device(images, target, args)

                # compute output
                output = self.compute_model_output(model, images, args)
                loss = self.compute_loss(output=output, target=target, criterion=criterion, model=model, args=args, inputs=images)
                if isinstance(loss, tuple):
                    loss, pred = loss
                else:
                    pred = None
                    
                if isinstance(images, tuple):
                    images = images[0]
                    
                # measure accuracy and record loss
                if getattr(args, 'task_type', 'classification') == 'classification':
                    acc1, = self.score_function(output, target, topk=(1, ), pred_prob=pred)
                    if args.distributed:
                        loss, acc1 = self.reduce_distributed_info(args, loss, acc1)
                    top1.update(acc1.item(), images.size(0))
                # measure accuracy and record loss
                losses.update(loss.item(), images.size(0))
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

            if args.local_rank == 0 and getattr(args, 'log_path', None) is not None:
                info = 'Validation result: ' + progress.display_summary() + '\n'
                func.RecordMethods.writing_log(args.log_path, info)
        if getattr(args, 'task_type', 'classification') == 'classification':
            return top1.avg
        else:
            return - losses.avg
    
    def score_function(self, output, target, *args, **kwargs):
        return self.accuracy(output, target, *args, **kwargs)

    def accuracy(self, output, target, topk=(1,), pred_prob=None):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            if pred_prob is None:
                if not isinstance(output, torch.Tensor):
                    output = output[0]
            else:
                output = pred_prob
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def save_checkpoint(self, state, is_best, save_path, save_name='checkpoint'):
        torch.save(state, save_path + save_name + '.pth')
        if is_best:
            shutil.copyfile(save_path + save_name + '.pth', save_path + save_name + '_best_model.pth')

    def resume_model(self, args, model, local_rank=0):
        if args.use_cuda:
            loc = 'cuda:{}'.format(local_rank)
        else:
            loc = 'cpu'
        save_path = getattr(args, 'dump_path', './checkpoint/') + args.dir + '/'
        resume_best = getattr(args, 'resume_best', False)
        if resume_best:
            checkpoint_path = save_path + args.save_name + '_best_model.pth'
        else:
            checkpoint_path = save_path + args.save_name + '.pth'
        checkpoint = torch.load(checkpoint_path, map_location=loc)
        continue_train = getattr(args, 'continue_train', True)
        if continue_train:
            args.start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint.get('best_acc1', None)
        model.load_state_dict(checkpoint['state_dict'])
        return args, model, best_acc1

    def resume_optimizer_scheduler(self, args, optimizer, lr_scheduler=None, local_rank=0):
        if args.use_cuda:
            loc = 'cuda:{}'.format(local_rank)
        else:
            loc = 'cpu'
        save_path = getattr(args, 'dump_path', './checkpoint/') + args.dir + '/'
        resume_best = getattr(args, 'resume_best', False)
        if resume_best:
            checkpoint_path = save_path + args.save_name + '_best_model.pth'
        else:
            checkpoint_path = save_path + args.save_name + '.pth'
            
        checkpoint = torch.load(checkpoint_path, map_location=loc)
        optimizer_state = checkpoint.get('optimizer_state', None)
        if lr_scheduler is not None:
            scheduler_state = checkpoint.get('scheduler_state', None)
        else:
            scheduler_state = None
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if lr_scheduler is not None and scheduler_state is not None:
            lr_scheduler.load_state_dict(scheduler_state)
        return optimizer, lr_scheduler

    def run_training(self, args, model, train_loader, val_loader, criterion, optimizer, lr_scheduler,
                    train_func, best_acc1, save_path, local_rank=0):
        best_epoch = args.start_epoch
        for epoch in range(args.start_epoch, args.epochs):
            if isinstance(train_loader.sampler, general_prepare.DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            
            if getattr(args, 'save_epoch_state', False) and args.local_rank == 0:
                save_state = {
                    'epoch': epoch,
                    'arch': args.save_name,
                    'state_dict': model.state_dict(),
                }
                train_func.save_checkpoint(save_state, is_best=False, save_path=save_path, save_name=args.save_name + '_epoch_{}'.format(epoch))
            
            train_func.train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)
            acc1 = train_func.validate(val_loader, model, criterion, args, epoch=epoch)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                best_epoch = epoch
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)
            if general_prepare.zero_redundancy_optimizer_available:
                if isinstance(optimizer, general_prepare.ZeroRedundancyOptimizer):
                    optimizer.consolidate_state_dict()
            if local_rank == 0:
                save_state = {
                    'epoch': epoch + 1,
                    'arch': args.save_name,
                    'best_acc1': best_acc1,
                    'best_epoch': best_epoch,
                    'state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': lr_scheduler.state_dict() if lr_scheduler is not None else None
                }
                train_func.save_checkpoint(save_state, is_best, save_path, save_name=args.save_name)
        if local_rank == 0:
            info = '-*- Summary: after {} epochs training, the model hit {}% top1 acc at epoch [{}]\n'.format(
                args.epochs - args.start_epoch, best_acc1, best_epoch)
            func.RecordMethods.writing_log(args.log_path, info)


def general_distributed_train_pipeline(local_rank, nprocs, args, train_func=TrainProcessCollections):
    train_func = train_func()
    args.local_rank = local_rank
    if args.seed is not None:
        train_func.set_random_seed(args.seed)
    
    if hasattr(args, 'MnnActivationConfig') and local_rank == 0:
        general_prepare.config_mnn_activation(args.MnnActivationConfig)
    best_acc1 = -np.inf

    func.DistributedOps.setup(local_rank, nprocs)
    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True

    save_path = getattr(args, 'dump_path', './checkpoint/') + args.dir + '/'
    args.log_path = save_path + args.save_name + '_log.txt'
    if local_rank == 0:
        func.RecordMethods.make_dir(save_path)
        func.RecordMethods.record_hyper_parameter(save_path, args.save_name, **args.__dict__)
    

    # Create model
    model = train_func.make_model(args)
    loc = 'cuda:{}'.format(local_rank)
    if args.resume:
        args, model, best_acc1 = train_func.resume_model(args, model, local_rank)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), 'initial_weight.pt')
        if local_rank == 0:
            torch.save(model.state_dict(), checkpoint_path)
        func.dist.barrier()
        model.load_state_dict(torch.load(checkpoint_path, map_location=loc))
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # prepare dataloader
    train_loader, val_loader = train_func.prepare_dataloader(args)

    # criterion, optimizer and lr_scheduler
    criterion = train_func.prepare_criterion(args).cuda(local_rank)
    params_group = train_func.specify_params_group(model)
    optimizer, lr_scheduler = train_func.prepare_optimizer_scheduler(params_group, args)

    if args.resume:
        optimizer, lr_scheduler = train_func.resume_optimizer_scheduler(args, optimizer, lr_scheduler, local_rank)

    if lr_scheduler is not None and args.start_epoch > 0:
        lr_scheduler.step(args.start_epoch)

    if args.evaluate:
        train_func.validate(val_loader, model, criterion, args)
        return

    # run training process
    train_func.run_training(args, model, train_loader, val_loader, criterion, optimizer, lr_scheduler,
                 train_func, best_acc1, save_path, local_rank)
    func.dist.barrier()
    func.DistributedOps.cleanup()


def general_train_pipeline(args, train_func=TrainProcessCollections):
    train_func = train_func()
    local_rank = 0
    if args.seed is not None:
        train_func.set_random_seed(args.seed)
    if hasattr(args, 'MnnActivationConfig'):
        general_prepare.config_mnn_activation(args.MnnActivationConfig)
    best_acc1 = - np.inf # To support regression task.
    if args.use_cuda:
        torch.cuda.set_device(local_rank)
        cudnn.benchmark = True

    save_path = getattr(args, 'dump_path', './checkpoint/') + args.dir + '/'
    args.log_path = save_path + args.save_name + '_log.txt'
    if local_rank == 0:
        func.RecordMethods.make_dir(save_path)
        func.RecordMethods.record_hyper_parameter(save_path, args.save_name, **args.__dict__)
    

    model = train_func.make_model(args)
    if args.resume:
        args, model, best_acc1 = train_func.resume_model(args, model, local_rank)
    if args.use_cuda:
        model.cuda(local_rank)
    # prepare dataloader
    train_loader, val_loader = train_func.prepare_dataloader(args)

    # criterion, optimizer and lr_scheduler
    criterion = train_func.prepare_criterion(args)
    if args.use_cuda:
        criterion = criterion.cuda(local_rank)
    params_group = train_func.specify_params_group(model)
    optimizer, lr_scheduler = train_func.prepare_optimizer_scheduler(params_group, args)

    if args.resume:
        optimizer, lr_scheduler = train_func.resume_optimizer_scheduler(args, optimizer, lr_scheduler, local_rank)

    if lr_scheduler is not None and args.start_epoch > 0:
        lr_scheduler.step(args.start_epoch)

    if args.evaluate:
        validate_train = getattr(args, 'validate_train', False)
        if validate_train:
            train_func.validate(train_loader, model, criterion, args)
        else:
            train_func.validate(val_loader, model, criterion, args)
        return
    # run training process
    train_func.run_training(args, model, train_loader, val_loader, criterion, optimizer, lr_scheduler,
                 train_func, best_acc1, save_path, local_rank)
