# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import random
import shutil
import tempfile
import time
import warnings
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn
import torchvision
import yaml
from torch import Tensor
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .. import mnn_core
from .. import models
from . import dataloaders

try:
    from torch.distributed.optim import ZeroRedundancyOptimizer

    zero_redundancy_optimizer_available = True
except ImportError:
    warnings.warn("Try to import ZeroRedundancyOptimizer for distributed training but failed")
    zero_redundancy_optimizer_available = False


_SPECIAL_ARGS = ["LR_SCHEDULER", "OPTIMIZER", "DATASET", "DATALOADER"]


class RecordMethods:
    @staticmethod
    def make_dir(dir_path: str) -> None:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def save_state(file_path, net_state, trained_epochs, acc, **kwargs) -> None:
        state = {"net": net_state, "epochs": trained_epochs, "acc": acc}
        state.update(kwargs)
        torch.save(state, file_path)

    @staticmethod
    def writing_log(log_path: str, info: str, encoding: str = "utf-8", mode: str = "a+") -> None:
        with open(log_path, encoding=encoding, mode=mode) as file:
            file.write(info)

    @staticmethod
    def load_state_dict(net: torch.nn.Module, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint["net"])
        return net

    @staticmethod
    def record_hyper_parameter(path: str, name: str, **kwargs) -> None:
        with open(path + f"{name}_config.yaml", "w") as file:
            yaml.dump(kwargs, file, default_flow_style=False)

    @staticmethod
    def rename_duplicate_file(file_path, file_name, suffix_pos=None):
        candidate = file_name
        index = 1
        while os.path.exists(file_path + candidate):
            if suffix_pos is None:
                name, suffix = file_name.split(".")
            else:
                name = file_name[: suffix_pos - 1]
                suffix = file_name[suffix_pos:]
            name += f"({index})"
            index += 1
            candidate = f"{name}.{suffix}"
        return candidate


class InputPreprocess:
    def __init__(self, device="cpu", dtype=None, mask_mean: bool = False, mask_cov: bool = False) -> None:
        self.device = device
        self.dtype = dtype
        self.mask_mean = mask_mean
        self.mask_cov = mask_cov

    def __call__(self, inputs):
        return self.mnn_inputs_preprocess(inputs)

    def mnn_inputs_preprocess(self, inputs):
        if isinstance(inputs, Tensor):
            return self.to_device_and_dtype(inputs, self.device, self.dtype)
        if not isinstance(inputs[0], Tensor):
            return tuple(self._tuple_to(item) for item in inputs)
        return self._tuple_to(inputs)

    def _tuple_to(self, data):
        mean, covariance = data
        mean = self.to_device_and_dtype(mean, self.device, self.dtype, self.mask_mean)
        covariance = self.to_device_and_dtype(covariance, self.device, self.dtype, self.mask_cov)
        return mean, covariance

    @staticmethod
    def to_device_and_dtype(data: Tensor, device="cpu", dtype=None, mask: bool = False):
        if dtype == "float":
            data = data.to(torch.float32)
        elif dtype == "double":
            data = data.to(torch.float64)
        data = data.to(device)
        return torch.zeros_like(data) if mask else data


class PredictMethods:
    @staticmethod
    def max_mean_predictor(outputs):
        logits = outputs if isinstance(outputs, Tensor) else outputs[0]
        _, predicted = logits.max(1)
        return predicted

    @staticmethod
    def min_risk_predictor(mean, std, gamma):
        risk = -mean + gamma * std / 2
        _, predicted = torch.min(risk, dim=-1)
        return predicted


class BinaryPredictor:
    def __init__(self, threshold: float = 0.5) -> None:
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        self.threshold = threshold

    def __call__(self, output: Tensor):
        return torch.gt(output, self.threshold).to(torch.long).view(-1)


class ScoreMethods:
    @staticmethod
    def equal_protocol(predicted, targets):
        return predicted.eq(targets.view(predicted.size())).sum().item()

    @staticmethod
    def regression_protocol(predicted, targets, threshold: float, gamma: int = 1):
        return torch.sum(torch.abs(predicted - targets) < threshold) / gamma


def check_nan(x, raise_err: bool = True) -> None:
    if torch.sum(torch.isnan(x)) > 0:
        print("Input Has NaN!")
        if raise_err:
            raise ValueError
    else:
        print("Input has no NaN!")


def batch_numpy2tensor(device, *args):
    return [torch.from_numpy(item).to(device).to(torch.float64) for item in args]


def batch_cat_tensor(dim=0, *args):
    return [torch.cat(item, dim=dim) for item in args]


class DistributedOps:
    @staticmethod
    def setup(rank, world_size, backend="nccl") -> None:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(backend, rank=rank, world_size=world_size)

    @staticmethod
    def cleanup() -> None:
        dist.destroy_process_group()

    @staticmethod
    def checkpoint(model, save_path, save_name, rank) -> None:
        if rank == 0:
            torch.save(model.state_dict(), save_path + save_name + ".pt")
        dist.barrier()

    @staticmethod
    def prepare_dataloader(dataset, rank, world_size, batch_size=256, pin_memory=False, num_workers=0, shuffle=True, drop_last=False):
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last)
        return DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=drop_last, shuffle=False, sampler=sampler)

    @staticmethod
    def wrap_model(model, rank):
        return DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    @staticmethod
    def ddp_runs(func, world_size) -> None:
        mp.spawn(func, args=(world_size,), nprocs=world_size)

    @staticmethod
    def reduce_mean(tensor, nprocs):
        reduced = tensor.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        return reduced / nprocs


def remove_file(path) -> None:
    if os.path.exists(path):
        os.remove(path)
    else:
        warnings.warn(f"File not exist! {path}")


def print_peak_memory(prefix, device) -> None:
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


class PrepareMethods:
    @staticmethod
    def prepare_optimizer(net, optimizer, *args, **kwargs):
        return optimizer(filter(lambda parameter: parameter.requires_grad, net.parameters()), *args, **kwargs)

    @staticmethod
    def seed_everything(seed) -> None:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True

    @staticmethod
    def device_prepare(is_cpu: bool = False):
        if is_cpu:
            return "cpu"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            cudnn.benchmark = True
        return device


def make_model(model_args: dict):
    meta_info: dict = model_args["meta"]
    if meta_info.get("arch", "") == "mnn_mlp":
        return make_mnn_mlp_model(model_args[meta_info["mlp_type"]], meta_info["mlp_type"])
    return None


def make_ann_model(model_args):
    meta_info = model_args["meta"]
    cnn_args = model_args.get("backbone", None)
    model = getattr(torchvision.models, meta_info["cnn_type"])(pretrained=cnn_args["pretrained"])
    if cnn_args["frozen"]:
        freeze_layers(model, cnn_args.get("freeze_layer", None))
    return model


def make_mnn_mlp_model(model_args, model_type="mnn_mlp"):
    if model_type == "mnn_mlp":
        return models.MnnMlp(**model_args)
    if model_type == "snn_mlp":
        return models.SnnMlp(**model_args)
    return models.MnnMlpNoRho(**model_args)


def freeze_parameters(modules):
    for parameter in modules.parameters():
        parameter.requires_grad = False
    return modules


def freeze_layers(modules, layer_depth=None):
    if layer_depth is None:
        return freeze_parameters(modules)
    for index in range(layer_depth):
        modules[index] = freeze_parameters(modules[index])
    return modules


def make_transforms_compose(params):
    transform_list = []
    for key in params["aug_order"]:
        if key == "ToTensor":
            transform_list.append(torchvision.transforms.ToTensor())
        else:
            transform_list.append(getattr(torchvision.transforms, key)(**params[key]))
    return torchvision.transforms.Compose(transform_list)


def make_dataloader(dataset, args, train: bool = True):
    if args.distributed:
        sampler = DistributedSampler(dataset) if train else DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None
    return DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=train and sampler is None,
        sampler=sampler,
        pin_memory=args.pin_mem,
        num_workers=args.workers,
    )


def make_image_fold_dataset(args):
    transform_train = make_transforms_compose(args.DATAAUG_TRAIN)
    transform_val = make_transforms_compose(args.DATAAUG_VAL)
    train_set = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, args.train_split), transform=transform_train)
    val_set = torchvision.datasets.ImageFolder(os.path.join(args.data_dir, args.val_split), transform=transform_val)
    return train_set, val_set


def make_torchvision_dataset(args, data_dir="./data/"):
    transform_train = make_transforms_compose(args.DATAAUG_TRAIN)
    transform_val = make_transforms_compose(args.DATAAUG_VAL)
    train_set = getattr(torchvision.datasets, args.dataset)(root=data_dir, train=True, download=True, transform=transform_train)
    test_set = getattr(torchvision.datasets, args.dataset)(root=data_dir, train=False, download=True, transform=transform_val)
    return train_set, test_set


def prepare_dataloader(args, data_dir="./data/"):
    if args.dataset == "mnist":
        transform_train = make_transforms_compose(args.DATAAUG_TRAIN)
        transform_val = make_transforms_compose(args.DATAAUG_VAL)
        return dataloaders.classic_mnist_loader(data_dir, args.bs, args.bs, transform_train=transform_train, transform_test=transform_val)
    train_set, test_set = make_torchvision_dataset(args, data_dir)
    return make_dataloader(train_set, args, True), make_dataloader(test_set, args, False)


def make_optimizer(params_group, args):
    if args.distributed and zero_redundancy_optimizer_available:
        return ZeroRedundancyOptimizer(params_group, getattr(torch.optim, args.OPTIMIZER["name"]), **args.OPTIMIZER["args"])
    return getattr(torch.optim, args.OPTIMIZER["name"])(params_group, **args.OPTIMIZER["args"])


def make_schedule(optimizer, args):
    return getattr(torch.optim.lr_scheduler, args.LR_SCHEDULER["name"])(optimizer=optimizer, **args.LR_SCHEDULER["args"])


def prepare_optimizer_scheduler(params_group, args):
    optimizer = torch.optim.AdamW(params_group, lr=args.lr) if args.OPTIMIZER is None else make_optimizer(params_group, args)
    scheduler = None if args.LR_SCHEDULER is None else make_schedule(optimizer, args)
    return optimizer, scheduler


def make_criterion(criterion_args: dict):
    criterion_args_value = criterion_args.get("args", None)
    if criterion_args["source"] == "pytorch":
        criterion = getattr(torch.nn, criterion_args["name"])
    else:
        criterion = getattr(mnn_core.nn, criterion_args["name"])
    return criterion() if criterion_args_value is None else criterion(**criterion_args_value)


def prepare_criterion(args):
    try:
        return make_criterion(args.CRITERION)
    except AttributeError:
        return mnn_core.nn.CrossEntropyOnMean()


def prepare_data_augmentation(args):
    return make_transforms_compose(args.DATAAUG_TRAIN), make_transforms_compose(args.DATAAUG_VAL)


def read_yaml_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


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
    args.save_path = getattr(args, "dump_path", "./checkpoint/") + args.dir + "/"
    if args.config is not None:
        args = set_config2args(args)
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.world_size = torch.cuda.device_count()
    args.distributed = args.world_size > 1 and args.multiprocessing_distributed
    args.local_rank = 0
    args.use_cuda = torch.cuda.is_available() and not args.cpu
    return args


class TempArgs:
    pass


def model_generator(checkpoint_path, save_name, to_cuda=False, resume_model=True, local_rank=0, resume_best=False, make_func=None):
    if make_func is None:
        config = read_yaml_config(f"{checkpoint_path}{save_name}_config.yaml")
        model = make_model(config["MODEL"])
    else:
        args = TempArgs()
        args.config = f"{checkpoint_path}{save_name}_config.yaml"
        set_config2args(args)
        model = make_func.make_model(args)
    if resume_model:
        suffix = "_best_model.pth" if resume_best else ".pth"
        checkpoint = torch.load(f"{checkpoint_path}{save_name}{suffix}", map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    if to_cuda:
        model.cuda(local_rank)
    model.eval()
    return model


def config_mnn_activation(activation_config: dict) -> None:
    for key, value in activation_config.items():
        mnn_core.set_core_attr(key, value)


def deploy_config(is_parse: bool = True):
    parser = argparse.ArgumentParser(description="Pytorch Mnn Training Template")
    parser.add_argument("-c", "--config", default=None, type=str, metavar="FILE", help="YAML config file")
    parser.add_argument("--bs", default=50, type=int, help="batch size")
    parser.add_argument("--resume", action="store_true", default=False, help="resume from checkpoint")
    parser.add_argument("-p", "--print-freq", default=20, type=int, metavar="N", help="print frequency (default: 10)")
    parser.add_argument("--dir", default="mnist", type=str, help="dir path that used to save checkpoint")
    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU only or not")
    parser.add_argument("--trials", default=1, type=int, help="number of trials to run")
    parser.add_argument("--dataset_type", default="poisson", type=str, help="methods used in data loaders")
    parser.add_argument("--dataset", default="mnist", type=str, help="type of dataset")
    parser.add_argument("--data_dir", default="./data/", type=str, help="type of dataset")
    parser.add_argument("--eps", default=0.1, type=float, help="eps")
    parser.add_argument("--gpu", default=None, type=str, help="specify gpu idx if use gpu")
    parser.add_argument("--which_run", default="main", type=str, help="specify the model to run")
    parser.add_argument("--save_name", default="mnn_net", type=str, help="alias to save net")
    parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
    parser.add_argument("--workers", default=1, type=int, help="num workers for dataloader. ")
    parser.add_argument("--multiprocessing_distributed", action="store_true", default=False, help="Use multi-processing distributed training")
    parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--world_size", default=-1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.05)")
    parser.add_argument("--epochs", type=int, default=25, metavar="N", help="number of epochs to train (default: 25)")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    return prepare_args(parser) if is_parse else parser


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE) -> None:
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self) -> str:
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError(f"invalid summary type {self.summary_type!r}")
        return fmtstr.format(**self.__dict__)


def _get_batch_fmtstr(num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = "{:" + str(num_digits) + "d}"
    return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix="") -> None:
        self.batch_fmtstr = _get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        info = "\t".join(entries)
        print(info)
        return info

    def display_summary(self):
        entries = [self.prefix + " *"]
        entries += [meter.summary() for meter in self.meters]
        info = " ".join(entries)
        print(info)
        return info


def to_cuda(data, local_rank):
    if isinstance(data, torch.Tensor):
        return data.cuda(local_rank, non_blocking=True)
    return [item.cuda(local_rank, non_blocking=True) for item in data]


class TrainProcessCollections:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def set_random_seed(self, seed) -> None:
        PrepareMethods.seed_everything(seed)

    def make_model(self, args):
        return make_model(model_args=args.MODEL)

    def prepare_dataloader(self, args):
        return prepare_dataloader(args, getattr(args, "data_dir", "./data/"))

    def prepare_optimizer_scheduler(self, params_group, args):
        return prepare_optimizer_scheduler(params_group, args)

    def prepare_criterion(self, args):
        return prepare_criterion(args)

    def params_clip(self, model, MIN=-1, MAX=1):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.clamp_(MIN, MAX)
        return model

    def params_frozen(self, model):
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model

    def specify_params_group(self, model):
        return filter(lambda parameter: parameter.requires_grad, model.parameters())

    def input_preprocessing(self, data, args):
        if isinstance(data, tuple):
            data, covariance = data
        else:
            if getattr(args, "flatten_input", True):
                data = torch.flatten(data, start_dim=1)
            data = data * getattr(args, "scale_factor", 1.0)
            if getattr(args, "input_prepare", None) == "flatten_poisson":
                covariance = torch.diag_embed(torch.abs(data))
            elif getattr(args, "input_prepare", None) == "poisson_no_rho":
                covariance = torch.abs(data)
            else:
                covariance = None
        if getattr(args, "background_noise", None) is not None and covariance is not None:
            if data.size() == covariance.size():
                covariance = covariance + torch.ones_like(covariance, device=data.device) * getattr(args, "background_noise")
            else:
                covariance = covariance + torch.eye(data.size(-1), device=data.device) * getattr(args, "background_noise")
        return data, covariance

    def data2device(self, data, target, args):
        if args.use_cuda:
            if isinstance(data, tuple):
                data, covariance = data
                data = data.cuda(args.local_rank, non_blocking=True)
                covariance = covariance.cuda(args.local_rank, non_blocking=True)
                data = (data, covariance)
            else:
                data = data.cuda(args.local_rank, non_blocking=True)
            if isinstance(target, torch.Tensor):
                target = target.cuda(args.local_rank, non_blocking=True)
        data, covariance = self.input_preprocessing(data, args)
        return ((data, covariance), target) if covariance is not None else (data, target)

    def clip_model_params(self, model, args) -> None:
        clip_args: Optional[dict] = getattr(args, "CLIP_PARAMS", None)
        if clip_args is None:
            return
        for key in clip_args.keys():
            sub_model = model
            for module_key in key.split("."):
                sub_model = getattr(sub_model, module_key)
            self.params_clip(sub_model, MIN=clip_args[key]["min"], MAX=clip_args[key]["max"])

    def metric_init(self, data_loader, epoch, prefix="Epoch: [{}]"):
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.2f")
        progress = ProgressMeter(len(data_loader), [batch_time, data_time, losses, top1], prefix=prefix.format(epoch))
        return batch_time, data_time, losses, top1, progress

    def reduce_distributed_info(self, args, *metrics):
        return [DistributedOps.reduce_mean(metric, args.nprocs) for metric in metrics]

    def compute_model_output(self, model, inputs, args=None):
        return model(inputs)

    def compute_loss(self, output, target, criterion, model=None, args=None, inputs=None):
        if hasattr(model, "criterion_params"):
            return criterion(output, target, model.criterion_params)
        return criterion(output, target)

    def train_one_epoch(self, train_loader, model, criterion, optimizer, epoch, args):
        batch_time, data_time, losses, top1, progress = self.metric_init(train_loader, epoch)
        model.train()
        end = time.time()
        for index, (images, target) in enumerate(train_loader):
            self.clip_model_params(model, args)
            images, target = self.data2device(images, target, args)
            data_time.update(time.time() - end)
            output = self.compute_model_output(model, images, args)
            loss = self.compute_loss(output=output, target=target, criterion=criterion, model=model, args=args, inputs=images)
            if isinstance(loss, tuple):
                loss, pred = loss
            else:
                pred = None
            optimizer.zero_grad()
            loss.backward()
            if getattr(args, "max_grad_norm", None) is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            if getattr(args, "max_grad_value", None) is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=args.max_grad_value)
            optimizer.step()
            if torch.cuda.is_available() and getattr(args, "use_cuda", False):
                torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            inputs_for_size = images[0] if isinstance(images, tuple) else images
            if getattr(args, "task_type", "classification") == "classification":
                (acc1,) = self.score_function(output, target, topk=(1,), pred_prob=pred)
                if args.distributed:
                    loss, acc1 = self.reduce_distributed_info(args, loss, acc1)
                top1.update(acc1.item(), inputs_for_size.size(0))
            losses.update(loss.item(), inputs_for_size.size(0))
            end = time.time()
            if index % args.print_freq == 0 and args.local_rank == 0:
                info = progress.display(index) + "\n"
                if getattr(args, "log_path", None) is not None:
                    RecordMethods.writing_log(args.log_path, info)
        if args.local_rank == 0 and getattr(args, "log_path", None) is not None:
            RecordMethods.writing_log(args.log_path, "Training result: " + progress.display_summary() + "\n")

    def validate(self, val_loader, model, criterion, args, epoch=0):
        batch_time, data_time, losses, top1, progress = self.metric_init(val_loader, epoch, "Test [Epoch:{}]: ")
        model.eval()
        with torch.no_grad():
            end = time.time()
            for images, target in val_loader:
                images, target = self.data2device(images, target, args)
                output = self.compute_model_output(model, images, args)
                loss = self.compute_loss(output=output, target=target, criterion=criterion, model=model, args=args, inputs=images)
                if isinstance(loss, tuple):
                    loss, pred = loss
                else:
                    pred = None
                inputs_for_size = images[0] if isinstance(images, tuple) else images
                if getattr(args, "task_type", "classification") == "classification":
                    (acc1,) = self.score_function(output, target, topk=(1,), pred_prob=pred)
                    if args.distributed:
                        loss, acc1 = self.reduce_distributed_info(args, loss, acc1)
                    top1.update(acc1.item(), inputs_for_size.size(0))
                losses.update(loss.item(), inputs_for_size.size(0))
                batch_time.update(time.time() - end)
                end = time.time()
            if args.local_rank == 0 and getattr(args, "log_path", None) is not None:
                RecordMethods.writing_log(args.log_path, "Validation result: " + progress.display_summary() + "\n")
        return top1.avg if getattr(args, "task_type", "classification") == "classification" else -losses.avg

    def score_function(self, output, target, *args, **kwargs):
        return self.accuracy(output, target, *args, **kwargs)

    def accuracy(self, output, target, topk=(1,), pred_prob=None):
        with torch.no_grad():
            logits = output if pred_prob is not None else (output if isinstance(output, torch.Tensor) else output[0])
            if pred_prob is not None:
                logits = pred_prob
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = logits.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            return [correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(100.0 / batch_size) for k in topk]

    def save_checkpoint(self, state, is_best, save_path, save_name="checkpoint") -> None:
        torch.save(state, save_path + save_name + ".pth")
        if is_best:
            shutil.copyfile(save_path + save_name + ".pth", save_path + save_name + "_best_model.pth")

    def resume_model(self, args, model, local_rank=0):
        loc = f"cuda:{local_rank}" if args.use_cuda else "cpu"
        save_path = getattr(args, "dump_path", "./checkpoint/") + args.dir + "/"
        suffix = "_best_model.pth" if getattr(args, "resume_best", False) else ".pth"
        checkpoint = torch.load(save_path + args.save_name + suffix, map_location=loc)
        if getattr(args, "continue_train", True):
            args.start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint.get("best_acc1", None)
        model.load_state_dict(checkpoint["state_dict"])
        return args, model, best_acc1

    def resume_optimizer_scheduler(self, args, optimizer, lr_scheduler=None, local_rank=0):
        loc = f"cuda:{local_rank}" if args.use_cuda else "cpu"
        save_path = getattr(args, "dump_path", "./checkpoint/") + args.dir + "/"
        suffix = "_best_model.pth" if getattr(args, "resume_best", False) else ".pth"
        checkpoint = torch.load(save_path + args.save_name + suffix, map_location=loc)
        optimizer_state = checkpoint.get("optimizer_state", None)
        scheduler_state = checkpoint.get("scheduler_state", None) if lr_scheduler is not None else None
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if lr_scheduler is not None and scheduler_state is not None:
            lr_scheduler.load_state_dict(scheduler_state)
        return optimizer, lr_scheduler

    def run_training(self, args, model, train_loader, val_loader, criterion, optimizer, lr_scheduler, train_func, best_acc1, save_path, local_rank=0):
        best_epoch = args.start_epoch
        for epoch in range(args.start_epoch, args.epochs):
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            train_func.train_one_epoch(train_loader, model, criterion, optimizer, epoch, args)
            acc1 = train_func.validate(val_loader, model, criterion, args, epoch=epoch)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            if is_best:
                best_epoch = epoch
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)
            if zero_redundancy_optimizer_available and isinstance(optimizer, ZeroRedundancyOptimizer):
                optimizer.consolidate_state_dict()
            if local_rank == 0:
                state_dict = model.module.state_dict() if args.distributed else model.state_dict()
                save_state = {
                    "epoch": epoch + 1,
                    "arch": args.save_name,
                    "best_acc1": best_acc1,
                    "best_epoch": best_epoch,
                    "state_dict": state_dict,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                }
                train_func.save_checkpoint(save_state, is_best, save_path, save_name=args.save_name)
        if local_rank == 0:
            info = f"-*- Summary: after {args.epochs - args.start_epoch} epochs training, the model hit {best_acc1}% top1 acc at epoch [{best_epoch}]\n"
            RecordMethods.writing_log(args.log_path, info)


def general_distributed_train_pipeline(local_rank, nprocs, args, train_func=TrainProcessCollections):
    train_func = train_func()
    args.local_rank = local_rank
    if args.seed is not None:
        train_func.set_random_seed(args.seed)
    if hasattr(args, "MnnActivationConfig") and local_rank == 0:
        config_mnn_activation(args.MnnActivationConfig)
    best_acc1 = -np.inf
    DistributedOps.setup(local_rank, nprocs)
    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True
    save_path = getattr(args, "dump_path", "./checkpoint/") + args.dir + "/"
    args.log_path = save_path + args.save_name + "_log.txt"
    if local_rank == 0:
        RecordMethods.make_dir(save_path)
        RecordMethods.record_hyper_parameter(save_path, args.save_name, **args.__dict__)
    model = train_func.make_model(args)
    loc = f"cuda:{local_rank}"
    if args.resume:
        args, model, best_acc1 = train_func.resume_model(args, model, local_rank)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weight.pt")
        if local_rank == 0:
            torch.save(model.state_dict(), checkpoint_path)
        dist.barrier()
        model.load_state_dict(torch.load(checkpoint_path, map_location=loc))
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    train_loader, val_loader = train_func.prepare_dataloader(args)
    criterion = train_func.prepare_criterion(args).cuda(local_rank)
    optimizer, lr_scheduler = train_func.prepare_optimizer_scheduler(train_func.specify_params_group(model), args)
    if args.resume:
        optimizer, lr_scheduler = train_func.resume_optimizer_scheduler(args, optimizer, lr_scheduler, local_rank)
    if lr_scheduler is not None and args.start_epoch > 0:
        lr_scheduler.step(args.start_epoch)
    if args.evaluate:
        train_func.validate(val_loader, model, criterion, args)
        return
    train_func.run_training(args, model, train_loader, val_loader, criterion, optimizer, lr_scheduler, train_func, best_acc1, save_path, local_rank)
    dist.barrier()
    DistributedOps.cleanup()


def general_train_pipeline(args, train_func=TrainProcessCollections, **kwargs):
    train_func = train_func(args, **kwargs)
    local_rank = 0
    if args.seed is not None:
        train_func.set_random_seed(args.seed)
    if hasattr(args, "MnnActivationConfig"):
        config_mnn_activation(args.MnnActivationConfig)
    best_acc1 = -np.inf
    if args.use_cuda:
        torch.cuda.set_device(local_rank)
        cudnn.benchmark = True
    save_path = getattr(args, "dump_path", "./checkpoint/") + args.dir + "/"
    args.log_path = save_path + args.save_name + "_log.txt"
    if local_rank == 0:
        RecordMethods.make_dir(save_path)
        RecordMethods.record_hyper_parameter(save_path, args.save_name, **args.__dict__)
    model = train_func.make_model(args)
    if args.resume:
        args, model, best_acc1 = train_func.resume_model(args, model, local_rank)
    if args.use_cuda:
        model.cuda(local_rank)
    train_loader, val_loader = train_func.prepare_dataloader(args)
    criterion = train_func.prepare_criterion(args)
    if args.use_cuda:
        criterion = criterion.cuda(local_rank)
    optimizer, lr_scheduler = train_func.prepare_optimizer_scheduler(train_func.specify_params_group(model), args)
    if args.resume:
        optimizer, lr_scheduler = train_func.resume_optimizer_scheduler(args, optimizer, lr_scheduler, local_rank)
    if lr_scheduler is not None and args.start_epoch > 0:
        lr_scheduler.step(args.start_epoch)
    if args.evaluate:
        train_func.validate(train_loader if getattr(args, "validate_train", False) else val_loader, model, criterion, args)
        return
    train_func.run_training(args, model, train_loader, val_loader, criterion, optimizer, lr_scheduler, train_func, best_acc1, save_path, local_rank)


__all__ = [
    "RecordMethods",
    "InputPreprocess",
    "PredictMethods",
    "BinaryPredictor",
    "ScoreMethods",
    "check_nan",
    "batch_numpy2tensor",
    "batch_cat_tensor",
    "DistributedOps",
    "remove_file",
    "print_peak_memory",
    "PrepareMethods",
    "make_model",
    "make_ann_model",
    "make_mnn_mlp_model",
    "freeze_parameters",
    "freeze_layers",
    "make_transforms_compose",
    "make_dataloader",
    "make_image_fold_dataset",
    "make_torchvision_dataset",
    "prepare_dataloader",
    "make_optimizer",
    "make_schedule",
    "prepare_optimizer_scheduler",
    "make_criterion",
    "prepare_criterion",
    "prepare_data_augmentation",
    "read_yaml_config",
    "set_config2args",
    "reset_special_args",
    "prepare_args",
    "TempArgs",
    "model_generator",
    "config_mnn_activation",
    "deploy_config",
    "Summary",
    "AverageMeter",
    "ProgressMeter",
    "to_cuda",
    "TrainProcessCollections",
    "general_distributed_train_pipeline",
    "general_train_pipeline",
]
