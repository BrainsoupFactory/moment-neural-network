import torch
import numpy as np
from torch.utils.data import Subset, Dataset, DataLoader
from collections import defaultdict
from typing import Tuple
from torch import Tensor


def find_converge_step(pred: Tensor, dt=1, start_time=0):
    start_time = int(start_time / dt)
    x = torch.flip(torch.cumsum(torch.flip(pred[start_time:] == pred[-1], dims=[0]), dim=0), dims=[0])
    x = torch.movedim(x, 0, -1) # (time, trials, batch) -> (trials, batch, time)
    x = x / torch.arange(x.shape[0], 0, -1, device=x.device)
    x = torch.argmax((x == 1).float(), dim=-1) + start_time
    return x

def get_converge_step(prediction, dt=1, start_time=0):
    converge_step = {}
    for key, pred in prediction.items():
        converge_step[key] = find_converge_step(pred, dt=dt, start_time=start_time)
    return converge_step


def seperate_right_wrong(outputs, labels, correct_idx=None, wrong_idx=None):
    if correct_idx is None:
        correct_idx = torch.ones_like(labels, dtype=torch.bool)
    if wrong_idx is None:
        wrong_idx = torch.ones_like(labels, dtype=torch.bool)
    if isinstance(outputs, tuple):
        mu = outputs[0]
    correct_idx = correct_idx * (mu.argmax(dim=-1) == labels)
    wrong_idx = wrong_idx * (mu.argmax(dim=-1) != labels)
    return correct_idx, wrong_idx


@torch.no_grad()
def collect_mnn_outputs(model, loader, train_funcs, args):
    model.eval()
    output_mean = []
    output_cov = []
    labels = []
    for data, label in loader:
        data, label = train_funcs.data2device(data, label, args)
        output = model(data)
        output_mean.append(output[0].cpu())
        output_cov.append(output[1].cpu())
        labels.append(label.cpu())
    output_mean = torch.cat(output_mean, dim=0)
    output_cov = torch.cat(output_cov, dim=0)
    outputs = (output_mean, output_cov)
    labels = torch.cat(labels, dim=0)
    return outputs, labels


def compute_entropy(cov: Tensor, eps=1e-6) -> Tensor:
    dims = cov.size(-1)
    entropy = torch.linalg.det(cov)
    entropy = torch.where(entropy > 0, entropy, torch.ones_like(entropy) * eps)
    entropy = 0.5 * (torch.log(2*np.pi * entropy) + dims)
    return entropy


def chunk_sparse_spike_count(raw_spike_train: Tensor, dt=1e-2, time_window=1, pre_avg=False) -> Tensor:
    """
    The function return a sequence of spike count of each non-overlap time window
    """
    end_time = raw_spike_train.size(0) 
    num_trial = raw_spike_train.size(1)
    num_neuron = raw_spike_train.size(2)

    full_indices = raw_spike_train.coalesce().indices()
    time_idx = full_indices[0] 
    trial_idx = full_indices[1]
    neuron_idx = full_indices[2]
    
    start_time = 1 # shift time idx to align real simulation
    steps = int(time_window / dt)
    obs = []
    while start_time < end_time:
        indices = torch.logical_and(time_idx >= start_time, time_idx < start_time + steps)
        i = torch.cat([time_idx[indices].unsqueeze(0) - start_time, trial_idx[indices].unsqueeze(0),
                    neuron_idx[indices].unsqueeze(0)])
        v = torch.ones(i.size()[-1], device=i.device)
        x = torch.sparse_coo_tensor(i, v, (steps, num_trial, num_neuron))
        try:
            x = torch.sparse.sum(x, dim=0).to_dense().unsqueeze(0)
        except RuntimeError:
            x = torch.zeros(1, num_trial, num_neuron, device=x.device)
        if pre_avg:
            x = torch.mean(x, dim=1) / time_window
        obs.append(x)
        start_time += steps
    obs = torch.cat(obs)
    return obs


def find_sub_dataset(dataset: Dataset, prefer_class: Tuple, num_samples: int) -> Subset:
    idx = []
    found_class = defaultdict(lambda: 0)
    total_cases = len(prefer_class) * num_samples
    for i in range(len(dataset)):
        t = dataset.targets[i]
        if t in prefer_class and i not in idx and found_class[t] < num_samples:
            idx.append(i)
            found_class[t] += 1
        if len(idx) >= total_cases:
            break
    return Subset(dataset=dataset, indices=idx)


def z_score_normalize(x: Tensor) -> Tensor:
    var, mean = torch.var_mean(x)
    output = (x - mean) / torch.sqrt(var)
    return output

def min_max_norm(x: Tensor, w=1., b=0., eps=1e-5) -> Tensor:
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x) + eps) * w + b

class TempArgs:
    def __init__(self) -> None:
        pass
    
def efficient_dimension(matrix: Tensor) -> Tensor:
    try:
        eigenvalues, _ = torch.linalg.eigh(matrix)
    except:
        eigenvalues, _ = torch.linalg.eig(matrix)
        eigenvalues = eigenvalues.real

    ed = torch.sum(eigenvalues, dim=-1, keepdim=True) ** 2 / torch.sum(eigenvalues ** 2, dim=-1, keepdim=True)
    ed = torch.nan_to_num(ed)
    return ed.squeeze()


