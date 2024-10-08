import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import torch
from torch import Tensor
from typing import Tuple, Optional

def rename_duplicate_file(file_path, file_name, suffix_pos=None):
    temp = file_name
    i = 1
    while os.path.exists(file_path + temp):
        if suffix_pos is None:
            name, suffix = file_name.split('.')
        else:
            name = file_name[:suffix_pos-1]
            suffix = file_name[suffix_pos:]
        name += '({})'.format(i)
        i += 1
        temp = name + '.{}'.format(suffix)
    return temp


def get_plt_pixel_size():
    px = 1/plt.rcParams['figure.dpi']
    return px

def get_plt_cm_size():
    return 1 / 2.54

def save_fig(fig, fig_path, fig_name, dpi=300,  bbox_inches='tight', over_writer=True, format='pdf',**kwargs):
    if not over_writer:
        fig_name = rename_duplicate_file(fig_path, fig_name)
    fig.savefig(fig_path + fig_name, dpi=dpi, bbox_inches=bbox_inches, format=format, **kwargs)

def read_log(log_path, hint='Validation result', offset=-1, transform_func=float):
    temp = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for text in f.readlines():
            if hint in text:
                info = text.split(' ')[offset]
                if transform_func is not None:
                    info = transform_func(info)
                temp.append(info)
    return temp

def prepare_fig_axs(nrows: int, ncols: int, flatten=True, **kwargs):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, **kwargs)
    if flatten and nrows * ncols > 1:
        axs = axs.reshape(-1)
    return fig, axs

def normalize_mean_cov(u: Tensor, cov: Tensor,  eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
    weight = torch.sum(torch.pow(u, 2), dim=-1, keepdim=True) + eps
    cov = cov / weight.unsqueeze(-1)
    u = u / torch.sqrt(weight)
    return u, cov

def plot_ellipse(ax, u, cov, normalize=False, scale=1, n_std=1, **kwargs):
    if normalize:
        u, cov = normalize_mean_cov(u, cov, eps=1e-8)
    u = u * scale
    cov = cov * scale
    eig_value, eig_vector = torch.linalg.eigh(cov)
    th = torch.linspace(0, 2 * np.pi, 101)
    x = torch.sqrt(eig_value[0]) * torch.cos(th) * n_std
    y = torch.sqrt(eig_value[1]) * torch.sin(th) * n_std
    r = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
    r = torch.matmul(eig_vector, r.unsqueeze(-1)).squeeze(-1)
    ax.plot(r[:, 0] + u[0], r[:, 1] + u[1], **kwargs)
    return ax
