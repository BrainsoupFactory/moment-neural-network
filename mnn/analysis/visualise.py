import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


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