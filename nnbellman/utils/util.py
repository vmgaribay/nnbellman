import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    mps_avail = torch.backends.mps.is_available() 
    n_gpu = torch.cuda.device_count()
    if mps_avail==True:
        if n_gpu_use > 1:
            print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only 1 is "
              "available on this machine.")
            n_gpu_use = 1
        device = torch.device("mps")
        print(f"Set to utilize 1 mps device.")
    else:
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine,"
                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
                "available on this machine.")
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        print(f"Currently configured to use {f'{n_gpu_use} GPU(s).' if n_gpu_use > 0 else 'CPU.'}")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average','result'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1, aggregation="average"):
        '''Options for aggregation are "average" and "total".'''
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        
        self._data.total.at[key] += value * n
        self._data.counts.at[key] += n
        self._data.average.at[key] = self._data.total.at[key] / self._data.counts.at[key]
        
        if aggregation == "average":
            self._data.result.at[key] = self._data.average.at[key]
        elif aggregation == "total":
            self._data.result.at[key] = self._data.total.at[key]
        else:
            print(f'Error: Invalid method {aggregation} specified as {key} aggregation method; valid options are "average" and "total".')

    def avg(self, key):
        return self._data.average.at[key]

    def result(self):
        return dict(self._data.result)
