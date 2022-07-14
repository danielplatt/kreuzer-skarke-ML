import os, sys, json, numpy as np, pandas as pd, pickle, itertools, logging
from typing import TypeVar, Union, List, Literal, Tuple, Callable

# PYTORCH
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

# OUR CODE
from data.make_numpy import preprocessing_pipeline, UNZIPPED_FILE
from fundamental_domain_projections.example1 import fundamental_domain_projection, dirichlet_projection

class FundamentalDomainProjectionDataset(Dataset):
    def __init__(
        self, 
        file:str=UNZIPPED_FILE,
        extraction_key:str='h_1,1',
        num_classes:int=None,
        use_one_hot:bool=False,
        apply_random_permutation:bool=False,
        transformation:Callable=fundamental_domain_projection,
        use_cuda:bool=True,
        logger:logging.Logger=None,
    ):
        '''
        Arguments:
        ----------
            file (str): The input file to read. If compressed (i.e. `gunzip`-ed) will attempt to unzip it.

            extraction_key (str): The label to extract. If `None` will return the headers.
            
            num_classes (int): Number of classes in `y`. Defaults to `None`. If `None` will be set to 
                `max(y)+1`.
            
            use_one_hot (bool): whether or not to return `y` as a one-hot encoded vector or as its class label
            
            apply_random_permutation (bool): Whether or not to randomly permute each matrix rather than use
                than use FDP. Defaults to `False`.

            transformation (function): A transformation to apply to `X`. Defaults to 
                `fundamental_domain_projections.example1.fundamental_domain_projection`.

            use_cuda (str): Defaults to `True`. Whether or not to put tensors on cuda. 
            
            logger (logging.Logger): Optional logger.

        Returns:
        ----------
            matrices (np.ndarray): A list of the matricies from the provided `file`.

            values (np.ndarray): A list of the headers from the provided `file` or a list of just the value
                specified by `extraction_key`.         
        '''
        if logger: 
            logger.info(f'Reading raw data file {file}.')
            logger.info(
                f'NOTE: extraction key ({extraction_key}) is specified. Only this value will be returned as `y`.'
            )
            
        X, y = preprocessing_pipeline(file, extraction_key=extraction_key, apply_random_permutation=apply_random_permutation)
        
        if logger: 
            logger.info(f'Running fundamental_domain_projection')
            logger.info(f'NOTE: this is only run once! Not once per epoch.')

        if transformation is not None:        
            X = np.array(list(map(transformation, X)))
        
        self.X = X
        self.y = y
        self.num_classes = np.max(y)+1 if num_classes is None else num_classes
        self.data_shape = self.X[0].shape
        self.use_cuda = use_cuda
        self.use_one_hot = use_one_hot
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx])
        y = torch.as_tensor(self.y[idx])
        if self.use_one_hot:
            y = nn.functional.one_hot(y, num_classes=self.num_classes)
        if self.use_cuda:
            x = x.cuda()
            y = y.cuda()        
        return x, y

def generate_dataloaders(
    dataset:Dataset, 
    training_ratio:float=0.7,
    batch_size:int=64
) -> Tuple[DataLoader, DataLoader]:
    '''
    Arguments:
    ----------
        dataset (Dataset): Current PyTorch Dataset to split into training 
            and validation sets
        
        training_ratio (float): Value between `[0, 1]`. Defaults to `0.7`. The 
            percent of samples to be used for training. The rest go to validation.

        batch_size (int): Defaults to `64`. The batch size to yield from `DataLoader`s.

    Returns:
    ---------
        train_loader (DataLoader): `DataLoader` for training data.
        valid_loader (DataLoader): `DataLoader` for validation data.
    '''
    n_train = np.floor(training_ratio * len(dataset)).astype(int)
    n_valid = len(dataset) - n_train

    train_set, valid_set = torch.utils.data.random_split(dataset, [n_train, n_valid])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader

import matplotlib.pylot as plt, matplotlib.patches as mpatches
import matplotlib
def pretty_scatter_plot(
    x:Union[list[float], np.ndarray], 
    y:Union[list[float], np.ndarray], 
    c:Union[list[float], np.ndarray], 
    title:str=None, 
    xlabel:str=None, 
    ylabel:str=None,
    z:Union[list[float], np.ndarray]=None, 
    palette:str='viridis', 
    legend:bool=True
) -> Tuple[matplotlib.figure.Figure, matplotlib.axis.Axis]:
    '''
    Sensible defaults for a scatter plot

    Arguments:
    ----------
        x (list[float], np.ndarray): x axis coordinate values.

        y (list[float], np.ndarray): y axis coordinate values.

        c (list[float], np.ndarray): color values.

        title (str): Optional. Title to add to plot.

        xlabel (str): Optional. x axis label to add to plot.

        ylabel (str): Optional. y axis label to add to plot.

        z (list[float], np.ndarray): z axis coordinate values. Defaults to `None`.
            If is not `None` then plots in 3d.

        palette (str): Color palette. Defaults to `'viridis'`.

        legend (bool): Defaults to `True`. Whether or not to add a legend to plot.

    Returns:
    ---------
        fig (matplotlib.figure.Figure): Figure object.

        ax (matplotlib.axis.Axis): Axis object.
    '''
    fig = plt.figure(figsize=(14, 8), dpi=300)
    
    if z is None:
        ax = fig.add_subplot(1,1,1)
    else:
        ax = fig.add_subplot(1,1,1, projection='3d')

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if z is None:
        ax.scatter(x, y, s=120, c='black', cmap=palette)
        ax.scatter(x, y, s=100, c='white', cmap=palette)
        ax.scatter(x, y, s=80,  c=c, cmap=palette, alpha=0.7, marker='o', edgecolors=None)
    else:
        ax.scatter(x, y, z, s=120, c='black', cmap=palette)
        ax.scatter(x, y, z, s=100, c='white', cmap=palette)
        ax.scatter(x, y, z, s=80,  c=c, cmap=palette, alpha=0.7, marker='o', edgecolors=None)
    
    cmap = plt.get_cmap()
    if legend:
        fig.legend(
            handles=[
                mpatches.Patch(color=cmap(i/(len(np.unique(c))-1)), label=_type)
                for i, _type in enumerate(np.unique(c))
            ]
        )
    return fig, ax