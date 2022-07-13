import os, sys, json, numpy as np, pandas as pd, pickle, itertools, logging
from typing import TypeVar, Union, List, Literal, Tuple

# PYTORCH
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn


class FDPNN(nn.Module):
    def __init__(self, data_shape, num_classes):
        super(FDPNN, self).__init__()
        
        self.flatten = nn.Flatten()
        self.linears = nn.Sequential(
            nn.Linear(np.multiply(*data_shape), 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),            
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.linears(x)
        logits = self.softmax(x)
        return logits


class FDPCNN(nn.Module):
    def __init__(self, data_shape, num_classes):
        super(FDPCNN, self).__init__()
        
        self.linears = nn.Sequential(
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,              
                out_channels=32,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            # NOTE: 192 is calculate from strides, padding and kernel size
            nn.Linear(192, num_classes),           
        )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.linears(x)
        logits = self.softmax(x)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer, add_channel_dim=False):
    size = len(dataloader.dataset)
    use_one_hot = dataloader.dataset.dataset.use_one_hot
    
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        if add_channel_dim:
            X = X.reshape(X.shape[0], 1, *list(X.shape)[1:])

        pred = model(X)
        y = y.float() if use_one_hot else y
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, add_channel_dim=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    use_one_hot = dataloader.dataset.dataset.use_one_hot
    
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            if add_channel_dim:
                X = X.reshape(X.shape[0], 1, *list(X.shape)[1:])
            pred = model(X)
            y = y.float() if use_one_hot else y
            test_loss += loss_fn(pred, y).item()

            y = y.argmax(1) if use_one_hot else y            
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")