import os
import sys

from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from data.dataloader_fish import FISHDataset, seq_collate
from torch.utils.data import DataLoader
import numpy as np
import random
from model.GroupNet_nba import GroupNet
from Simulator import *
import torch.optim as optim
import torch
import torch.nn as nn
from config_classifier import parse_args
from main_classifier import *


args = parse_args()
dataset_path = f"trajectories_{args.method}_{args.length}.pt"

data = torch.load(dataset_path)
real, fake = data["real"], data["fake"]

print("real", real[:,:50,:])
print("fake", fake[:,:50,:])

dataset = TrajectoryDataset(real, fake)
train_set, val_set, test_set = split_dataset(dataset)

for i, x in enumerate(train_set):
    print(x)
    if i == 20:
        break