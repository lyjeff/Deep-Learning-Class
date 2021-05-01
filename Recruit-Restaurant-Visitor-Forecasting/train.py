import pandas as pd
import numpy as np
from tqdm import tqdm
# from torchsummary import summary

import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, random_split)

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from models import MyModel
from dataset import Visitor_Dataset

# learning rate, epoch and batch size. Can change the parameters here.
def train(dataset_path, lr, epoch, batch_size):
    train_loss_curve = []
    valid_loss_curve = []
    best = 100

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = MyModel()
    # model = model.to(device)
    # model.train()

    # dataset and dataloader
    train_dataset = Visitor_Dataset(dataset_path)

if __name__ == '__main__':
    dataset_path = "./dataset/"
    lr = 0.001
    epoch = 200
    batch_size = 64
    train(dataset_path, lr, epoch, batch_size)