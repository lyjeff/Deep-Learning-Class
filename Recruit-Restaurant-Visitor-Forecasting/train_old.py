import math
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import (DataLoader, random_split)
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from models import MyModel
from dataset import MLDataset
# from torchsummary import summary

# WRMSE
def WRMSE(preds, labels, device):
    weight = torch.tensor([
        0.05223, 0.0506, 0.05231, 0.05063, 0.05073,
        0.05227, 0.05177, 0.05186, 0.05076, 0.05063,
        0.0173, 0.05233, 0.05227, 0.05257, 0.05259,
        0.05222, 0.05204, 0.05185, 0.05229, 0.05074
    ]).to(device)
    wrmse = torch.pow(preds-labels, 2)
    wrmse = torch.sum(wrmse * weight)
    return wrmse.item()

# training curve
def visualize(train, valid, title, filename, best, wrmse=False):
    # print(best)
    # print(best[0], best[1])
    plt.title(title)
    plt.xlabel("Epochs")
    plt.plot(train, label=f"Train {title}")
    plt.plot(valid, label=f"Valid {title}")
    if wrmse == True:
        # plt.title(f'Best WRMSE = {best[1]:.6f}', loc='right', fontsize=8)
        plt.ylabel("WRMSE")
        x = np.arange(len(train))
        y = 0.078 + 0 * x
        plt.plot(x, y, label="WRMSE = 0.078", color="red")
    else:
        plt.ylabel("Loss")

    plt.plot(best[0], best[1], 'bo', label=f"Best Value\n(Epoch={best[0]}, Value={best[1]:.4f})")

    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close()

# learning rate, epoch and batch size. Can change the parameters here.
def train(lr=0.001, epoch=200, batch_size=64):
    train_loss_curve = []
    train_wrmse_curve = []
    valid_loss_curve = []
    valid_wrmse_curve = []
    best = 100

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel()
    model = model.to(device)
    model.train()

    # dataset and dataloader
    # load data
    full_dataset = pd.read_csv('train.csv', encoding='utf-8')

    # can use torch random_split to create the validation dataset
    lengths = [int(round(len(full_dataset) * 0.8)), int(round(len(full_dataset) * 0.2))]
    train_set, valid_set = random_split(full_dataset, lengths)

    train_dataset = MLDataset(full_dataset.iloc[train_set.indices])
    valid_dataset = MLDataset(full_dataset.iloc[valid_set.indices])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)

    # loss function and optimizer
    # can change loss function and optimizer you want
    criterion  = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # start training
    for e in tqdm(range(epoch)):
        train_loss, valid_loss = 0.0, 0.0
        train_wrmse, valid_wrmse = 0.0, 0.0
        print(f'\nEpoch: {e+1}/{epoch}')
        print('-' * len(f'Epoch: {e+1}/{epoch}'))
        # tqdm to disply progress bar
        for inputs, labels in train_dataloader:
            # data from data_loader
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            outputs = model(inputs)

            # MSE loss and WRMSE
            loss = criterion(outputs, labels)
            wrmse = WRMSE(outputs, labels, device)

            # weights update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss calculate
            train_loss += loss.item()
            train_wrmse += wrmse
        # =================================================================== #
        # If you have created the validation dataset,
        # you can refer to the for loop above and calculate the validation loss
        # tqdm to disply progress bar
        for inputs, labels in valid_dataloader:
            # data from data_loader
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            outputs = model(inputs)

            # MSE loss and WRMSE
            loss = criterion(outputs, labels)
            wrmse = WRMSE(outputs, labels, device)

            # loss calculate
            valid_loss += loss.item()
            valid_wrmse += wrmse

        # =================================================================== #
        # save the best model weights as .pth file
        train_loss_epoch = train_loss / len(train_dataset)
        train_wrmse_epoch = math.sqrt(train_wrmse / len(train_dataset))
        valid_loss_epoch = valid_loss / len(valid_dataset)
        valid_wrmse_epoch = math.sqrt(valid_wrmse/len(valid_dataset))

        if train_wrmse_epoch < best :
            best_wrmse = train_wrmse_epoch
            best_loss = train_loss_epoch
            best_epoch = e
            torch.save(model.state_dict(), 'mymodel.pth')

        print(f'Training loss: {train_loss_epoch:.6f}')
        print(f'Training WRMSE: {train_wrmse_epoch:.6f}')
        print(f'Valid loss: {valid_loss_epoch:.6f}')
        print(f'Valid WRMSE: {valid_wrmse_epoch:.6f}')

        # save loss and wrmse every epoch
        train_loss_curve.append(train_loss_epoch)
        train_wrmse_curve.append(train_wrmse_epoch)
        valid_loss_curve.append(valid_loss_epoch)
        valid_wrmse_curve.append(valid_wrmse_epoch)

    # print the best wrmse
    print(f"\nBest Epoch = {best_epoch}")
    print(f"Best Loss = {best_loss:.4f}")
    print(f"Best WRMSE = {best_wrmse:.4f}\n")

    # generate training curve
    visualize(
        train=train_loss_curve,
        valid=valid_loss_curve,
        title='Loss Curve',
        filename='loss.png',
        best=(e, best_loss)
    )
    visualize(
        train_wrmse_curve,
        valid_wrmse_curve,
        title='WRMSE Curve',
        filename='wrmse.png',
        best=(e, best_wrmse),
        wrmse=True
    )


if __name__ == '__main__':
    train()