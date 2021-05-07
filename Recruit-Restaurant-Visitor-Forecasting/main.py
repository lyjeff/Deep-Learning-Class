import pandas as pd
import numpy as np
from tqdm import tqdm
from math import sqrt

import torch, os
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from models import MyModel
from dataset import Visitor_Dataset

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = 1

    def forward(self, pred, actual):
        pred_log = torch.log(pred + 1)
        actual_log = torch.log(actual + 1)
        pred_log[pred_log != pred_log] = self.eps
        actual_log[actual_log != actual_log] = self.eps
        return torch.sum(torch.pow(torch.sub(pred_log, actual_log), 2))

def visualize(value, title, filename):
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(value, label=title)

    plt.legend(loc="best")
    plt.savefig(filename)
    plt.close()

def train(dataset_path, lr, epoch, batch_size, scaler_flag, state_name):
    train_loss_curve = []
    best = -1

    # load model
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = MyModel()
    model = model.to(device)
    model.train()

    # dataset and dataloader
    full_dataset = Visitor_Dataset(dataset_path, scaler_flag)
    train_dataloader = DataLoader(dataset=full_dataset, batch_size=batch_size, shuffle=True)

    # loss function and optimizer
    criterion = RMSLELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # start training
    for e in range(epoch):
        train_loss, train_rmsle = 0.0, 0.0

        print(f'\nEpoch: {e+1}/{epoch}')
        print('-' * len(f'Epoch: {e+1}/{epoch}'))
        # tqdm to disply progress bar
        for inputs, labels in tqdm(train_dataloader):

            # data from data_loader
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            outputs = model(inputs)

            # RMSLE Loss
            loss = criterion(outputs, labels)

            # weights update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss and rmsle calculate
            train_loss += loss.item()

        # save the best model weights as .pth file
        train_loss_epoch = sqrt(train_loss / len(full_dataset))

        if best == -1 or train_loss_epoch < best :
            best_loss = train_loss_epoch
            best_epoch = e
            torch.save(model.state_dict(), f'mymodel_{state_name}.pth')

        print(f'Training Loss: {train_loss_epoch:.6f}')

        # save loss and RMSLE every epoch
        train_loss_curve.append(train_loss_epoch)

    # print the best RMSLE
    print(f"Final Training RMSLE Loss = {best_loss:.6f}")

    visualize(
        value=train_loss_curve,
        title='Train Loss Curve',
        filename=f'rmsle_{state_name}.png'
    )

    return full_dataset

def test(full_dataset, state_name):
    # load model and use weights we saved before.
    model = MyModel()
    model.load_state_dict(torch.load(f'mymodel_{state_name}.pth', map_location='cpu'))
    model.eval()

    criterion = RMSLELoss()

    # convert dataframe to tensor
    inputs = full_dataset.test[full_dataset.col].astype(float).values
    inputs = torch.tensor(inputs).float()

    # predict
    outputs = model(inputs)

    # get labels
    labels = full_dataset.test_target.values

    # RMSLE Loss
    loss = criterion(outputs, torch.from_numpy(labels))
    test_loss = sqrt(loss / len(full_dataset.test))

    print(f'Testing Loss: {test_loss:.6f}')

    #save the result
    result = full_dataset.test["id"].to_frame()
    result.insert(1, "visitors_pred", outputs.detach().numpy())
    result.insert(2, "visitors_actual", labels)
    result.to_csv(f'result_{state_name}.csv', index=False)

if __name__ == '__main__':

    dataset_path = "./dataset/"
    state_name="final"
    lr = 0.00001
    epoch = 15
    batch_size = 64
    scaler_flag = True

    full_dataset = train(dataset_path, lr, epoch, batch_size, scaler_flag, state_name)
    test(full_dataset, state_name)