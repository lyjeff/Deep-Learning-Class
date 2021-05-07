import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.nn_layers = nn.Sequential(
            # ========================================================== #
            # fully connected layer
            nn.Linear(in_features=53, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=1),
            # ========================================================== #
        )

    def forward(self, x):
        # data fit into model
        x = self.nn_layers(x)
        return x


