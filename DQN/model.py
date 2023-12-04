import copy

import torch
import torch.nn.functional as F
from torch import nn, optim


class MugichaNet(nn.Module):
    """
     単純なCNNを実装
     input -> (conv2d + relu) × 3 -> flatten -> (dense + relu) × 2 -> output
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(1, -1),
            #nn.Linear(3136, 512),
            #nn.ReLU(),
            #nn.Linear(512, output_dim),
        )

        self.target = copy.deepcopy(self.online)

        # Q_target のパラメータは固定されます
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            s = self.online(input)
            print(s.shape)
            return s
        elif model == "target":
            return self.target(input)