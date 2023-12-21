import copy

import torch
import torch.nn.functional as F
from torch import nn, optim


class MugichaNet(nn.Module):
    """
     単純なCNNを実装
     input -> (conv2d + relu) × 3 -> flatten -> (dense + relu) × 2 -> output
    """

    def __init__(self, input_dim, output_dim, drop_poly_dim, poly_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(1, -1),
            # nn.Linear(3136, 512),
            # nn.ReLU(),
            # nn.Linear(512, output_dim),
        )

        # ポリゴン情報の全結合層
        self.fc_drop_poly = nn.Linear(drop_poly_dim, 128)
        self.fc_poly = nn.Linear(poly_dim, 128)

        # 画像とポリゴン情報を組み合わせた層
        self.fc_combined = nn.Sequential(
            nn.Linear(3136 + 128 + 128, 512),  # 3136は畳み込み層の出力サイズ
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.online = nn.Sequential(
            self.fc_combined
        )

        self.target = copy.deepcopy(self.online)

        # Q_target のパラメータは固定されます
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, image, drop_poly, poly, model):
        conv_output = self.conv_layers(image)
        drop_poly_output = F.relu(self.fc_drop_poly(drop_poly))
        poly_output = F.relu(self.fc_poly(poly))
        combined_input = torch.cat([conv_output, drop_poly_output, poly_output], dim=1)

        if model == "online":
            return self.online(combined_input)
        elif model == "target":
            return self.target(combined_input)