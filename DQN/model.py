import copy

import torch
import torch.nn.functional as F
from torch import nn, optim


class MugichaNet(nn.Module):
    """
     単純なCNNを実装
     input -> (conv2d + relu) × 3 -> flatten -> (dense + relu) × 2 -> output
    """

    def __init__(self, img_dim, poly_feature_dim, output_dim, img_weight=5.0, poly_weight=1.0, mode="img_only"):
        super().__init__()
        c, h, w = img_dim
        self.img_weight = img_weight
        self.poly_weight = poly_weight
        self.mode = mode

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        if self.mode == "img_only":
            self.conv_net = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            )
        else:
            self.conv_net = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )

        # ポリゴンの物理的特性用のネットワーク
        self.fc_poly = nn.Sequential(
            nn.Linear(poly_feature_dim, 128),
            nn.ReLU(),
            # nn.Linear(128, output_dim),
        )

        # ポリゴンと画像を組み合わせた最終的な全結合層
        self.fc_final = nn.Sequential(
            nn.Linear(3136 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.online = nn.ModuleDict({
            "conv_net": self.conv_net,
            "fc_poly": self.fc_poly,
            "fc_final": self.fc_final
        })

        self.target = copy.deepcopy(self.online)

        # Q_target のパラメータは固定
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, image_input, poly_input, model):
        model = self.online if model == "online" else self.target

        # 画像のみ使うとき
        if self.mode == "img_only":
            return model["conv_net"](image_input)

        # 画像データの処理
        conv_output = self.img_weight * model["conv_net"](image_input)

        # ポリゴンの特性データの処理
        poly_output = self.poly_weight * model["fc_poly"](poly_input)
        if poly_output.dim() == 1:
            poly_output = poly_output.unsqueeze(0)

        # 処理した特徴を結合
        combined_input = torch.cat((conv_output, poly_output), dim=1)

        # 結合した特徴を最終層に渡す
        return model["fc_final"](combined_input)