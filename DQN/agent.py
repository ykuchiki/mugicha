import random
from collections import deque

import torch
import numpy as np

from DQN.model import MugichaNet

CAPACITY = 100000
BATCH_SIZE = 32
GAMMA = 0.9
SAVE_FREQUENCY = 5e5

ERD = 0.9998  # イプシロンの減少率，経験的にこれが良さそう

IMG_WEIGHT = 5.0
POLY_WEIGHT = 1.0

# MODE = "img_only"
MODE = "yeah"


class Mugicha:
    def __init__(self, img_dim, poly_feature_dim, action_dim, save_dir):
        self.img_dim = img_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        self.memory = deque(maxlen=CAPACITY)
        self.batch_size = BATCH_SIZE

        self.gamma = GAMMA

        # モデルのインスタンス
        self.net = MugichaNet(self.img_dim, poly_feature_dim, self.action_dim, img_weight=IMG_WEIGHT, poly_weight=POLY_WEIGHT, mode=MODE).float()
        if self.use_cuda:
            self.net = self.net.to(device='cuda')

        self.exploration_rate = 1.0
        self.exploration_rate_decay = ERD
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.save_every = SAVE_FREQUENCY  # モデルを保存するまでの実験ステップの数

        self.burnin = 1e4  # 経験を訓練させるために最低限必要なステップ数
        self.learn_every = 3  # Q_onlineを更新するタイミングを示すステップ数
        self.sync_every = 1e4  # Q_target & Q_onlineを同期させるタイミングを示すステップ数

    def act(self, state):
        """
        状態が与えられたとき、ε-greedy法に従って行動を選択

        Inputs:
            state(LazyFrame):現在の状態における一つの観測オブジェクトで、(state_dim)次元となる
        Outputs:
            action_idx (int): AIが取る行動を示す整数値
        """
        # print(self.exploration_rate)
        # 探索（EXPLORE）
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # 活用（EXPLOIT）
        else:
            # state = state.__array__()
            image_data = state["image"].__array__()
            poly_features = state["poly_features"].__array__()

            if self.use_cuda:
                image_data = torch.tensor(image_data).cuda()
                poly_features = torch.tensor(poly_features).cuda()
            else:
                image_data = torch.tensor(image_data.copy())
                poly_features = torch.tensor(poly_features.copy())

            # state = state.unsqueeze(0)
            # 画像データの形状を変更
            image_data = image_data.unsqueeze(0)  # バッチ次元追加

            # action_values = self.net(state, model="online")
            action_values = self.net(image_data, poly_features, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # stepごとのイプシロンを減少させるときはこれ
        # self.update_exploration_rate()

        # ステップを+1します
        self.curr_step += 1
        return action_idx
    
    def update_exploration_rate(self):
        """イプシロンを減衰させる"""
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

    def cache(self, state, next_state, action, reward, done):
        """
        経験をself.memory (replay buffer)に保存

        Inputs:
            state (LazyFrame),
            next_state (LazyFrame),
            action (int),
            reward (float),
            done(bool))
        """
        # state = state.__array__()
        # next_state = next_state.__array__()
        # 画像データとポリゴンの特性データに分離
        image_data = state["image"].__array__()
        poly_features = state["poly_features"].__array__()

        next_image_data = next_state["image"].__array__()
        next_poly_features = next_state["poly_features"].__array__()

        if self.use_cuda:
            image_data = torch.tensor(image_data).cuda()
            poly_features = torch.tensor(poly_features).cuda()
            next_image_data = torch.tensor(next_image_data).cuda()
            next_poly_features = torch.tensor(next_poly_features).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            image_data = torch.tensor(image_data)
            poly_features = torch.tensor(poly_features)
            next_image_data = torch.tensor(next_image_data)
            next_poly_features = torch.tensor(next_poly_features)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((image_data, poly_features, next_image_data, next_poly_features, action, reward, done))

    def recall(self):
        """
        メモリから経験のバッチを取得
        """
        batch = random.sample(self.memory, self.batch_size)
        image_data, poly_features, next_image_data, next_poly_features, action, reward, done = map(torch.stack,
                                                                                                   zip(*batch))
        return image_data, poly_features, next_image_data, next_poly_features, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, image_data, poly_features, action):
        current_Q = self.net(image_data, poly_features, model='online')[np.arange(0, self.batch_size), action] # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_image_data, next_poly_features, done):
        next_state_Q = self.net(next_image_data, next_poly_features, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_image_data, next_poly_features, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
                self.save_dir / f"mugicha_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MugichaNet saved to {save_path} at step {self.curr_step}")

    def learn(self):
        """経験のデータのバッチで、オンラインに行動価値関数(Q)を更新"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # メモリからサンプリング
        image_data, poly_features, next_image_data, next_poly_features, action, reward, done = self.recall()

        # TD Estimateの取得
        td_est = self.td_estimate(image_data, poly_features, action)

        # TD Targetの取得
        td_tgt = self.td_target(reward, next_image_data, next_poly_features, done)

        # 損失をQ_onlineに逆伝播させる
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        # exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        # print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        # self.exploration_rate = exploration_rate
