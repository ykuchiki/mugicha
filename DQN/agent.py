import random
from collections import deque
import pickle
import torch
import numpy as np
import os

from DQN.model import MugichaNet

CAPACITY = 100000
BATCH_SIZE = 32
GAMMA = 0.9
SAVE_FREQUENCY = 5e5

ERD = 0.9998
# ERD = 0.0001
lr = 0.00025

class Mugicha:
    def __init__(self, state_dim, action_dim, drop_poly_dim, poly_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.drop_poly_dim = drop_poly_dim
        self.poly_dim = poly_dim
        self.save_dir = save_dir

        self.best_reward = -float('inf')

        self.use_cuda = torch.cuda.is_available()


        self.memory = deque(maxlen=CAPACITY)
        self.batch_size = BATCH_SIZE

        self.gamma = GAMMA

        # モデルのインスタンス
        self.net = MugichaNet(self.state_dim, self.action_dim, self.drop_poly_dim, self.poly_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")

        self.exploration_rate = 1
        self.exploration_rate_decay = ERD
        self.exploration_rate_min = 0.01
        # self.episode_number = 0  # エピソード数の初期化
        self.curr_step = 0

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.save_every = SAVE_FREQUENCY  # モデルを保存するまでの実験ステップの数

        self.burnin = 1e4  # 経験を訓練させるために最低限必要なステップ数
        # self.burnin = 41
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
        # 探索（EXPLORE）
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim) 

        # 活用（EXPLOIT）
        else:
            print("AI ACTION")
            # image_data = state["image"].__array__().astype(np.float32) / 255
            image_data = state["image"]
            drop_poly_info = torch.tensor(state["drop_poly"], dtype=torch.float32).unsqueeze(0)
            poly_info = torch.tensor(state["poly"], dtype=torch.float32).unsqueeze(0)
            
            if self.use_cuda:
                # image_data = torch.tensor(state).cuda()
                image_data = torch.tensor(image_data).cuda()
                drop_poly_info = drop_poly_info.cuda()
                poly_info = poly_info.cuda()
            else:
                image_data = torch.tensor(image_data.copy())

            image_data = image_data.unsqueeze(0)
            image_data = image_data.unsqueeze(0)
            action_values, _ = self.net(image_data, drop_poly_info, poly_info, model="online")
            action_idx = torch.argmax(action_values, axis=1).item() 

        #self.update_exploration_rate()

        # ステップを+1します
        self.curr_step += 1
        # アクションインデックスを実際のx座標に変換
        actual_action = action_idx * 29 + 66 
        return actual_action

    def update_exploration_rate(self):
        # exploration_rateを減衰させます
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        print("ERD is ", self.exploration_rate)

    def update_exploration_rate_(self):
        # 二次関数的な減衰を適用
        self.exploration_rate = max(self.exploration_rate_min, 1 - self.exploration_rate_decay * (self.episode_number ** 2))

    def cache(self, state, next_state, action, reward, done):
        state_image = torch.tensor(state['image'].__array__(), dtype=torch.float32)
        state_drop_poly = torch.tensor(state['drop_poly'], dtype=torch.float32)
        state_poly = torch.tensor(state['poly'], dtype=torch.float32)

        next_state_image = torch.tensor(next_state['image'].__array__(), dtype=torch.float32)
        next_state_drop_poly = torch.tensor(next_state['drop_poly'], dtype=torch.float32)
        next_state_poly = torch.tensor(next_state['poly'], dtype=torch.float32)

        action = torch.tensor([action], dtype=torch.int64)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.bool)

        if self.use_cuda:
            state_image = state_image.cuda()
            state_drop_poly = state_drop_poly.cuda()
            state_poly = state_poly.cuda()
            next_state_image = next_state_image.cuda()
            next_state_drop_poly = next_state_drop_poly.cuda()
            next_state_poly = next_state_poly.cuda()
            action = action.cuda()
            reward = reward.cuda()
            done = done.cuda()

        self.memory.append((state_image, state_drop_poly, state_poly, next_state_image, next_state_drop_poly, next_state_poly, action, reward, done))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state_image, state_drop_poly, state_poly, next_state_image, next_state_drop_poly, next_state_poly, action, reward, done = map(torch.stack, zip(*batch))

        state_image = state_image.unsqueeze(1)
        next_state_image = next_state_image.unsqueeze(1)

        return state_image, state_drop_poly, state_poly, next_state_image, next_state_drop_poly, next_state_poly, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state_image, state_drop_poly, state_poly, action):
        action = (action - 66) // 29
        state_image = state_image.float()
        current_Q_values, _ = self.net(state_image, state_drop_poly, state_poly, model='online')
        current_Q = current_Q_values[np.arange(0, self.batch_size), action] 
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state_image, next_state_drop_poly, next_state_poly, done):
        # 次の状態での最善の行動をオンラインモデルで決定
        next_state_Q_values, _ = self.net(next_state_image, next_state_drop_poly, next_state_poly, model="online")
        best_action = torch.argmax(next_state_Q_values, axis=1)

        # 次の状態での最善の行動のQ値をターゲットモデルで決定
        next_Q_values, _ = self.net(next_state_image, next_state_drop_poly, next_state_poly, model="target")
        next_Q = next_Q_values[np.arange(0, self.batch_size), best_action]

        # TDターゲットの計算
        td_target = reward + (1 - done.float()) * self.gamma * next_Q

        return td_target.float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self, SAVE_NUM):
        save_path = (
                self.save_dir / f"mugicha_net_{SAVE_NUM}.chkpt"
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
            self.save(0)

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

            # メモリからサンプリング
        state_image, state_drop_poly, state_poly, next_state_image, next_state_drop_poly, next_state_poly, action, reward, done = self.recall()

        # TD Estimateの取得
        td_est = self.td_estimate(state_image, state_drop_poly, state_poly, action)

        # TD Targetの取得
        td_tgt = self.td_target(reward, next_state_image, next_state_drop_poly, next_state_poly, done)

        # 損失をQ_onlineに逆伝播させる
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate

    def save_memory(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)

    def load_memory(self, filename):
        with open(filename, 'rb') as f:
            self.memory = pickle.load(f)

    def end_episode(self, total_reward, SAVE_NUM):
        # エピソード終了時の処理
        if total_reward > self.best_reward:
            self.best_reward = total_reward  # 最高報酬を更新
            self.save(SAVE_NUM)  # モデルを保存

    def end_episode_(self, total_reward, SAVE_NUM):
        # エピソード終了時の処理
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.save(SAVE_NUM)
        
        # 探索率の更新
        self.update_exploration_rate()
        self.episode_number += 1  # エピソード数をインクリメント