import datetime
from pathlib import Path

import torch.cuda
import gym
# from gym.wrappers import FrameStack
from tqdm import tqdm
import sys
sys.path.append('')

import src  # 消さないで

from DQN.agent import Mugicha
from DQN.utils import SkipFrame, MetricLogger, ResizeObservation, GrayScaleObservation, CustomFrameStack

# 環境の作成
env = gym.make("decoponEnv")

env = SkipFrame(env, skip=4)
# env = GrayScaleObservation(env)
# env = ResizeObservation(env, shape=84)
env = CustomFrameStack(env, num_stack=4)

#環境の初期化
env.reset()

next_state, reward, done, _, info = env.step(action=0)

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("trained_models")
log_dir = Path("log")

# save_dir.mkdir(parents=True)

mugicha = Mugicha(img_dim=(4, 84, 84), poly_feature_dim=8, action_dim=env.action_space.n, save_dir=save_dir)
load_path = Path("trained_models/mugicha_net_0.chkpt")
mugicha.load(load_path)

logger = MetricLogger(log_dir)

episodes = 10005


for e in tqdm(range(episodes)):

    state, _ = env.reset()

    # ゲーム開始！
    while True:

        # 現在の状態に対するエージェントの行動を決める
        action = mugicha.act(state)

        # エージェントが行動を実行
        next_state, reward, done, _, info = env.step(action)

        # 記憶
        mugicha.cache(state, next_state, action, reward, done)

        # 訓練
        q, loss = mugicha.learn()

        # ログ保存
        logger.log_step(reward, loss, q)

        # 状態の更新
        state = next_state

        # ゲーム画面の描画
        # env.render()

        # ゲームが終了したかどうかを確認
        # if done or info["flag_get"]:  # ゲームの終了条件によってはこれ
        if done:
            mugicha.save()
            break

    # エピソードが終了したので、イプシロンを更新
    mugicha.update_exploration_rate()

    logger.log_episode()

    if e % 20 == 0:
        logger.record(episode=e, epsilon=mugicha.exploration_rate, step=mugicha.curr_step)