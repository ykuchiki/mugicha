from pathlib import Path

import torch.cuda
import gym
from gym.wrappers import FrameStack
from tqdm import tqdm
from PIL import Image
import numpy as np

import sys
sys.path.append("")

import src  # 消さないで

from DQN.agent import Mugicha
from DQN.utils import SkipFrame, MetricLogger, ResizeObservation, GrayScaleObservation

# 環境の作成
env = gym.make("MugichaEnv")

# env = SkipFrame(env, skip=4)
# env = FrameStack(env, num_stack=1)

# 環境の初期化
env.reset()

next_state, reward, done, _, info = env.step(action=0)

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("trained_models")
log_dir = Path("log")

# save_dir.mkdir(parents=True)


mugicha = Mugicha(state_dim=(1, 84, 84), action_dim=348, save_dir=save_dir)
# load_path = Path("trained_models/mugicha_net_0.chkpt")
# mugicha.load(load_path)

logger = MetricLogger(log_dir)


episodes = 10000

for e in tqdm(range(episodes)):

    state, _ = env.reset()

    # ゲーム開始！
    while True:

        # 現在の状態に対するエージェントの行動を決める
        action = mugicha.act(state)

        # エージェントが行動を実行
        next_state, reward, done, _, info = env.step(action)
        print("reward is ", reward)

        # float32 -> 画像として保存
        image_data = Image.fromarray(np.int8(state*255), "L")
        image_data.save("game_state.png")

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