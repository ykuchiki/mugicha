from DQN import environment as ev
from decopon.controller import Human, AI
from gym.wrappers import FrameStack
from DQN.agent import Mugicha
from DQN.utils import SkipFrame, ResizeObservation, GrayScaleObservation, CustomFrameStack

import pygame
from PIL import Image
from pathlib import Path
import gym

import src

class Game:
    def __init__(self):
        # pygame.init()
        self.env = ev.MugichaEnv(Human())
        self.env.reset()

    def run(self):
        while True:
            # ゲームイベントの処理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # ユーザー入力の取得
            is_left, is_right, is_drop = self.env.controller.update()

            # 環境の更新
            action = None  # デフォルトアクション
            if is_left:
                action = 0
            elif is_right:
                action = 1
            elif is_drop:
                action = 2

            observation, reward, done, _, info = self.env.step(action)
            # image = Image.fromarray(observation, 'L')
            # image.save("game_state.png", format="PNG")

            # ゲーム画面の描画
            self.env.render()

            # エピソードが終了，リセットかゲームを閉じる
            if done:
                # self.env.reset()
                self.env.close()


class AIDrive:
    def __init__(self):
        # pygame.init()
        # 環境の作成
        self.env = gym.make("MugichaEnv")

        self.env = SkipFrame(self.env, skip=1)
        # self.env = GrayScaleObservation(self.env)
        # self.env = ResizeObservation(self.env, shape=84)
        # self.env = FrameStack(self.env, num_stack=4)
        self.env = CustomFrameStack(self.env, num_stack=4)
        self.env.reset()

        self.state, _ = self.env.reset()
        save_dir = Path("trained_models")
        self.mugicha = Mugicha(img_dim=(4, 84, 84), poly_feature_dim=8, action_dim=self.env.action_space.n, save_dir=save_dir)
        load_path = Path("trained_models/mugicha_net_0.chkpt")
        self.mugicha.load(load_path)


    def run(self):
        while True:
            # 現在の状態に対するエージェントの行動を決める
            action = self.mugicha.act(self.state)

            # エージェントが行動を実行
            next_state, reward, done, _, info =self.env.step(action)

            with open("test.txt", "a") as file:
                file.write(f"{reward} + \n")

            # ゲーム画面の描画
            self.env.render()

            self.state = next_state

            # エピソードが終了，リセットかゲームを閉じる
            if done:
                self.env.reset()
                # self.env.close()


if __name__ == "__main__":
    # game = Game()
    game = AIDrive()
    game.run()
