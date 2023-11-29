from collections import namedtuple
from PIL import Image
import random

import gym
from gym import spaces
import numpy as np
import pygame
import pymunk
import torch
import torchvision.transforms as T

from decopon.controller import Controller, Human

HEIGHT, WIDTH = 640, 480
TIMELIMIT = 1000
LINE = 200

POLY_FEAT_NUM = 8

# オブジェクト(質量，半径，色，スコア，インデックス)
Polygon = namedtuple("Polygon", ["mass", "radius", "color", "score", "index"])
Polygons = [
    Polygon(1, 13, (255, 0, 127), 0, 0),
    Polygon(2, 17, (255, 0, 255), 1, 1),
    Polygon(3, 24, (127, 0, 255), 3, 2),
    Polygon(4, 28, (0, 0, 255), 6, 3),
    Polygon(5, 35, (0, 127, 255), 10, 4),
    Polygon(6, 46, (0, 255, 255), 15, 5),
    Polygon(7, 53, (0, 255, 127), 21, 6),
    Polygon(8, 65, (0, 255, 0), 28, 7),
    Polygon(9, 73, (127, 255, 0), 36, 8),
    Polygon(10, 90, (255, 255, 0), 45, 9),
    Polygon(11, 100, (255, 127, 0), 55, 10),
]




class MugichaEnv(gym.Env):
    def __init__(self, controller: Controller):
        pygame.init()  # pygameの全システムを初期化
        pygame.font.init()  # フォントシステムの初期化
        super(MugichaEnv, self).__init__()

        # アクション空間を定義
        self.action_space = spaces.Discrete(3)  # 左，右，落とす
        # 観測空間を定義 84×84のグレースケール(128×128もしくはRGB画像にするか悩み)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

        # ゲーム設定の初期化
        self.controller = controller
        self._setup_game()

        self.episode_start_time = 0  # 残り時間

    def _setup_game(self):
        pygame.init()
        pygame.display.set_caption("Decopon")

        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        self.fps = pygame.time.Clock().tick
        self.space = pymunk.Space()
        self.space.gravity = (0, 1000)

        # 衝突処理の設定
        collision_handler = self.space.add_collision_handler(1, 1)
        collision_handler.begin = self.merge  # 引数はpymunkが自動で提供

        # ゲームオブジェクトの生成
        self.walls = []
        self.create_walls()

        self.indicator = pygame.Rect(WIDTH / 2, 100, 3, HEIGHT - 100)

        self.poly = []

        self.drop_ticks = pygame.time.get_ticks()

        self.current = random.randint(0, 4)
        self.next = random.randint(0, 4)

        self.font = pygame.font.Font("resources/BestTen-DOT.otf", 16)
        self.score = 0
        self.reward = 0


        self.previous_score = 0

        self.isGameOver = False
        self.countOverflow = 0

        self.progress = [pygame.Rect(10 + i * 20, 70, 20, 20) for i in range(11)]

    # ポリゴンが衝突した時の処理
    def merge(self, polys, space, _):
        p0, p1 = polys.shapes

        if p0.index == 10 and p1.index == 10:
            if p0 in self.poly:
                self.poly.remove(p0)
            if p1 in self.poly:
                self.poly.remove(p1)
            space.remove(p0, p0.body, p1, p1.body)
            self.score += 100
            return False

        if p0.index == p1.index:
            self.score += Polygons[p0.index].score
            x = (p0.body.position.x + p1.body.position.x) / 2
            y = (p0.body.position.y + p1.body.position.y) / 2
            self.create_poly(x, y, p1.index + 1)
            if p0 in self.poly:
                self.poly.remove(p0)
            if p1 in self.poly:
                self.poly.remove(p1)

            space.remove(p0, p0.body, p1, p1.body)

        return True

    def create_walls(self):
        floor = pymunk.Segment(self.space.static_body, (50, HEIGHT - 10), (WIDTH - 50, HEIGHT - 10), 10)
        floor.friction = 0.8
        floor.elasticity = 0.8

        self.walls.append(floor)
        self.space.add(floor)

        left = pymunk.Segment(self.space.static_body, (50, 200), (50, HEIGHT), 10)
        left.friction = 1.0
        left.elasticity = 0.95
        self.walls.append(left)
        self.space.add(left)

        right = pymunk.Segment(self.space.static_body, (WIDTH - 50, 200), (WIDTH - 50, HEIGHT), 10)
        right.friction = 1.0
        right.elasticity = 0.95
        self.walls.append(right)
        self.space.add(right)

        self.start_time = pygame.time.get_ticks()

    # イベントのチェック
    def check_event(self, event):
        for e in pygame.event.get():
            if e.type == event:
                return True
        return False

    # ポリゴンの生成
    def create_poly(self, x: int, y: int, idx: int):
        poly = Polygons[idx]

        mass = poly.mass
        radius = poly.radius
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = (x, y)
        body.radius = radius
        shape = pymunk.Circle(body, radius)
        shape.friction = 0.5
        shape.elasticity = 0.4
        shape.collision_type = 1
        shape.index = poly.index
        shape.color = poly.color
        self.space.add(body, shape)
        self.poly.append(shape)

    # 壁の描画
    def draw_walls(self):
        pygame.draw.line(self.window, (240, 190, 153), (40, 200), (WIDTH - 40, 200), 3)
        for wall in self.walls:
            p1 = int(wall.a[0]), int(wall.a[1])
            p2 = int(wall.b[0]), int(wall.b[1])
            pygame.draw.line(self.window, (240, 190, 153), p1, p2, 20)

    # オーバーフローのチェック
    def check_overflow(self):
        for poly in self.poly:
            if poly.body.position.y - (poly.radius) < LINE:
                return True
        return False

    def reset(self):
        """エピソード開始時に状態をリセットする"""
        self._setup_game()
        self.episode_start_time = pygame.time.get_ticks()  # エピソード開始時間を記録
        return self._get_observation(), {}

    def _get_observation(self):
        """ゲーム画面からの現在の観測を取得する"""
        # pygameの画面をキャプチャする
        screen_surface = pygame.display.get_surface()
        screen_data = pygame.surfarray.array3d(screen_surface)

        # PILライブラリを使用して画像を処理する
        image = Image.fromarray(screen_data, 'RGB')

        # NumPy配列に変換する
        image_feature = np.array(image)

        # 画像の向きを正しく
        image_feature = np.rot90(image_feature, k=-1)

        image_feature = self.process_image(image_feature)

        # ポリゴンの特性の総和を計算
        if self.poly:
            poly_features = [self._get_poly_features(poly) for poly in self.poly]
            poly_feature = np.sum(poly_features, axis=0, dtype=np.float32)
        else:
            # ポリゴンがない場合は、ゼロベクトルを使用
            poly_feature = np.zeros(POLY_FEAT_NUM, dtype=np.float32)

        observation = {
            "image": image_feature,
            "poly_features": poly_feature
        }
        return observation

    def step(self, action):
        """与えられたアクションに基づいて環境を更新し，新しい状態と報酬を返す"""
        # アクションに応じてゲームの状態を更新
        #if self.check_event(pygame.QUIT):
        #   break
        if self.check_overflow():
            self.countOverflow += 1

        if action == 0:
            self.indicator.centerx -= 3
        elif action == 1:
            self.indicator.centerx += 3
        elif action == 2 and pygame.time.get_ticks() - self.drop_ticks > 500 and not self.check_overflow():
            self.create_poly(self.indicator.centerx, self.indicator.topleft[1], self.current)
            self.drop_ticks = pygame.time.get_ticks()
            self.current = self.next
            self.next = random.randint(0, 4)
            self.countOverflow = 0

        if self.indicator.centerx < 65:
            self.indicator.centerx = WIDTH - 65
        if self.indicator.centerx > WIDTH - 65:
            self.indicator.centerx = 65

        # ゲーム状態の更新
        # self.space.step(1 / 60)

        observation = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()

        truncated = {}
        info = {}  # 必要な追加情報があればここ実装

        return observation, reward, done, {}, info

    def render(self, mode='human'):
        """環境の状態を描画"""
        # ゲーム画面の描画
        self.window.fill((89, 178, 36))  # 背景色
        pygame.draw.rect(self.window, (255, 255, 255), self.indicator)

        # 現在と次のポリゴンを描画
        current_poly = Polygons[self.current]
        pygame.draw.circle(self.window, current_poly.color, (self.indicator.centerx, self.indicator.topleft[1]),
                           current_poly.radius)
        next_poly = Polygons[self.next]
        pygame.draw.circle(self.window, next_poly.color, (WIDTH - 60, 60), next_poly.radius)

        for poly in self.poly:
            pygame.draw.circle(
                self.window, poly.color, (int(poly.body.position.x), int(poly.body.position.y)), poly.radius
            )

        # 壁の描画
        self.draw_walls()

        # スコアと残り時間を描画
        score_text = self.font.render(f"スコア: {self.score}", True, (255, 255, 255))
        self.window.blit(score_text, (10, 10))
        elapsed_time = (pygame.time.get_ticks() - self.episode_start_time) // 1000  # 経過時間を計算
        time_text = self.font.render(f"残り時間: {TIMELIMIT - elapsed_time}", True, (255, 255, 255))
        self.window.blit(time_text, (10, 30))

        text = self.font.render("シンカ", True, (255, 255, 255))
        self.window.blit(text, (10, 50))
        for i, poly in enumerate(self.progress):
            pygame.draw.rect(self.window, Polygons[i].color, poly)

        self.space.step(1 / 60)
        # 画面を更新
        pygame.display.update()
        # フレームレートの制限
        self.fps(60)

    def close(self):
        """環境を閉じる際のクリーンアップ"""
        pygame.quit()

    def _get_reward(self):
        """現在のアクションに対する報酬を取得する"""
        # 現在のスコアと前のステップのスコアを比較
        score_change = self.score - self.previous_score
        self.previous_score = self.score

        # スコアが増加した場合，正の報酬
        if score_change > 0:
            self.reward = 0.5 * score_change
        # 時間だけ経過してるなら負の報酬
        else:
            self.reward -= 0.0003

        if self.check_threshold():
            self.reward -= 0.04

        return self.reward

    def _get_poly_features(self, poly):
        """ポリゴンの特徴を取得する"""
        # ポリゴンの位置、質量、半径を正規化
        x_norm = (poly.body.position.x - WIDTH / 2) / (WIDTH / 2)
        y_norm = (poly.body.position.y - HEIGHT / 2) / (HEIGHT / 2)
        mass_norm = poly.mass / max(p.mass for p in Polygons)
        radius_norm = poly.radius / max(p.radius for p in Polygons)

        # 色の値を0-1の範囲に正規化
        r_norm, g_norm, b_norm = [c / 255.0 for c in poly.color]


        score_norm = Polygons[poly.index].score / max(p.score for p in Polygons)

        return np.array([x_norm, y_norm, mass_norm, radius_norm, r_norm, g_norm, b_norm, score_norm], dtype=np.float32)

    def _is_done(self):
        """ゲームが終了したかどうかを判定"""
        # タイムリミット
        if (pygame.time.get_ticks() - self.start_time) // 1000 > TIMELIMIT:
            return True

        if self.isGameOver:
            return True

        # この値は調整の余地あり
        if self.countOverflow > 300:
            self.show_game_over()
            return True
        return False

    def show_game_over(self):
        # ゲームオーバーメッセージのフォントとサイズを設定
        game_over_font = pygame.font.Font(None, 90)  # ここで None はデフォルトフォント
        game_over_surface = game_over_font.render("GAME OVER!!", True, (255, 0, 0))  # 赤色のテキスト
        game_over_rect = game_over_surface.get_rect()
        game_over_rect.center = (WIDTH // 2, HEIGHT // 2)  # 画面の中央に配置

        # ゲームオーバーメッセージを画面に描画
        self.window.blit(game_over_surface, game_over_rect)
        pygame.display.update()  # 画面の更新

        # 一定時間表示した後に画面を閉じる
        pygame.time.wait(300)  # 3000ミリ秒（3秒）待機

    def check_threshold(self):
        threshold = LINE * 1.1  # 閾値の110%
        for poly in self.poly:
            if poly.body.position.y + poly.radius > threshold:
                return True
        return False

    def process_image(self, image, shape=84):
        image = torch.tensor(image.copy(), dtype=torch.float).transpose(0, 2).transpose(1, 2)
        # グレースケール化
        transform = T.Grayscale()
        image = transform(image)

        # リサイズ
        resize_shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        transform_to_resize = T.Compose([
            T.ToPILImage(),
            T.Resize(resize_shape),
            T.ToTensor()
        ])
        image = transform_to_resize(image)

        image = image.squeeze(0)

        return image

