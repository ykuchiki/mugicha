from abc import ABC, abstractmethod
from typing import Tuple

import pygame


class Controller(ABC):
    @abstractmethod
    def update(self, info) -> Tuple[bool, bool, bool]:
        return True, True, True


class Human(Controller):
    def __init__(self) -> None:
        super().__init__()

    def update(self, info) -> Tuple[bool, bool, bool]:
        pressedKeys = pygame.key.get_pressed()
        return pressedKeys[pygame.K_LEFT], pressedKeys[pygame.K_RIGHT], pressedKeys[pygame.K_SPACE]

import sys 
sys.path.append("")
from DQN.model import MugichaNet
import torch

class AI(Controller):
    def __init__(self) -> None:
        super().__init__()
        self.net = MugichaNet((1, 84, 84), 12, 3, 120).float()
        cpt = torch.load("trained_models/mugicha_net_0.chkpt")
        stdict_m = cpt["model"]
        self.net.load_state_dict(stdict_m)

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.net = self.net.to(device="cuda")

    @torch.no_grad()
    def update(self, data_pack) -> Tuple[bool, bool, bool]:
        state, current_poly_index = data_pack
        image_data = state["image"]
        drop_poly_info = torch.tensor(state["drop_poly"], dtype=torch.float32).unsqueeze(0)
        poly_info = torch.tensor(state["poly"], dtype=torch.float32).unsqueeze(0)

        # GPUの場合
        if self.use_cuda:
            # image_data = torch.tensor(state).cuda()
            image_data = torch.tensor(image_data).cuda()
            drop_poly_info = drop_poly_info.cuda()
            poly_info = poly_info.cuda()
        else:
            image_data = torch.tensor(image_data)

        image_data = image_data.unsqueeze(0)
        image_data = image_data.unsqueeze(0)
        action_values, _ = self.net(image_data, drop_poly_info, poly_info, model="online")
        action_idx = torch.argmax(action_values, axis=1).item()
        actual_action = action_idx * 29 + 66
        return actual_action

        #if abs(indicator_centerx - actual_action) <=2:# 投下
        #    self.wait_counter = int(60 * 2.0)
        #    self.setted_flag = False
        #    return (False, False, True)

        #if (actual_action - indicator_centerx) < 0:#左
        #    return (True, False, False)

        #return (False, True, False)# 右
    

class HybridAI(Controller):
    def __init__(self) -> None:
        super().__init__()
        self.net = MugichaNet((1, 84, 84), 12, 3, 120).float()
        cpt = torch.load("trained_models/mugicha_net_0.chkpt")
        stdict_m = cpt["model"]
        self.net.load_state_dict(stdict_m)

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.net = self.net.to(device="cuda")

    @torch.no_grad()
    def update(self, data_pack) -> Tuple[bool, bool, bool]:
        state, current_poly_index = data_pack
        current_time = pygame.time.get_ticks()  # 現在の時刻を取得
        image_data = state["image"]
        drop_poly_info = torch.tensor(state["drop_poly"], dtype=torch.float32).unsqueeze(0)
        poly_info = torch.tensor(state["poly"], dtype=torch.float32).unsqueeze(0)

        # GPUの場合
        if self.use_cuda:
            # image_data = torch.tensor(state).cuda()
            image_data = torch.tensor(image_data).cuda()
            drop_poly_info = drop_poly_info.cuda()
            poly_info = poly_info.cuda()
        else:
            image_data = torch.tensor(image_data)

        image_data = image_data.unsqueeze(0)
        image_data = image_data.unsqueeze(0)
        action_values, _ = self.net(image_data, drop_poly_info, poly_info, model="online")
        action_idx = torch.argmax(action_values, axis=1).item()

        # 画面上のポリゴンの情報から最適なx座標を計算
        target_x = self.calculate_best_position(poly_info, current_poly_index/11, action_idx)
        
        return target_x
    
    def calculate_best_position(self, poly_info, current_poly_index, action_idx):
        # 誤差の許容範囲を設定
        tolerance = 0.00001
        
        # PyTorch テンソルを Python リストに変換
        poly_info = poly_info.cpu().numpy().tolist()[0]  # 2次元リストの最初の要素を取得

        # 画面上に表示されているポリゴンをフィルタリング
        visible_polygons = [(x, y) for index, x, y in zip(poly_info[0::3], poly_info[1::3], poly_info[2::3]) if y >= 0]

        # 画面上で最も高い位置にある上位3つのポリゴンを選択
        top_3_polygons = sorted(visible_polygons, key=lambda item: item[1])[:5]
        #print(top_3_polygons)
        # 上位3つの中に目的のポリゴンが含まれているか確認
        for x, y in top_3_polygons:
            for i in range(0, len(poly_info), 3):
                if abs(poly_info[i+1] - x) < tolerance and abs(poly_info[i+2] - y) < tolerance:
                    # 一致するポリゴンのインデックスを確認
                    index = poly_info[i]
                    if abs(index - current_poly_index) < tolerance:
                        # 一致するポリゴンが見つかった場合、そのX座標を返す
                        print("Match!")
                        return x * 480  # X座標をスケールアップ

        # 該当するポリゴンが見つからない場合、モデルの出力を使用
        print("AI ACTION!")
        return action_idx * 29 + 66