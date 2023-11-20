from abc import ABC, abstractmethod
from typing import Tuple

import pygame


class Controller(ABC):
    @abstractmethod
    def update(self) -> Tuple[bool, bool, bool]:
        return True, True, True


class Human(Controller):
    def __init__(self) -> None:
        super().__init__()

    def update(self) -> Tuple[bool, bool, bool]:
        pressedKeys = pygame.key.get_pressed()
        return pressedKeys[pygame.K_LEFT], pressedKeys[pygame.K_RIGHT], pressedKeys[pygame.K_SPACE]


class AI(Controller):
    def __init__(self) -> None:
        super().__init__()
        self.auto_action = {"LEFT": False, "RIGHT": False, "SPACE": False}

    def set_auto_action(self, l, r, s):  # l,r,s bool
        self.auto_action["LEFT"] = l
        self.auto_action["RIGHT"] = r
        self.auto_action["SPACE"] = s

    def update(self) -> Tuple[bool, bool, bool]:
        return self.auto_action["LEFT"], self.auto_action["RIGHT"], self.auto_action["SPACE"]