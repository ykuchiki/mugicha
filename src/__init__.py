from gym.envs.registration import register
from decopon.controller import AI
import gym

from DQN import MugichaEnv # 消さないで

register(
    id="MugichaEnv",
    entry_point="DQN.MugichaEnv:MugichaEnv",
    kwargs={"controller": AI()}
)

env = gym.make("MugichaEnv")