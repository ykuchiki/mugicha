from gym.envs.registration import register
from decopon.controller import AI
import gym

from DQN import environment # 消さないで

register(
    id="decoponEnv",
    entry_point="DQN.environment:MugichaEnv",
    kwargs={"controller": AI()}
)

env = gym.make("decoponEnv")