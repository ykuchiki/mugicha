import src
import gym
from gym.wrappers import FrameStack

from DQN.utils import SkipFrame

env = gym.make("decoponEnv")
print(env.action_space.n)   # 3
print(env.observation_space.shape[0])  # 84

env = SkipFrame(env, skip=4)
env = FrameStack(env, num_stack=4)
next_state, reward, done, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")