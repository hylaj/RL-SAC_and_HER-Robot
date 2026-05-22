import gymnasium as gym
import panda_gym
import torch
from asdf.algos import SAC
from asdf.buffers import HerReplayBuffer
from asdf.extractors import DictExtractor
from asdf.policies import MlpPolicy

env_id = "PandaPush-v3"
env = gym.make(env_id)

policy = MlpPolicy(
    env.observation_space,
    env.action_space,
    hidden_sizes=[64, 64],
    extractor_type=DictExtractor,
)

buffer = HerReplayBuffer(env=env, size=1000, device="cpu")

algo = SAC(env, policy=policy, buffer=buffer, alpha=0.05, max_episode_len=100)
algo.load("PandaPush_alfa_auto_future_sampled3_v2.pth") #change the path here if you want to load a different model

env.close()
env = gym.make(env_id, render_mode="human")
results = algo.test(env, n_episodes=20, sleep=1/20)
test_rew = results["mean_ep_ret"]
test_ep_len = results["mean_ep_len"]
env.close()
print(f"Test reward {test_rew}, Test episode length: {test_ep_len}")