import gymnasium as gym
import numpy as np
from .env import CoopEnv

class GymCoopEnv(gym.Env):
    def __init__(self):
        super().__init__()

        level_layout = [
            "###S#######",
            "#    #    #",
            "#    #    #",
            "#    #    P",
            "I    #    #",
            "# B  #  A #",
            "#    #    G",
            "##R#####J##",
        ]

        orders = [
             {"meal": "onion-soup", "onions": 1, "tomatoes": 0, "start": 0},
             {"meal": "tomato-soup", "onions": 0, "tomatoes": 1, "start": 1500},
             {"meal": "onion-tomato-soup", "onions": 1, "tomatoes": 1, "start": 3000}
        ]
        
        self.env = CoopEnv(level_layout, orders, reward_mode="shaped")
        
        self.action_space = gym.spaces.MultiDiscrete([6, 6])
        self.observation_space = gym.spaces.Box(low=0, high=12, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        raw_obs = self.env.reset()
        
        obs = self._get_obs(raw_obs)
        
        return obs, {}
    
    def step(self, action):
        a1 = int(action[0])
        a2 = int(action[1])

        raw_obs, reward, done, info = self.env.step(a1, a2)

        obs = self._get_obs(raw_obs)

        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def _get_obs(self, raw_obs):
        agent1_view = raw_obs[0]
        
        x1, y1 = agent1_view["self_pos"]
        x2, y2 = agent1_view["other_pos"]

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def render(self):
        pass