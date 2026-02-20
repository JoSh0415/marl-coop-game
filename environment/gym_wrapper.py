import gymnasium as gym
import numpy as np
from .env import CoopEnv, find_char
from .levels import LEVELS
from gymnasium.utils import seeding
from collections import deque

class GymCoopEnv(gym.Env):
    def __init__(self, level_name="level_3", render=False):

        # Initialize the Gym environment and wrap the CoopEnv
        super().__init__()

        level_layout = LEVELS[level_name]
        self.env = CoopEnv(level_layout, render=render)
        self.action_space = gym.spaces.MultiDiscrete([6, 6])
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(74,), dtype=np.float32)

        self._np_random = None
        self._seed = None

    def reset(self, seed=None, options=None):

        # Reset the underlying CoopEnv and return the initial observation and info
        super().reset(seed=seed)

        if seed is not None or self._np_random is None:
            self._np_random, self._seed = seeding.np_random(seed)

        # Generate a new random seed for the episode to ensure variability in training
        episode_seed = int(self._np_random.integers(0, 2**31 - 1))

        raw_obs = self.env.reset(seed=episode_seed)

        # Pre-calculate distance maps to key stations for the new episode
        self.pot_pos = find_char(self.env.level, "P")
        self.rack_pos = find_char(self.env.level, "R")
        self.serve_pos = find_char(self.env.level, "S")
        self.onion_pos = find_char(self.env.level, "I")
        self.tomato_pos = find_char(self.env.level, "J")
        self.garbage_pos = find_char(self.env.level, "G")

        # Precompute BFS distance maps to each station
        self._station_dist_maps = {
            "P": self._bfs_dist_map_to_station(self.pot_pos),
            "R": self._bfs_dist_map_to_station(self.rack_pos),
            "S": self._bfs_dist_map_to_station(self.serve_pos),
            "G": self._bfs_dist_map_to_station(self.garbage_pos),
            "I": self._bfs_dist_map_to_station(self.onion_pos),
            "J": self._bfs_dist_map_to_station(self.tomato_pos),
        }

        # Determine the maximum finite distance for normalisation
        finite_max = 1.0
        for dm in self._station_dist_maps.values():
            finite = dm[np.isfinite(dm)]
            if finite.size:
                finite_max = max(finite_max, float(finite.max()))
        self._max_bfs_dist = finite_max

        obs = self._get_obs(raw_obs)

        info = {"episode_seed": episode_seed, "wrapper_seed": int(self._seed) if self._seed is not None else None}
        return obs, info

    def step(self, action):
        a1 = int(action[0])
        a2 = int(action[1])

        raw_obs, reward, done, info = self.env.step(a1, a2)

        # Convert the raw observation from the CoopEnv into a structured observation
        obs = self._get_obs(raw_obs)

        truncated = self.env.step_count >= self.env.max_steps
        terminated = done and not truncated

        return obs, reward, terminated, truncated, info

    def _hold_onehot(self, item):
        # One-hot encode: [nothing, onion, tomato, bowl, done_soup, burnt_soup]
        vec = [0.0] * 6
        if item is None:
            vec[0] = 1.0
        elif item == "onion":
            vec[1] = 1.0
        elif item == "tomato":
            vec[2] = 1.0
        elif item == "bowl":
            vec[3] = 1.0
        elif isinstance(item, str) and "done" in item:
            vec[4] = 1.0
        elif isinstance(item, str) and "burnt" in item:
            vec[5] = 1.0
        else:
            vec[0] = 1.0
        return vec

    def _pot_state_onehot(self):
        # One-hot encode: [idle, cooking, done, burnt]
        vec = [0.0] * 4
        s = self.env.pot_state
        if s == "start":
            vec[1] = 1.0
        elif s == "done":
            vec[2] = 1.0
        elif s == "burnt":
            vec[3] = 1.0
        else:
            vec[0] = 1.0
        return vec

    def _get_obs(self, raw_obs):
        agent1_view = raw_obs[0]

        # Directions
        dv1, dw1 = agent1_view["self_dir"]
        dv2, dw2 = agent1_view["other_dir"]

        # Holdings (one-hot, 6 each)
        hold1 = self._hold_onehot(self.env.agent1_holding)
        hold2 = self._hold_onehot(self.env.agent2_holding)

        # Front tile features
        def front_features(pos, direction):
            tile, (fx, fy) = self.env.tile_in_front(pos, direction)
            feats = [
                1.0 if tile == "P" else 0.0,
                1.0 if tile == "R" else 0.0,
                1.0 if tile == "S" else 0.0,
                1.0 if tile == "G" else 0.0,
                1.0 if tile == "I" else 0.0,
                1.0 if tile == "J" else 0.0,
                1.0 if tile == "#" else 0.0,
            ]
            has_item = 0.0
            is_handoff = 0.0
            item_type = 0.0
            if tile == "#":
                key = (fx, fy)
                is_handoff = 1.0 if self.env._is_handoff_counter(key) else 0.0
                item = self.env.wall_items.get(key)
                if item is not None:
                    has_item = 1.0
                    if item == "onion":        item_type = 0.2
                    elif item == "tomato":     item_type = 0.4
                    elif item == "bowl":       item_type = 0.6
                    elif isinstance(item, str) and item.startswith("bowl-"): item_type = 0.8
            feats += [has_item, is_handoff, item_type]
            return feats

        front1 = front_features(agent1_view["self_pos"], agent1_view["self_dir"])
        front2 = front_features(agent1_view["other_pos"], agent1_view["other_dir"])

        # BFS distances (24 features)
        x1, y1 = agent1_view["self_pos"]
        x2, y2 = agent1_view["other_pos"]
        a1_pos = (x1, y1)
        a2_pos = (x2, y2)

        p1_d, p1_r = self._dist_and_reach(a1_pos, "P"); p2_d, p2_r = self._dist_and_reach(a2_pos, "P")
        r1_d, r1_r = self._dist_and_reach(a1_pos, "R"); r2_d, r2_r = self._dist_and_reach(a2_pos, "R")
        s1_d, s1_r = self._dist_and_reach(a1_pos, "S"); s2_d, s2_r = self._dist_and_reach(a2_pos, "S")
        g1_d, g1_r = self._dist_and_reach(a1_pos, "G"); g2_d, g2_r = self._dist_and_reach(a2_pos, "G")
        i1_d, i1_r = self._dist_and_reach(a1_pos, "I"); i2_d, i2_r = self._dist_and_reach(a2_pos, "I")
        j1_d, j1_r = self._dist_and_reach(a1_pos, "J"); j2_d, j2_r = self._dist_and_reach(a2_pos, "J")

        bfs_feats = [
            p1_d, p1_r, p2_d, p2_r,
            r1_d, r1_r, r2_d, r2_r,
            s1_d, s1_r, s2_d, s2_r,
            g1_d, g1_r, g2_d, g2_r,
            i1_d, i1_r, i2_d, i2_r,
            j1_d, j1_r, j2_d, j2_r,
        ]

        # Pot state
        pot_oh = self._pot_state_onehot()
        pot_contents = [float(self.env.pot_onions), float(self.env.pot_tomatoes)]
        pot_timer_norm = float(np.clip(self.env.pot_timer / 350.0, 0.0, 1.0))

        # Order info
        target = None
        unserved_active = [o for o in self.env.active_orders if not o.get("served", False)]
        if unserved_active:
            target = min(unserved_active, key=lambda o: o["deadline"])
        elif self.env.pending_orders:
            target = self.env.pending_orders[0]

        order_time_left = 0.0
        target_on = 0.0
        target_to = 0.0
        if target:
            target_on = float(target["onions"])
            target_to = float(target["tomatoes"])
            if "deadline" in target:
                raw_time = target["deadline"] - self.env.step_count
                order_time_left = max(0.0, float(raw_time) / 600.0)
            elif "start" in target:
                raw_time = target["start"] - self.env.step_count
                order_time_left = np.clip(float(raw_time) / 600.0, 0.0, 1.0)
        order_time_left = float(np.clip(order_time_left, 0.0, 1.0))

        # Handoff counter summary
        handoff_onions = 0
        handoff_tomatoes = 0
        handoff_bowls = 0
        handoff_soups = 0

        for (wx, wy), item in self.env.wall_items.items():
            if not self.env._is_handoff_counter((wx, wy)):
                continue
            if item == "onion":
                handoff_onions += 1
            elif item == "tomato":
                handoff_tomatoes += 1
            elif item == "bowl":
                handoff_bowls += 1
            elif isinstance(item, str) and item.startswith("bowl-done-"):
                handoff_soups += 1

        # Build final observation (74 features)
        obs = (
            [(dv1 + 1) / 2.0, (dw1 + 1) / 2.0, (dv2 + 1) / 2.0, (dw2 + 1) / 2.0]  # 4  directions
            + hold1              # 6  agent1 holding
            + hold2              # 6  agent2 holding
            + front1             # 10 agent1 front tile
            + front2             # 10 agent2 front tile
            + bfs_feats          # 24 BFS distances
            + pot_oh             # 4  pot state
            + pot_contents       # 2  pot onions/tomatoes
            + [pot_timer_norm]   # 1  pot timer
            + [order_time_left, target_on, target_to]  # 3  order info
            + [
                float(np.clip(handoff_onions / 5.0, 0.0, 1.0)),
                float(np.clip(handoff_tomatoes / 5.0, 0.0, 1.0)),
                float(np.clip(handoff_bowls / 5.0, 0.0, 1.0)),
                float(np.clip(handoff_soups / 5.0, 0.0, 1.0)),
            ]  # 4 handoff
        )

        return np.array(obs, dtype=np.float32)


    def render(self):
        pass

    # Function: Get neighboring walkable tiles for BFS
    def _neighbors4(self, x, y):
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.env.grid_width and 0 <= ny < self.env.grid_height:
                yield nx, ny

    # Function: BFS to calculate distance maps to key stations for the observation
    def _bfs_dist_map_to_station(self, station_pos):
        H, W = self.env.grid_height, self.env.grid_width
        dist = np.full((H, W), np.inf, dtype=np.float32)

        if station_pos is None:
            return dist

        sx, sy = station_pos
        q = deque()

        for nx, ny in self._neighbors4(sx, sy):
            if self.env._is_walkable(nx, ny):
                dist[ny, nx] = 0.0
                q.append((nx, ny))

        while q:
            x, y = q.popleft()
            base = dist[y, x]
            for nx, ny in self._neighbors4(x, y):
                if not self.env._is_walkable(nx, ny):
                    continue
                if np.isinf(dist[ny, nx]):
                    dist[ny, nx] = base + 1.0
                    q.append((nx, ny))

        return dist

    # Function: Calculate normalised distance and reachability to a station for an agent
    def _dist_and_reach(self, agent_pos, station_key):
        ax, ay = agent_pos
        dist_map = self._station_dist_maps.get(station_key, None)
        if dist_map is None:
            return 1.0, 0.0

        d = dist_map[ay, ax]
        if np.isinf(d):
            return 1.0, 0.0

        d_norm = float(np.clip(d / self._max_bfs_dist, 0.0, 1.0))
        return d_norm, 1.0