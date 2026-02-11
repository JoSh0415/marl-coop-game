import gymnasium as gym
import numpy as np
from .env import CoopEnv, find_char
from .levels import LEVELS
from gymnasium.utils import seeding
from collections import deque

# Function: Relative position helper for normalising distances
def rel(ax, ay, bx, by, w, h):
    return (bx - ax) / w, (by - ay) / h

class GymCoopEnv(gym.Env):
    def __init__(self, level_name="level_1", render=False):

        # Initialize the Gym environment and wrap the CoopEnv
        super().__init__()

        level_layout = LEVELS[level_name]
        self.env = CoopEnv(level_layout, reward_mode="shaped", render=render)
        self.action_space = gym.spaces.MultiDiscrete([6, 6])
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(83,), dtype=np.float32)

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

        # Determine the maximum finite distance for normalisation
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

        # Convert the raw observation from the CoopEnv into a structured observation
        obs = self._get_obs(raw_obs)

        truncated = self.env.step_count >= self.env.max_steps
        terminated = done and not truncated

        return obs, reward, terminated, truncated, info

    def _get_obs(self, raw_obs):
        agent1_view = raw_obs[0]
        
        w = float(self.env.grid_width)
        h = float(self.env.grid_height)

        # Extract positions, directions, and holding states for both agents
        x1, y1 = agent1_view["self_pos"]
        dv1, dw1 = agent1_view["self_dir"]
        x2, y2 = agent1_view["other_pos"]
        dv2, dw2 = agent1_view["other_dir"]

        needed_onions = 0.0
        needed_tomatoes = 0.0
        order_time_left = 0.0

        target = None
        unserved_active = [o for o in self.env.active_orders if not o.get("served", False)]

        if unserved_active:
            target = min(unserved_active, key=lambda o: o["deadline"])
        elif self.env.pending_orders:
            target = self.env.pending_orders[0]

        # Calculate time left for the current target order
        order_time_left = 0.0
        if target:
            if "deadline" in target:
                raw_time = target["deadline"] - self.env.step_count
                order_time_left = max(0.0, float(raw_time) / 600.0)
            elif "start" in target:
                raw_time = target["start"] - self.env.step_count
                order_time_left = np.clip(float(raw_time) / 600.0, 0.0, 1.0)

        # Determine if onions or tomatoes are needed 
        # for the current target order or pot target
        if self.env.pot_target_onions is not None:
            if self.env.pot_target_onions > self.env.pot_onions:
                needed_onions = 1.0
            if self.env.pot_target_tomatoes > self.env.pot_tomatoes:
                needed_tomatoes = 1.0
        else:
            if target:
                if target["onions"] > self.env.pot_onions:
                    needed_onions = 1.0
                if target["tomatoes"] > self.env.pot_tomatoes:
                    needed_tomatoes = 1.0
        
        pot_target_onions   = -1.0 if self.env.pot_target_onions   is None else float(self.env.pot_target_onions)
        pot_target_tomatoes = -1.0 if self.env.pot_target_tomatoes is None else float(self.env.pot_target_tomatoes)

        agent1_holding = 0.0
        agent2_holding = 0.0

        # Encode the holding state of the first agent into a 0.0-1.0 value
        if self.env.agent1_holding == None:
            agent1_holding = 0.0
        elif self.env.agent1_holding == "onion":
            agent1_holding = 0.2
        elif self.env.agent1_holding == "tomato":
            agent1_holding = 0.4
        elif self.env.agent1_holding == "bowl":
            agent1_holding = 0.6
        elif self.env.agent1_holding == "bowl-burnt":
            agent1_holding = 0.5
        elif self.env.agent1_holding.startswith("bowl-"):
                parts = self.env.agent1_holding.split("-", 2)
                if len(parts) == 3:
                    _, soup_state, soup_recipe = parts
                else:
                    soup_state = "unknown"
                    soup_recipe = "invalid"

                soup_onions, soup_tomatoes = self.env._recipe_to_counts(soup_recipe)

                if soup_state == "done" and soup_onions is not None:
                    bowl_correct = False

                    for order in self.env.active_orders:
                        if not order["served"]:
                            if (order["onions"] == soup_onions and order["tomatoes"] == soup_tomatoes):
                                bowl_correct = True
                                break

                    if bowl_correct:
                        agent1_holding = 1.0
                    else:
                        agent1_holding = 0.8
                elif soup_state == "burnt":
                    agent1_holding = 0.5
                elif soup_state == 'start':
                    agent1_holding = 0.7
        else:
            agent1_holding = 0.0

        # Determine the type of tile in front of each agent and encode it for the observation
        front_tile_1, _ = self.env.tile_in_front(agent1_view["self_pos"], agent1_view["self_dir"])
        front_is_pot_1  = 1.0 if front_tile_1 == "P" else 0.0
        front_is_rack_1 = 1.0 if front_tile_1 == "R" else 0.0
        front_is_serve_1= 1.0 if front_tile_1 == "S" else 0.0
        front_is_garbage_1 = 1.0 if front_tile_1 == "G" else 0.0
        front_is_onion_1 = 1.0 if front_tile_1 == "I" else 0.0
        front_is_tomato_1 = 1.0 if front_tile_1 == "J" else 0.0

        front_tile_2, _ = self.env.tile_in_front(agent1_view["other_pos"], agent1_view["other_dir"])
        front_is_pot_2  = 1.0 if front_tile_2 == "P" else 0.0
        front_is_rack_2 = 1.0 if front_tile_2 == "R" else 0.0
        front_is_serve_2= 1.0 if front_tile_2 == "S" else 0.0
        front_is_garbage_2 = 1.0 if front_tile_2 == "G" else 0.0
        front_is_onion_2 = 1.0 if front_tile_2 == "I" else 0.0
        front_is_tomato_2 = 1.0 if front_tile_2 == "J" else 0.0

        # Encode the holding state of the second agent into a 0.0-1.0 value
        if self.env.agent2_holding == None:
            agent2_holding = 0.0
        elif self.env.agent2_holding == "onion":
            agent2_holding = 0.2
        elif self.env.agent2_holding == "tomato":
            agent2_holding = 0.4
        elif self.env.agent2_holding == "bowl":
            agent2_holding = 0.6
        elif self.env.agent2_holding == "bowl-burnt":
            agent2_holding = 0.5
        elif self.env.agent2_holding.startswith("bowl-"):
                parts = self.env.agent2_holding.split("-", 2)
                if len(parts) == 3:
                    _, soup_state, soup_recipe = parts
                else:
                    soup_state = "unknown"
                    soup_recipe = "invalid"

                soup_onions, soup_tomatoes = self.env._recipe_to_counts(soup_recipe)

                if soup_state == "done" and soup_onions is not None:
                    served_correct = False

                    for order in self.env.active_orders:
                        if not order["served"]:
                            if (order["onions"] == soup_onions and order["tomatoes"] == soup_tomatoes):
                                served_correct = True
                                break

                    if served_correct:
                        agent2_holding = 1.0
                    else:
                        agent2_holding = 0.8
                elif soup_state == "burnt":
                    agent2_holding = 0.5
                elif soup_state == 'start':
                    agent2_holding = 0.7
        else:
            agent2_holding = 0.0
        

        pot_state = 0.0

        # Encode the state of the pot (idle, cooking, done, burnt) into a 0.0-1.0 value
        if self.env.pot_state == "start":
            pot_state = 0.33
        elif self.env.pot_state == "done":
            pot_state = 0.66
        elif self.env.pot_state == "burnt":
            pot_state = 1.0
        else:
            pot_state = 0.0
        
        order_time_left = np.clip(order_time_left, 0.0, 1.0)
        pot_timer_norm  = np.clip(self.env.pot_timer / 350.0, 0.0, 1.0)

        px, py = self.pot_pos
        rx, ry = self.rack_pos
        sx, sy = self.serve_pos
        gx, gy = self.garbage_pos
        ox, oy = self.onion_pos
        tx, ty = self.tomato_pos

        # Calculate relative positions to key stations for both agents
        dxp1,dyp1 = rel(x1,y1, px,py,w,h)
        dxp2,dyp2 = rel(x2,y2, px,py,w,h)
        dxr1,dyr1 = rel(x1,y1, rx,ry,w,h)
        dxr2,dyr2 = rel(x2,y2, rx,ry,w,h)
        dxs1,dys1 = rel(x1,y1, sx,sy,w,h)
        dxs2,dys2 = rel(x2,y2, sx,sy,w,h)
        dxg1,dyg1 = rel(x1,y1, gx,gy,w,h)
        dxg2,dyg2 = rel(x2,y2, gx,gy,w,h)
        dxo1,dyo1 = rel(x1,y1, ox,oy,w,h)
        dxo2,dyo2 = rel(x2,y2, ox,oy,w,h)
        dxt1,dyt1 = rel(x1,y1, tx,ty,w,h)
        dxt2,dyt2 = rel(x2,y2, tx,ty,w,h)

        a1_pos = (x1, y1)
        a2_pos = (x2, y2)

        # Calculate BFS distance and reachability of key stations for both agents
        p1_d, p1_r = self._dist_and_reach(a1_pos, "P"); p2_d, p2_r = self._dist_and_reach(a2_pos, "P")
        r1_d, r1_r = self._dist_and_reach(a1_pos, "R"); r2_d, r2_r = self._dist_and_reach(a2_pos, "R")
        s1_d, s1_r = self._dist_and_reach(a1_pos, "S"); s2_d, s2_r = self._dist_and_reach(a2_pos, "S")
        g1_d, g1_r = self._dist_and_reach(a1_pos, "G"); g2_d, g2_r = self._dist_and_reach(a2_pos, "G")
        i1_d, i1_r = self._dist_and_reach(a1_pos, "I"); i2_d, i2_r = self._dist_and_reach(a2_pos, "I")
        j1_d, j1_r = self._dist_and_reach(a1_pos, "J"); j2_d, j2_r = self._dist_and_reach(a2_pos, "J")

        # Compile BFS features into a list for the observation
        bfs_feats = [
            p1_d, p1_r, p2_d, p2_r,
            r1_d, r1_r, r2_d, r2_r,
            s1_d, s1_r, s2_d, s2_r,
            g1_d, g1_r, g2_d, g2_r,
            i1_d, i1_r, i2_d, i2_r,
            j1_d, j1_r, j2_d, j2_r,
        ]

        # Determine whether a counter is in front of each agent
        front_tile_1, _ = self.env.tile_in_front(agent1_view["self_pos"], agent1_view["self_dir"])
        front_is_counter_1 = 1.0 if front_tile_1 == "#" else 0.0
        
        front_tile_2, _ = self.env.tile_in_front(agent1_view["other_pos"], agent1_view["other_dir"])
        front_is_counter_2 = 1.0 if front_tile_2 == "#" else 0.0

        target_on = 0.0
        target_to = 0.0
        active = [o for o in self.env.active_orders if not o.get("served", False)]

        # Determine the target onion and tomato counts for the oldest active order
        if active:
            best = min(active, key=lambda o: o["deadline"])
            target_on = float(best["onions"])
            target_to = float(best["tomatoes"])
        elif self.env.pending_orders:
            best = min(self.env.pending_orders, key=lambda o: o["start"])
            target_on = float(best["onions"])
            target_to = float(best["tomatoes"])

        # Compile all the features into a single observation array for the agents
        return np.array([
            x1 / w, 
            y1 / h, 
            (dv1 + 1) / 2.0, 
            (dw1 + 1) / 2.0, 
            x2 / w, 
            y2 / h, 
            (dv2 + 1) / 2.0, 
            (dw2 + 1) / 2.0,
            dxp1,
            dyp1,
            dxr1,
            dyr1,
            dxs1,
            dys1,
            dxg1,
            dyg1,
            dxo1,
            dyo1,
            dxt1,
            dyt1,
            dxp2,
            dyp2,
            dxr2,
            dyr2,
            dxs2,
            dys2,
            dxg2,
            dyg2,
            dxo2,
            dyo2,
            dxt2,
            dyt2,
            front_is_pot_1,
            front_is_rack_1,
            front_is_serve_1,
            front_is_garbage_1,
            front_is_onion_1,
            front_is_tomato_1,
            front_is_pot_2,
            front_is_rack_2,
            front_is_serve_2,
            front_is_garbage_2,
            front_is_onion_2,
            front_is_tomato_2,
            front_is_counter_1,
            front_is_counter_2,
            needed_onions, 
            needed_tomatoes,
            pot_target_onions,
            pot_target_tomatoes,
            order_time_left, 
            agent1_holding, 
            agent2_holding, 
            self.env.pot_onions,
            self.env.pot_tomatoes, 
            pot_timer_norm,
            pot_state,
            *bfs_feats,
            target_on,
            target_to,
        ], dtype=np.float32)

    def render(self):
        pass
    
    # Function: Get neighboring walkable tiles for BFS
    def _neighbors4(self, x, y):
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.env.grid_width and 0 <= ny < self.env.grid_height:
                yield nx, ny

    # Function: BFS to calculate distance maps to key stations for the observation
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