import random
from pygame import *
import os
from collections import deque
from gymnasium.utils import seeding


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.dirname(THIS_DIR)

# Cooking times in steps
COOK_TIME = 200
BURN_TIME = 350

# Function: this will load an image from the assets folder, given a relative path
def load_image(*path_parts):
    full_path = os.path.join(BASE_DIR, "assets", *path_parts)
    img = image.load(full_path).convert_alpha()
    return img

# Function: find the coordinates of a specific character in the level grid
def find_char(grid, target):
    for y, row in enumerate(grid):
        for x, char in enumerate(row):
            if char == target:
                return x, y
    return None

# Function: convert an action index to a movement delta (dx, dy)
def action_to_delta(action):
    if action == 0 or action == 5:
        return (0, 0)
    elif action == 1:
        return (0, -1)
    elif action == 2:
        return (0, 1)
    elif action == 3:
        return (-1, 0)
    elif action == 4:
        return (1, 0)
    else:
        return (0, 0)

class CoopEnv:
    def __init__(self, level, tile_size=60, max_steps=1000, order_time=450, header_size=0, reward_mode="sparse", render=False):
        
        # Initialise the environment with the level layout and parameters
        self.level = level
        self.order_time = order_time
        self.grid_width = len(self.level[0])
        self.grid_height = len(self.level)
        self.env_render = render

        self.reward_mode = reward_mode

        self.tile_size = tile_size
        self.max_steps = max_steps
        self.header_size = header_size

        self.header_bg_color = (30, 30, 45)
        self.header_text_color = (240, 240, 240)

        self.initial_agent1_pos = list(find_char(self.level, "A"))
        self.initial_agent2_pos = list(find_char(self.level, "B"))

        # Initialise the state variables for the environment
        self.active_orders = []
        self.completed_orders = []
        self.failed_orders = []

        self._np_random = None
        self._seed = None

        self.feedback_text = ""
        self.feedback_color = self.header_text_color

        self.wall_items = {}

        # Load sprites if rendering is enabled, otherwise set sprite attributes to None
        if self.env_render:
            self.tile_sprites = {
                " ": load_image("tiles", "floor.png"),
                "#": load_image("tiles", "wall.png"),
                "I": load_image("tiles", "ingredient-box-onion.png"),
                "S": load_image("tiles", "serving-station.png"),
                "J": load_image("tiles", "ingredient-box-tomato.png"),
                "R": load_image("tiles", "bowl-rack.png"),
                "G": load_image("tiles", "garbage.png"),
            }

            self.pot_sprites = {
                "idle":  load_image("tiles", "pot-idle.png"),
                "start":  load_image("tiles", "pot-start.png"),
                "done":  load_image("tiles", "pot-done.png"),
                "burnt": load_image("tiles", "pot-burnt.png"),
            }

            for key, surf in self.pot_sprites.items():
                self.pot_sprites[key] = transform.scale(surf, (self.tile_size, self.tile_size))
                self.pot_sprites[key] = transform.rotate(self.pot_sprites[key], 90)
            
            self.agent1_sprites = {
                ("up",    "empty"): load_image("agents", "up/agent1-up-empty.png"),
                ("up",    "carry"): load_image("agents", "up/agent1-up-carry.png"),
                ("down",  "empty"): load_image("agents", "down/agent1-down-empty.png"),
                ("down",  "bowl"): load_image("agents", "down/agent1-down-bowl.png"),
                ("down",  "onion"): load_image("agents", "down/agent1-down-onion.png"),
                ("down",  "soup"): load_image("agents", "down/agent1-down-soup.png"),
                ("down",  "tomato"): load_image("agents", "down/agent1-down-tomato.png"),
                ("left",  "empty"): load_image("agents", "left/agent1-left-empty.png"),
                ("left",  "bowl"): load_image("agents", "left/agent1-left-bowl.png"),
                ("left",  "onion"): load_image("agents", "left/agent1-left-onion.png"),
                ("left",  "soup"): load_image("agents", "left/agent1-left-soup.png"),
                ("left",  "tomato"): load_image("agents", "left/agent1-left-tomato.png"),
                ("right", "empty"): load_image("agents", "right/agent1-right-empty.png"),
                ("right", "bowl"): load_image("agents", "right/agent1-right-bowl.png"),
                ("right", "onion"): load_image("agents", "right/agent1-right-onion.png"),
                ("right", "soup"): load_image("agents", "right/agent1-right-soup.png"),
                ("right", "tomato"): load_image("agents", "right/agent1-right-tomato.png"),
            }

            self.agent2_sprites = {
                ("up",    "empty"): load_image("agents", "up/agent2-up-empty.png"),
                ("up",    "carry"): load_image("agents", "up/agent2-up-carry.png"),
                ("down",  "empty"): load_image("agents", "down/agent2-down-empty.png"),
                ("down",  "bowl"): load_image("agents", "down/agent2-down-bowl.png"),
                ("down",  "onion"): load_image("agents", "down/agent2-down-onion.png"),
                ("down",  "soup"): load_image("agents", "down/agent2-down-soup.png"),
                ("down",  "tomato"): load_image("agents", "down/agent2-down-tomato.png"),
                ("left",  "empty"): load_image("agents", "left/agent2-left-empty.png"),
                ("left",  "bowl"): load_image("agents", "left/agent2-left-bowl.png"),
                ("left",  "onion"): load_image("agents", "left/agent2-left-onion.png"),
                ("left",  "soup"): load_image("agents", "left/agent2-left-soup.png"),
                ("left",  "tomato"): load_image("agents", "left/agent2-left-tomato.png"),
                ("right", "empty"): load_image("agents", "right/agent2-right-empty.png"),
                ("right", "bowl"): load_image("agents", "right/agent2-right-bowl.png"),
                ("right", "onion"): load_image("agents", "right/agent2-right-onion.png"),
                ("right", "soup"): load_image("agents", "right/agent2-right-soup.png"),
                ("right", "tomato"): load_image("agents", "right/agent2-right-tomato.png"),
            }

            self.item_sprites = {
                "bowl": load_image("items", "bowl-empty.png"),
                "bowl-start": load_image("items", "bowl-start.png"),
                "bowl-done": load_image("items", "bowl-done.png"),
                "bowl-burnt": load_image("items", "bowl-burnt.png"),
                "onion": load_image("items", "onion.png"),
                "tomato": load_image("items", "tomato.png"),
            }

            for key, surf in self.tile_sprites.items():
                self.tile_sprites[key] = transform.scale(surf, (self.tile_size, self.tile_size))
                if key == "P":
                    self.tile_sprites[key] = transform.rotate(self.tile_sprites[key], 90)

            for key, surf in self.agent1_sprites.items():
                self.agent1_sprites[key] = transform.scale(surf, (self.tile_size, self.tile_size))
            
            for key, surf in self.agent2_sprites.items():
                self.agent2_sprites[key] = transform.scale(surf, (self.tile_size, self.tile_size))

            for key, surf in self.item_sprites.items():
                self.item_sprites[key] = transform.scale(surf, (self.tile_size, self.tile_size))
        else:
            self.tile_sprites = None
            self.pot_sprites = None
            self.agent1_sprites = None
            self.agent2_sprites = None
            self.item_sprites = None

        self.reset()

    def reset(self, seed=None):

        # Set the random seed for reproducibility, and initialise the environment state for a new episode
        if seed is not None or self._np_random is None:
            self._np_random, self._seed = seeding.np_random(seed)

        self.step_count = 0
        self.score = 0

        self.agent1_pos = list(self.initial_agent1_pos)
        self.agent2_pos = list(self.initial_agent2_pos)
        self.agent1_dir = (0, -1)
        self.agent2_dir = (0, -1)
        self.agent1_holding = None
        self.agent2_holding = None

        # Pre-calculate connected components
        self._build_components()

        # Positions of key stations for quick access
        self.pot_pos = find_char(self.level, "P")
        self.rack_pos = find_char(self.level, "R")
        self.serve_pos = find_char(self.level, "S")
        self.onion_pos = find_char(self.level, "I")
        self.tomato_pos = find_char(self.level, "J")
        self.garbage_pos = find_char(self.level, "G")

        # Components adjacent to key stations for quick access
        self.pot_side_comps  = self._station_adjacent_comps(self.pot_pos)  if self.pot_pos  else set()
        self.serve_side_comps = self._station_adjacent_comps(self.serve_pos) if self.serve_pos else set()

        # Reset the internal state variables for the episode
        self.stuck_steps = 0
        self.prev_a1 = tuple(self.agent1_pos)
        self.prev_a2 = tuple(self.agent2_pos)

        self.pot_onions = 0
        self.pot_tomatoes = 0
        self.pot_target_onions = None
        self.pot_target_tomatoes = None
        self.pot_recipe = None
        self.pot_timer = 0
        self.pot_state = "idle"
        self.dishes_ready = 0

        self.feedback_text = ""
        self.feedback_color = self.header_text_color
        self.feedback_timer = 0

        self.serving_time = 0
        self.serving_state = "idle"

        # Generate a new set of random orders for the episode, 
        # and clear the active/completed/failed order lists
        self.pending_orders = list(self._random_orders())
        self.active_orders.clear()
        self.completed_orders.clear()
        self.failed_orders.clear()

        # Clear any items on the counters from the previous episode
        self.wall_items.clear()

        return self.get_observation()

    # Function: Get the current observation for both agents,
    # e.g. positions and directions of both agents
    def get_observation(self):
        obs_agent1 = {
            "self_pos": self.agent1_pos,
            "self_dir": self.agent1_dir,
            "other_pos": self.agent2_pos,
            "other_dir": self.agent2_dir
        }

        obs_agent2 = {
            "self_pos": self.agent2_pos,
            "self_dir": self.agent2_dir,
            "other_pos": self.agent1_pos,
            "other_dir": self.agent1_dir
        }

        return (obs_agent1, obs_agent2)

    def step(self, action1, action2):
        # Step penalty to encourage efficiency
        reward = -0.005

        # Check for new orders and update active/pending lists
        new_active = []
        still_pending = []
        for order in self.pending_orders:
            if self.step_count >= order["start"]:
                order_runtime = {
                    "meal": order["meal"],
                    "start": order["start"],
                    "deadline": order["start"] + self.order_time,
                    "onions": order["onions"],
                    "tomatoes": order["tomatoes"],
                    "served": False,
                }
                new_active.append(order_runtime)
            else:
                still_pending.append(order)

        self.pending_orders = still_pending
        self.active_orders.extend(new_active)

        # Check for failed orders and apply a penalty
        still_active = []
        for order in self.active_orders:
            if order["served"]:
                continue
            if self.step_count > order["deadline"]:
                self.failed_orders.append(order)
                reward -= 10.0
            else:
                still_active.append(order)

        self.active_orders = still_active
        
        if self.feedback_text:
            self.feedback_timer += 1
            if self.feedback_timer >= 180:
                self.feedback_text = ""
                self.feedback_timer = 0

        self.step_count += 1

        # Set agent directions based on movement actions
        dx1, dy1 = action_to_delta(action1)
        dx2, dy2 = action_to_delta(action2)

        if action1 in (1, 2, 3, 4):
            self.agent1_dir = (dx1, dy1)
        if action2 in (1, 2, 3, 4):
            self.agent2_dir = (dx2, dy2)

        # Calculate candidate new positions and check for swap attempts
        candidate1 = (self.agent1_pos[0] + dx1, self.agent1_pos[1] + dy1)
        candidate2 = (self.agent2_pos[0] + dx2, self.agent2_pos[1] + dy2)

        a1_pos = tuple(self.agent1_pos)
        a2_pos = tuple(self.agent2_pos)

        swap_attempt = (candidate1 == a2_pos and candidate2 == a1_pos)
        a1_hits_a2  = (candidate1 == a2_pos)
        a2_hits_a1  = (candidate2 == a1_pos)

        # Apply a penalty for attempted collisions to encourage better coordination 
        if a1_hits_a2 or a2_hits_a1:
            reward -= 0.005

        # Handle movement including collision and swap logic
        if swap_attempt:
            self.agent1_pos, self.agent2_pos = list(a2_pos), list(a1_pos)
        else:
            if (0 <= candidate1[0] < self.grid_width and 0 <= candidate1[1] < self.grid_height
                    and not a1_hits_a2):
                if self.level[candidate1[1]][candidate1[0]] in (" ", "A", "B"):
                    self.agent1_pos = list(candidate1)

            a1_pos_after = tuple(self.agent1_pos)

            if (0 <= candidate2[0] < self.grid_width and 0 <= candidate2[1] < self.grid_height
                    and candidate2 != a1_pos_after):
                if self.level[candidate2[1]][candidate2[0]] in (" ", "B", "A"):
                    self.agent2_pos = list(candidate2)
        
        # Handle interaction actions
        if action1 == 5:
            reward += self.handle_interact(agent=1)

        if action2 == 5:
            reward += self.handle_interact(agent=2)
        
        # Handle cooking logic and apply rewards/penalties for pot state and timing
        if self.pot_state in ("start", "done"):
            self.pot_timer += 1

            if self.pot_state == "start" and self.pot_timer >= COOK_TIME:
                self.pot_state = "done"
                if self.reward_mode == "shaped":
                    reward += 1.5

            elif self.pot_state == "done" and self.pot_timer >= BURN_TIME:
                self.pot_state = "burnt"
                if self.reward_mode == "shaped":
                    reward -= 10.0
        
        # To encourage timely pickups, small reward for holding bowl adjacent to cooking pot
        for aid in [1, 2]:
            pos = self.agent1_pos if aid == 1 else self.agent2_pos
            holding = self.agent1_holding if aid == 1 else self.agent2_holding
            
            if holding == "bowl" and self.pot_state == "start":
                if self._is_adjacent(pos, "P"):
                    reward += 0.06
        
        if self.reward_mode == "shaped":
            # Small penalty if not holding bowl when pot is about to be done, to encourage readiness
            if self.pot_state == "start" and self.pot_timer > COOK_TIME - 150:
                a1_ready = (self.agent1_holding == "bowl")
                a2_ready = (self.agent2_holding == "bowl")
                if not (a1_ready or a2_ready):
                    reward -= 0.06

            elif self.pot_state == "done":
                a1_ready = (self.agent1_holding == "bowl")
                a2_ready = (self.agent2_holding == "bowl")

                # Penalty if no one is holding a bowl when the soup is done, to encourage timely pickups
                if not (a1_ready or a2_ready) and not self._has_bowl_accessible_to_pot():
                    reward -= 0.06
                if self.pot_timer > COOK_TIME + 10:
                    reward -= 0.06
                if self.pot_timer > COOK_TIME + 60:
                    reward -= 0.12
                
                for aid in (1,2):
                    if aid == 1:
                        holding = self.agent1_holding
                        pos = self.agent1_pos
                        direction = self.agent1_dir
                    else:
                        holding = self.agent2_holding
                        pos = self.agent2_pos
                        direction = self.agent2_dir

                    tile, _ = self.tile_in_front(pos, direction)

                    # Small reward for holding ingredient in front of the pot, 
                    # to encourage correct pot preparation
                    if holding in ("onion","tomato") and tile == "P": reward += 0.006

                    # Small reward for standing in front of station with empty hands, 
                    # to encourage interaction
                    if holding is None and tile in ("I","J","R"): reward += 0.003

                    # Small reward for holding done soup in front of serving station, 
                    # to encourage serving
                    if isinstance(holding,str) and holding.startswith("bowl-done") and tile == "S": reward += 0.012

                    # Small penalty for holding done soup but not in front of serving station, 
                    # to encourage serving
                    if holding and holding.startswith("bowl-done-"):
                        if self._can_reach_station(aid, self.serve_pos) and not self._is_adjacent(pos, "S"):
                            reward -= 0.01
        
        if self.serving_state.startswith("bowl-"):
            self.serving_time += 1

            if self.serving_time >= 100:
                self.serving_state = "idle"
                self.serving_time = 0
        
        a1_now = tuple(self.agent1_pos)
        a2_now = tuple(self.agent2_pos)

        active_unserved = any(not o["served"] for o in self.active_orders)
        pot_busy = (self.pot_state in ("start", "done"))
        tasks_exist = active_unserved or pot_busy

        waiting_ok = (self.pot_state == "start" and self.pot_timer < COOK_TIME - 30)

        # Penalty for not moving when there are active tasks, 
        # to encourage exploration and prevent stalling
        if tasks_exist and (not waiting_ok) and a1_now == self.prev_a1 and a2_now == self.prev_a2:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        if self.stuck_steps >= 80:
            reward -= 0.05

        self.prev_a1, self.prev_a2 = a1_now, a2_now

        # Check if the episode is done: 
        # either max steps reached or all orders completed/failed
        done = (self.step_count >= self.max_steps) or (
                    len(self.pending_orders) == 0 and all(o["served"] for o in self.active_orders)
                )
        
        # End of episode reward/penalty based on performance, 
        # to encourage perfect completion
        if done and self.reward_mode == "shaped":
            perfect = (self.score == 3 and len(self.failed_orders) == 0)
            reward += 30.0 if perfect else -10.0

        obs = self.get_observation()
        info = {}

        if done:
            info["score"] = self.score
            info["failed_orders"] = len(self.failed_orders)

        return obs, reward, done, info

    def handle_interact(self, agent):
        reward = 0
        
        # Determine the agent's position, direction, 
        # and held item based on the agent ID
        if agent == 1:
            pos = self.agent1_pos
            direction = self.agent1_dir
            holding = self.agent1_holding
        else:
            pos = self.agent2_pos
            direction = self.agent2_dir
            holding = self.agent2_holding

        # Check the tile in front of the agent
        tile, (tx, ty) = self.tile_in_front(pos, direction)

        if tile is None:
            return 0

        # Handle interactions with onion box
        if tile == "I":
            if self.reward_mode == "shaped":

                # Reward/penalty depending on whether the ingredient 
                # is useful or whether the agent is holding anything, 
                # to encourage efficient ingredient pickup
                if holding is None:
                    if self._ingredient_useful("onion", agent):
                        reward += 3.0
                    else:
                        reward -= 1.0
                else:
                    reward -= 0.05
            if holding is None:
                holding = "onion"

        # Handle interactions with tomato box
        elif tile == "J":
            if self.reward_mode == "shaped":

                # Reward/penalty depending on whether the ingredient 
                # is useful or whether the agent is holding anything, 
                # to encourage efficient ingredient pickup
                if holding is None:
                    if self._ingredient_useful("tomato", agent):
                        reward += 3.0
                    else:
                        reward -= 1.0
                else:
                    reward -= 0.05
            if holding is None:
                holding = "tomato"

        # Handle interactions with bowl rack
        elif tile == "R":
            if holding is None:
                if self.reward_mode == "shaped":

                    # Reward for picking up a bowl when the soup is done 
                    # or about to be done, to encourage timely pickups
                    if self.pot_state == "done" and self._need_bowl():
                        print("picked up bowl when done")
                        reward += 10.0
                    elif self.pot_state == "start" and self._need_bowl():
                        if self.pot_timer > COOK_TIME - 150:
                            reward += 10.0
                        else:
                            reward += 5.0
                        print("picked up bowl when undercooked")

                holding = "bowl"
            else:
                # Small penalty for interactions when holding something
                if holding == "bowl":
                    if self.reward_mode == "shaped":
                        reward -= 0.5

        # Handle interactions with cooking pot
        elif tile == "P" and holding is not None:
            if holding in ("onion", "tomato"):

                # Penalty for trying to add ingredients when the pot is already cooking
                if self.pot_state != "idle":
                    if self.reward_mode == "shaped":
                        reward -= 1.0
                    return reward
                
                # Reward/penalty depending on whether the ingredient is useful, 
                # to encourage correct pot preparation
                if self.reward_mode == "shaped":
                    if self._ingredient_useful_for_pot(holding):
                        reward += 8.0
                    else:
                        reward -= 6.0
                        return reward

                # Update pot ingredient counts
                if holding == "onion":
                    self.pot_onions = min(1, self.pot_onions + 1)
                else:
                    self.pot_tomatoes = min(1, self.pot_tomatoes + 1)

                holding = None

                if self.pot_state == "idle" and (self.pot_onions + self.pot_tomatoes) > 0:
                    target = self._get_target_order_for_pot_contents(self.pot_onions, self.pot_tomatoes)

                    # If the current pot contents match an active order, 
                    # set that as the target recipe.
                    if target:
                        self.pot_target_onions = target["onions"]
                        self.pot_target_tomatoes = target["tomatoes"]
                        if self.pot_onions == self.pot_target_onions and self.pot_tomatoes == self.pot_target_tomatoes:
                            self.pot_state = "start"
                            self.pot_timer = 0
                    else:
                        self.pot_onions = 0
                        self.pot_tomatoes = 0
                        self.pot_recipe = None
                        self.pot_target_onions = None
                        self.pot_target_tomatoes = None
                    
                    # If the pot is idle and now has correct ingredients,
                    # start cooking immediately
                    if (self.pot_target_onions is not None and
                        self.pot_onions == self.pot_target_onions and
                        self.pot_tomatoes == self.pot_target_tomatoes):
                        self.pot_state = "start"
                        self.pot_timer = 0
                
                self.pot_recipe = self._counts_to_recipe(self.pot_onions, self.pot_tomatoes)

            elif holding == "bowl" and self.pot_state != "idle":
                if self.pot_state == "start":
                    return 0.0
    
                soup_name = self.pot_recipe or "invalid"
                bowl_state = f"bowl-{self.pot_state}-{soup_name}"

                if self.reward_mode == "shaped":

                    # Reward/penalty depending on whether the soup is correct, 
                    # to encourage timely pickups and correct serving
                    if self.pot_state == "done":
                        soup_onions, soup_tomatoes = self._recipe_to_counts(soup_name)
                        is_wanted = False
                        for order in self.active_orders:
                            if not order["served"]:
                                if (order["onions"] == soup_onions and 
                                    order["tomatoes"] == soup_tomatoes):
                                    is_wanted = True
                                    break
                        
                        if is_wanted:
                            reward += 20.0
                            print(f"pick up correct done soup {soup_name}")
                        else:
                            reward -= 4.0
                            print(f"pick up incorrect done soup {soup_name}")

                    # Penalty for picking up burnt soup, 
                    # to encourage timely pickups and prevent holding burnt soup
                    elif self.pot_state == "burnt":
                        reward -= 4.0
                        print("pick up burnt soup")

                holding = bowl_state
                self.pot_onions = 0
                self.pot_tomatoes = 0
                self.pot_recipe = None
                self.pot_state = "idle"
                self.pot_timer = 0
                self.pot_target_onions = None
                self.pot_target_tomatoes = None

        # Handle interactions with serving station
        elif tile == "S":
            if isinstance(holding, str) and holding.startswith("bowl-"):
                parts = holding.split("-", 2)
                if len(parts) == 3:
                    _, soup_state, soup_recipe = parts
                else:
                    soup_state = "unknown"
                    soup_recipe = "invalid"

                soup_onions, soup_tomatoes = self._recipe_to_counts(soup_recipe)

                if soup_state == "done" and soup_onions is not None:
                    served_correct = False
                    target_deadline = 0

                    for order in self.active_orders:
                        if not order["served"]:
                            if (order["onions"] == soup_onions and
                                order["tomatoes"] == soup_tomatoes):
                                order["served"] = True
                                target_deadline = order["deadline"]
                                self.completed_orders.append(order)
                                served_correct = True
                                break
                    
                    # Reward/penalty depending on whether the served soup matches an active order,
                    # and how quickly it was served, to encourage correct and timely serving
                    if served_correct:
                        print("served correct done order")
                        self.score += 1
                        time_left = target_deadline - self.step_count
                        time_bonus = max(0, time_left * 0.01)
                        reward += 50 + time_bonus
                        self.serving_state = "bowl-done"
                        nice_name = soup_recipe.replace("-", " ")
                        self.feedback_text = f"Correct: {nice_name}!"
                        self.feedback_color = (80, 220, 120)
                    else:
                        print("served incorrect done order")
                        reward -= 80.0
                        self.serving_state = "bowl-done"
                        self.feedback_text = "Wrong order!"
                        self.feedback_color = (220, 80, 80)

                    holding = None
                
                # Penalty for trying to serve burnt soup,
                # to encourage timely serving and prevent holding burnt soup
                else:
                    print("served burnt order")
                    reward -= 20.0

                    if soup_state in ("start", "burnt"):
                        self.serving_state = f"bowl-{soup_state}"
                    else:
                        self.serving_state = "bowl-burnt"

                    self.feedback_text = "Undercooked / burnt / invalid soup!"
                    self.feedback_color = (220, 80, 80)
                    holding = None
        
        # Handle interactions with garbage
        elif tile == "G" and holding is not None:
            if self.reward_mode == "shaped":

                # Weighted reward/penalty depending on whether the agent is discarding something useful,
                # to encourage efficient inventory management
                if holding in ("onion", "tomato"):
                    if self._ingredient_useful(holding, agent):
                        reward -= 15.0
                    else:
                        reward += 0.05
                if holding == "bowl":
                    if self.pot_state == "done":
                        reward -= 15.0
                    elif self.pot_state == "start":
                        reward -= 15.0
                if isinstance(holding, str) and holding.startswith("bowl-"):
                    parts = holding.split("-", 2)
                    if len(parts) == 3:
                        _, soup_state, soup_recipe = parts
                    else:
                        soup_state = "unknown"
                        soup_recipe = "invalid"
                    
                    soup_onions, soup_tomatoes = self._recipe_to_counts(soup_recipe)

                    if soup_state == "done" and soup_onions is not None:
                        correct_soup = False

                        for order in self.active_orders:
                            if not order["served"]:
                                if (order["onions"] == soup_onions and
                                    order["tomatoes"] == soup_tomatoes):
                                    correct_soup = True
                                    break

                        if correct_soup:
                            reward -= 50.0
                        else:
                            reward += 0.05
                    elif soup_state == "burnt":
                        reward += 0.5
                    
            holding = None

        # Handle interactions with counters
        elif tile == "#":
            key = (tx, ty)
            item_here = self.wall_items.get(key)

            # If not holding anything and there's an item on the counter, pick it up
            if holding is None and item_here is not None:
                holding = item_here
                del self.wall_items[key]

            elif holding is not None and item_here is None:
                if self.reward_mode == "shaped":
                    agent_can_serve = self._can_reach_station(agent, self.serve_pos)
                    other = 2 if agent == 1 else 1
                    other_can_serve = self._can_reach_station(other, self.serve_pos)

                    # Small reward/penalty for placing items on counters depending 
                    # on whether they are useful and whether the agent can reach the required station,
                    # to encourage efficient inventory management and collaboration encouragement
                    if isinstance(holding, str) and holding.startswith("bowl-done-"):
                        if agent_can_serve:
                            return reward - 10.0
                        elif other_can_serve:
                            reward += 0.3
                        else:
                            reward -= 1.0

                    if holding in ("onion", "tomato"):
                        if self._can_reach_station(agent, self.pot_pos) and self.pot_state == "idle":
                            return reward - 2.0
                        if (not self._can_reach_station(agent, self.pot_pos)) and self._can_reach_station(other, self.pot_pos):
                            reward += 0.2
                        else:
                            reward -= 0.2

                    if holding == "bowl":
                        if self._can_reach_station(agent, self.pot_pos) and self.pot_state in ("start", "done"):
                            return reward - 1.5

                # Place the item on the counter if it's empty
                self.wall_items[key] = holding
                holding = None
        
        if agent == 1:
            self.agent1_holding = holding
        else:
            self.agent2_holding = holding
        
        return reward

    # Function: Return the tile in front of the agent based on its position and direction
    def tile_in_front(self, pos, direction):
        x, y = pos
        dx, dy = direction
        tx = x + dx
        ty = y + dy

        if 0 <= tx < self.grid_width and 0 <= ty < self.grid_height:
            return self.level[ty][tx], (tx, ty)
        else:
            return None, (tx, ty)

    def render(self, screen):
        if not self.env_render:
            return
        
        # Draw header background with score, orders, and feedback text
        header_rect = (0, 0, self.grid_width * self.tile_size, self.header_size)
        draw.rect(screen, self.header_bg_color, header_rect)

        font1 = font.SysFont("arial", 24, bold=True)
        score_text = font1.render(f"Score: {self.score}", True, self.header_text_color)
        screen.blit(score_text, (10, 13))

        if self.feedback_text:
            font_feedback = font.SysFont("arial", 20)
            feedback_surf = font_feedback.render(self.feedback_text, True, self.feedback_color)
            screen.blit(feedback_surf, (10, 50))

        orders_to_show = [o for o in self.active_orders if not o.get("served")]
        orders_to_show = sorted(orders_to_show, key=lambda o: o["deadline"])

        if orders_to_show:
            font_orders = font.SysFont("arial", 18)
            start_x = 150
            y = 10
            gap = 8

            icon_size = 26
            icon_gap = 4
            onion_icon_small = transform.smoothscale(self.item_sprites["onion"], (icon_size, icon_size))
            tomato_icon_small = transform.smoothscale(self.item_sprites["tomato"], (icon_size, icon_size))

            x = start_x

            for order in orders_to_show[:4]:
                remaining = max(0, order["deadline"] - self.step_count)

                fill_color = (45, 45, 75)
                border_color = (110, 110, 170)
                if remaining < 300:
                    border_color = (200, 90, 90)

                onions = order.get("onions", 0)
                tomatoes = order.get("tomatoes", 0)
                total_icons = onions + tomatoes

                seconds_left = remaining / 60.0
                time_text = f"{seconds_left:.1f}s"
                time_surf = font_orders.render(time_text, True, self.header_text_color)

                icons_width = 0
                if total_icons > 0:
                    icons_width = total_icons * icon_size + (total_icons - 1) * icon_gap

                card_w = 40 + icons_width + time_surf.get_width() + 20
                card_h = 32

                rect = Rect(x, y, card_w, card_h)
                draw.rect(screen, fill_color, rect, border_radius=8)
                draw.rect(screen, border_color, rect, width=2, border_radius=8)

                icon_y = y + (card_h - icon_size) // 2
                icon_x = x + 10

                for _ in range(onions):
                    screen.blit(onion_icon_small, (icon_x, icon_y))
                    icon_x += icon_size + icon_gap

                for _ in range(tomatoes):
                    screen.blit(tomato_icon_small, (icon_x, icon_y))
                    icon_x += icon_size + icon_gap

                time_rect = time_surf.get_rect()
                time_rect.midright = (x + card_w - 10, y + card_h // 2)
                screen.blit(time_surf, time_rect)

                x += card_w + gap

        # Draw the grid with tiles, cooking pot (with timer), 
        # items on counters, serving station state, and agents
        for y, row in enumerate(self.level):
            for x, char in enumerate(row):
                if char == "A" or char == "B":
                    char = " "

                if char == "P":
                    pot_sprite = self.pot_sprites[self.pot_state]
                    screen.blit(pot_sprite, (x * self.tile_size, y * self.tile_size + self.header_size))

                    if self.pot_state != "idle":
                        font1 = font.SysFont("arial", 16, bold=True)

                        if self.pot_state == "start":
                            pot_time = font1.render(f"{(COOK_TIME - self.pot_timer) / 60:.1f}", True, (255, 255, 255))

                        elif self.pot_state == "done":
                            pot_time = font1.render(f"{(BURN_TIME - self.pot_timer) / 60:.1f}", True, (0, 255, 0))

                        else:
                            if (self.step_count // 10) % 2 == 0:
                                color = (255, 0, 0)
                            else:
                                color = (255, 100, 100)
                            pot_time = font1.render("!", True, color)

                        pot_rect = pot_time.get_rect(center=(x * self.tile_size + self.tile_size // 2,
                                                    y * self.tile_size + self.header_size + self.tile_size // 2))
                        screen.blit(pot_time, pot_rect)

                else:
                    sprite = self.tile_sprites[char]
                    screen.blit(sprite, (x * self.tile_size, y * self.tile_size + self.header_size))
        
        for (x, y), item_name in self.wall_items.items():
            if item_name.startswith("bowl-done"):
                item_name = "bowl-done"
            elif item_name.startswith("bowl-start"):
                item_name = "bowl-start"
            elif item_name.startswith("bowl-burnt"):
                item_name = "bowl-burnt"
            sprite = self.item_sprites[item_name]
            screen.blit(sprite, (x * self.tile_size, y * self.tile_size + self.header_size))
        
        if self.serving_state != "idle":
            sprite = self.item_sprites[self.serving_state]
            serving_x, serving_y = find_char(self.level, "S")
            screen.blit(sprite, (serving_x * self.tile_size, serving_y * self.tile_size + self.header_size))

        # Agent 1 sprite selection and rendering
        dir_name = self._dir_to_name(self.agent1_dir)
        carry_name = self._carry_to_name(self.agent1_holding)
        if dir_name == "up" and carry_name != "empty":
            carry_name = "carry"

        sprite = self.agent1_sprites[(dir_name, carry_name)]
        screen.blit(sprite, (self.agent1_pos[0] * self.tile_size,
                            self.agent1_pos[1] * self.tile_size + self.header_size))

        # Agent 2 sprite selection and rendering
        dir_name = self._dir_to_name(self.agent2_dir)
        carry_name = self._carry_to_name(self.agent2_holding)
        if dir_name == "up" and carry_name != "empty":
            carry_name = "carry"
            
        sprite = self.agent2_sprites[(dir_name, carry_name)]
        screen.blit(sprite, (self.agent2_pos[0] * self.tile_size,
                            self.agent2_pos[1] * self.tile_size + self.header_size))
    
    # Function: Convert a direction vector to a string name for sprite selection
    def _dir_to_name(self, direction):
        if direction == (0, -1):
            return "up"
        elif direction == (0, 1):
            return "down"
        elif direction == (-1, 0):
            return "left"
        elif direction == (1, 0):
            return "right"
        else:
            return "down"
    
    # Function: Convert the held item to a string name for sprite selection,
    def _carry_to_name(self, holding):
        if holding is None:
            return "empty"
        elif holding == "bowl":
            return "bowl"
        elif holding == "onion":
            return "onion"
        elif holding == "tomato":
            return "tomato"
        elif holding.startswith("bowl-"):
            return "soup"
        else:
            return "empty"
    
    # Function: Convert ingredient counts to a recipe name
    def _counts_to_recipe(self, onions, tomatoes):
        if onions == 1 and tomatoes == 0:
            return "onion-soup"
        elif onions == 0 and tomatoes == 1:
            return "tomato-soup"
        elif onions == 1 and tomatoes == 1:
            return "onion-tomato-soup"
        else:
            return None

    # Function: Convert a recipe name to ingredient counts
    def _recipe_to_counts(self, recipe):
        if recipe == "onion-soup":
            return 1, 0
        elif recipe == "tomato-soup":
            return 0, 1
        elif recipe == "onion-tomato-soup":
            return 1, 1
        else:
            return None, None
    
    # Function: Determine if picking up a given ingredient 
    # would be useful for completing any active orders,
    # taking into account current inventory, pot contents, 
    # and accessible items on counters
    def _ingredient_useful(self, ingredient, agent_id):
        active_orders = sorted([o for o in self.active_orders if not o["served"]], key=lambda o: o["deadline"])

        target = None
        for order in active_orders:
            recipe = order["meal"]
            if self._count_existing_soups(recipe) > 0:
                continue
            target = order
            break

        if not target:
            return False

        comp = self._agent_comp(agent_id)

        supply_onions = 0
        supply_tomatoes = 0

        supply_onions += self.pot_onions
        supply_tomatoes += self.pot_tomatoes

        for (wx, wy), item in self.wall_items.items():
            if not self._wall_accessible_from_comp((wx, wy), comp):
                continue
            if item == "onion":
                supply_onions += 1
            elif item == "tomato":
                supply_tomatoes += 1

        if agent_id == 1:
            if self.agent1_holding == "onion": supply_onions += 1
            if self.agent1_holding == "tomato": supply_tomatoes += 1
        else:
            if self.agent2_holding == "onion": supply_onions += 1
            if self.agent2_holding == "tomato": supply_tomatoes += 1

        other_id = 2 if agent_id == 1 else 1
        other_comp = self._agent_comp(other_id)
        if other_comp == comp:
            other_hold = self.agent2_holding if agent_id == 1 else self.agent1_holding
            if other_hold == "onion": supply_onions += 1
            if other_hold == "tomato": supply_tomatoes += 1

        needed_onions = target["onions"]
        needed_tomatoes = target["tomatoes"]

        if ingredient == "onion":
            return supply_onions < needed_onions
        if ingredient == "tomato":
            return supply_tomatoes < needed_tomatoes

        return False

    # Function: Determine if adding a given ingredient 
    # to the pot would be useful for completing any active orders,
    # taking into account pot contents and target pot recipe
    def _ingredient_useful_for_pot(self, ingredient):
        if self.pot_target_onions is not None:
            target_onions = self.pot_target_onions
            target_tomatoes = self.pot_target_tomatoes
        else:
            target = self._get_target_order_for_new_pot()
            if not target:
                return False
            target_onions = target["onions"]
            target_tomatoes = target["tomatoes"]

        if self.pot_onions > target_onions or self.pot_tomatoes > target_tomatoes:
            return False

        if ingredient == "onion":
            return self.pot_onions < target_onions
        if ingredient == "tomato":
            return self.pot_tomatoes < target_tomatoes
        return False

    # Function: Determine if there is a bowl accessible to the pot, 
    # either by being held by an agent or on a wall
    def _has_bowl_accessible_to_pot(self):
        if not self.pot_side_comps:
            return False

        a1c = self._agent_comp(1)
        a2c = self._agent_comp(2)

        if self.agent1_holding == "bowl" and a1c in self.pot_side_comps:
            return True
        if self.agent2_holding == "bowl" and a2c in self.pot_side_comps:
            return True

        for (wx, wy), item in self.wall_items.items():
            if item != "bowl":
                continue
            for comp in self.pot_side_comps:
                if self._wall_accessible_from_comp((wx, wy), comp):
                    return True

        return False
    
    # Function: Determine if a bowl is needed for the pot
    def _need_bowl(self):
        if self.pot_state not in ("start", "done"):
            return False

        if self.pot_state == "start" and self.pot_timer < COOK_TIME - 150:
            return False

        return not self._has_bowl_accessible_to_pot()
    
    # Function: Count how many bowls of a recipe are currently present in the environment,
    # either held by agents, on walls, or in the pot if it's done
    def _count_existing_soups(self, recipe_name):
        count = 0
        
        for h in [self.agent1_holding, self.agent2_holding]:
            if isinstance(h, str) and h == f"bowl-done-{recipe_name}":
                count += 1
        
        for item in self.wall_items.values():
            if isinstance(item, str) and item == f"bowl-done-{recipe_name}":
                count += 1
                
        if self.pot_recipe == recipe_name and self.pot_state in ["start", "done"]:
            count += 1
            
        return count
    
    # Function: Get the next target order for a new pot 
    # based on active orders and pending orders
    def _get_target_order_for_new_pot(self):
        active_orders = sorted([o for o in self.active_orders if not o["served"]],
                            key=lambda o: o["deadline"])

        for order in active_orders:
            recipe = order["meal"]
            if self._count_existing_soups(recipe) > 0:
                continue
            return order

        for order in self.pending_orders:
            recipe = order["meal"]
            if self._count_existing_soups(recipe) > 0:
                continue
            return order

        return None
    
    # Function: Generate random orders with random meals and start times,
    # ensuring some spacing between their start times
    def _random_orders(self):
        orders = []
        meals = ["onion-soup", "tomato-soup", "onion-tomato-soup"]

        for _ in range(3):
            meal = meals[int(self._np_random.integers(0, len(meals)))]
            onions, tomatoes = self._recipe_to_counts(meal)
            start = 0
            if _ == 0:
                start = 0
            elif _ == 1:
                start = int(self._np_random.integers(200, 300))
            else:                
                start = int(self._np_random.integers(400, 499))
            orders.append({
                "meal": meal,
                "onions": onions,
                "tomatoes": tomatoes,
                "start": start,
            })
        return orders

    # Function: Check if a tile at (x, y) is walkable for the agents
    def _is_walkable(self, x, y):
        return self.level[y][x] in (" ", "A", "B")

    # Function: Generate neighboring coordinates in all four directions
    def _neighbors4(self, x, y):
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                yield nx, ny
    
    # Function: Build connected components of walkable tiles for pathfinding and accessibility checks
    def _build_components(self):
        self.comp_id = [[-1 for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        cid = 0

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.comp_id[y][x] != -1:
                    continue
                if not self._is_walkable(x, y):
                    continue

                q = deque([(x, y)])
                self.comp_id[y][x] = cid

                while q:
                    cx, cy = q.popleft()
                    for nx, ny in self._neighbors4(cx, cy):
                        if self.comp_id[ny][nx] != -1:
                            continue
                        if not self._is_walkable(nx, ny):
                            continue
                        self.comp_id[ny][nx] = cid
                        q.append((nx, ny))

                cid += 1

    # Function: Get the component ID at a given position
    def _comp_at_pos(self, pos):
        x, y = pos
        return self.comp_id[y][x]

    # Function: Get the component ID of the tile the agent is currently on
    def _agent_comp(self, agent_id):
        pos = self.agent1_pos if agent_id == 1 else self.agent2_pos
        return self._comp_at_pos(pos)

    # Function: Check if a wall tile at wall_pos is accessible from a given component
    def _wall_accessible_from_comp(self, wall_pos, comp):
        wx, wy = wall_pos
        for nx, ny in self._neighbors4(wx, wy):
            if self._is_walkable(nx, ny) and self.comp_id[ny][nx] == comp:
                return True
        return False

    # Function: Get the set of component IDs that are adjacent to a station position
    def _station_adjacent_comps(self, station_pos):
        sx, sy = station_pos
        comps = set()
        for nx, ny in self._neighbors4(sx, sy):
            if self._is_walkable(nx, ny):
                comps.add(self.comp_id[ny][nx])
        return comps

    # Function: Check if an agent can reach a station position 
    def _can_reach_station(self, agent_id, station_pos):
        if station_pos is None:
            return False
        comp = self._agent_comp(agent_id)

        if station_pos == self.pot_pos:
            comps = self.pot_side_comps
        elif station_pos == self.serve_pos:
            comps = self.serve_side_comps
        else:
            comps = self._station_adjacent_comps(station_pos)

        return comp in comps

    # Function: Check if an agent is adjacent to a station position
    def _is_adjacent(self, pos, target):
        if target is None:
            return False
        tx, ty = find_char(self.level, target)
        for nx, ny in self._neighbors4(tx, ty):
            if (nx, ny) == tuple(pos):
                return True
        return False
    
    # Function: Get the best target order for the current pot contents,
    # prioritising active orders, then pending orders
    def _get_target_order_for_pot_contents(self, cur_onions, cur_tomatoes):
        active = sorted(
            [o for o in self.active_orders if not o.get("served", False)],
            key=lambda o: o["deadline"]
        )

        candidates = active if active else list(self.pending_orders)

        def is_available_elsewhere(order):
            return self._count_existing_soups(order["meal"]) > 0

        for order in candidates:
            if is_available_elsewhere(order):
                continue
            if order["onions"] == cur_onions and order["tomatoes"] == cur_tomatoes:
                return order

        best = None
        best_extra = None
        best_time = None

        for order in candidates:
            if is_available_elsewhere(order):
                continue

            if order["onions"] < cur_onions or order["tomatoes"] < cur_tomatoes:
                continue

            extra = (order["onions"] - cur_onions) + (order["tomatoes"] - cur_tomatoes)

            time_key = order.get("deadline", order.get("start", 10**9))

            if best is None or extra < best_extra or (extra == best_extra and time_key < best_time):
                best = order
                best_extra = extra
                best_time = time_key

        return best



