from pygame import *
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.dirname(THIS_DIR)

def load_image(*path_parts):
    full_path = os.path.join(BASE_DIR, "assets", *path_parts)
    img = image.load(full_path).convert_alpha()
    return img

def find_char(grid, target):
    for y, row in enumerate(grid):
        for x, char in enumerate(row):
            if char == target:
                return x, y
    return None

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
    def __init__(self, level, tile_size=60, max_steps=1000):
        self.level = level
        self.grid_width = len(self.level[0])
        self.grid_height = len(self.level)

        self.tile_size = tile_size
        self.max_steps = max_steps

        self.initial_agent1_pos = list(find_char(self.level, "A"))
        self.initial_agent2_pos = list(find_char(self.level, "B"))

        self.order_schedule = []

        self.tile_sprites = {
            " ": load_image("tiles", "floor.png"),
            "#": load_image("tiles", "wall.png"),
            "I": load_image("tiles", "ingredient-box-onion.png"),
            "P": load_image("tiles", "pot-idle.png"),
            "S": load_image("tiles", "serving-station.png"),
        }
        
        self.agent1_sprites = {
            ("up",    "empty"): load_image("agents", "agent1-up-empty.png"),
            ("down",  "empty"): load_image("agents", "agent1-down-empty.png"),
            ("left",  "empty"): load_image("agents", "agent1-left-empty.png"),
            ("right", "empty"): load_image("agents", "agent1-right-empty.png"),
        }

        self.agent2_sprites = {
            ("up",    "empty"): load_image("agents", "agent2-up-empty.png"),
            ("down",  "empty"): load_image("agents", "agent2-down-empty.png"),
            ("left",  "empty"): load_image("agents", "agent2-left-empty.png"),
            ("right", "empty"): load_image("agents", "agent2-right-empty.png"),
        }

        for key, surf in self.tile_sprites.items():
            self.tile_sprites[key] = transform.scale(surf, (self.tile_size, self.tile_size))
            if key == "P":
                self.tile_sprites[key] = transform.rotate(self.tile_sprites[key], 90)

        for key, surf in self.agent1_sprites.items():
            self.agent1_sprites[key] = transform.scale(surf, (self.tile_size, self.tile_size))
        
        for key, surf in self.agent2_sprites.items():
            self.agent2_sprites[key] = transform.scale(surf, (self.tile_size, self.tile_size))

        self.reset()

    def reset(self):
        self.step_count = 0
        self.score = 0

        self.agent1_pos = list(self.initial_agent1_pos)
        self.agent2_pos = list(self.initial_agent2_pos)
        self.agent1_dir = (0, -1)
        self.agent2_dir = (0, -1)
        self.agent1_holding = None
        self.agent2_holding = None

        self.pot_ingredients = []
        self.pot_timer = 0
        self.dishes_ready = 0

        self.active_orders = []
        self.completed_orders = []
        self.failed_orders = []

        return self.get_observation()

    def get_observation(self):
        obs_agent1 = {
            "self_pos": self.agent1_pos,
            "other_pos": self.agent2_pos,
        }

        obs_agent2 = {
            "self_pos": self.agent2_pos,
            "other_pos": self.agent1_pos,
        }

        return (obs_agent1, obs_agent2)

    def step(self, action1, action2):
        old_score = self.score

        self.step_count += 1

        dx1, dy1 = action_to_delta(action1)
        dx2, dy2 = action_to_delta(action2)

        if action1 in (1, 2, 3, 4):
            self.agent1_dir = (dx1, dy1)
        if action2 in (1, 2, 3, 4):
            self.agent2_dir = (dx2, dy2)

        candidate1 = (self.agent1_pos[0] + dx1, self.agent1_pos[1] + dy1)
        candidate2 = (self.agent2_pos[0] + dx2, self.agent2_pos[1] + dy2)

        if 0 <= candidate1[0] < self.grid_width and 0 <= candidate1[1] < self.grid_height:
            if self.level[candidate1[1]][candidate1[0]] == " " or self.level[candidate1[1]][candidate1[0]] == "A":
                self.agent1_pos = list(candidate1)

        if 0 <= candidate2[0] < self.grid_width and 0 <= candidate2[1] < self.grid_height:
            if self.level[candidate2[1]][candidate2[0]] == " " or self.level[candidate2[1]][candidate2[0]] == "B":
                self.agent2_pos = list(candidate2)
        
        if action1 == 5:
            self.handle_interact(agent=1)

        if action2 == 5:
            self.handle_interact(agent=2)

        reward = -0.01
        reward += (self.score - old_score) * 1.0

        done = self.step_count >= self.max_steps

        obs = self.get_observation()
        info = {}
        return obs, reward, done, info

    def handle_interact(self, agent):
        if agent == 1:
            pos = self.agent1_pos
            direction = self.agent1_dir
            holding = self.agent1_holding
        else:
            pos = self.agent2_pos
            direction = self.agent2_dir
            holding = self.agent2_holding

        tile, (tx, ty) = self.tile_in_front(pos, direction)

        if tile is None:
            return 

        if tile == "I" and holding is None:
            holding = "ingredient"

        elif tile == "P" and holding == "ingredient":
            self.pot_ingredients.append("ingredient")
            holding = None

        elif tile == "S" and len(self.pot_ingredients) > 0:
            self.pot_ingredients.clear()
            self.score += 1

        if agent == 1:
            self.agent1_holding = holding
        else:
            self.agent2_holding = holding

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
        for y, row in enumerate(self.level):
            for x, char in enumerate(row):
                if char == "#":
                    sprite = self.tile_sprites.get(char, self.tile_sprites["#"])
                    screen.blit(sprite, (x * self.tile_size, y * self.tile_size))
                elif char == "I":
                    sprite = self.tile_sprites.get(char, self.tile_sprites["I"])
                    screen.blit(sprite, (x * self.tile_size, y * self.tile_size))
                elif char == "P":
                    sprite = self.tile_sprites.get(char, self.tile_sprites["P"])
                    screen.blit(sprite, (x * self.tile_size, y * self.tile_size))
                elif char == "S":
                    sprite = self.tile_sprites.get(char, self.tile_sprites["S"])
                    screen.blit(sprite, (x * self.tile_size, y * self.tile_size))
                elif char == " " or char == "A" or char == "B":
                    sprite = self.tile_sprites.get(char, self.tile_sprites[" "])
                    screen.blit(sprite, (x * self.tile_size, y * self.tile_size))

        # Agent 1
        dir_name = self._dir_to_name(self.agent1_dir)
        carry_name = self._carry_to_name(self.agent1_holding)
        sprite = self.agent1_sprites[(dir_name, carry_name)]
        screen.blit(sprite, (self.agent1_pos[0] * self.tile_size,
                            self.agent1_pos[1] * self.tile_size))

        # Agent 2
        dir_name = self._dir_to_name(self.agent2_dir)
        carry_name = self._carry_to_name(self.agent2_holding)
        sprite = self.agent2_sprites[(dir_name, carry_name)]
        screen.blit(sprite, (self.agent2_pos[0] * self.tile_size,
                            self.agent2_pos[1] * self.tile_size))
    
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
    
    def _carry_to_name(self, holding):
        if holding is None:
            return "empty"
        else:
            return "ingredient"