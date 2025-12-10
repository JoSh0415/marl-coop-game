from pygame import *
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.dirname(THIS_DIR)

COOK_TIME = 1500
BURN_TIME = 2500

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
    def __init__(self, level, orders, tile_size=60, max_steps=1000, order_time=3600, header_size=0):
        self.level = level
        self.orders = orders
        self.order_time = order_time
        self.grid_width = len(self.level[0])
        self.grid_height = len(self.level)

        self.tile_size = tile_size
        self.max_steps = max_steps
        self.header_size = header_size

        self.header_bg_color = (30, 30, 45)
        self.header_text_color = (240, 240, 240)

        self.initial_agent1_pos = list(find_char(self.level, "A"))
        self.initial_agent2_pos = list(find_char(self.level, "B"))

        self.active_orders = []
        self.completed_orders = []
        self.failed_orders = []

        self.feedback_text = ""
        self.feedback_color = self.header_text_color

        self.wall_items = {}

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

        self.reset()

    def reset(self):
        self.step_count = 0
        self.score = 0
        self.reward = 0

        self.agent1_pos = list(self.initial_agent1_pos)
        self.agent2_pos = list(self.initial_agent2_pos)
        self.agent1_dir = (0, -1)
        self.agent2_dir = (0, -1)
        self.agent1_holding = None
        self.agent2_holding = None

        self.pot_ingredients = []
        self.pot_timer = 0
        self.pot_state = "idle"
        self.dishes_ready = 0

        self.feedback_text = ""
        self.feedback_color = self.header_text_color
        self.feedback_timer = 0

        self.serving_time = 0
        self.serving_state = "idle"

        self.pending_orders = list(self.orders)
        self.active_orders.clear()
        self.completed_orders.clear()
        self.failed_orders.clear()

        self.wall_items.clear()

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
        new_active = []
        still_pending = []
        for order in self.pending_orders:
            if self.step_count >= order["start"]:
                order_runtime = {
                    "meal": order["meal"],
                    "start": order["start"],
                    "deadline": order["start"] + self.order_time,
                    "served": False,
                }
                new_active.append(order_runtime)
            else:
                still_pending.append(order)

        self.pending_orders = still_pending
        self.active_orders.extend(new_active)

        still_active = []
        for order in self.active_orders:
            if not order["served"] and self.step_count > order["deadline"]:
                self.failed_orders.append(order)

            else:
                still_active.append(order)

        self.active_orders = still_active

        if self.feedback_text:
            self.feedback_timer += 1
            if self.feedback_timer >= 180:
                self.feedback_text = ""
                self.feedback_timer = 0

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
        
        if self.pot_state in ("start", "done"):
            self.pot_timer += 1

            if self.pot_state == "start" and self.pot_timer >= COOK_TIME:
                self.pot_state = "done"

            elif self.pot_state == "done" and self.pot_timer >= BURN_TIME:
                self.pot_state = "burnt"
        
        if self.serving_state.startswith("bowl-"):
            self.serving_time += 1

            if self.serving_time >= 100:
                self.serving_state = "idle"
                self.serving_time = 0

        self.reward = -0.01

        done = self.step_count >= self.max_steps

        obs = self.get_observation()
        info = {}
        return obs, self.reward, done, info

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

        if tile == "I":
            holding = "onion"

        elif tile == "J":
            holding = "tomato"

        elif tile == "R" and holding is None:
            holding = "bowl"

        elif tile == "P" and holding is not None:
            if holding in ("onion", "tomato"):
                self.pot_ingredients.append(holding)
                holding = None

                if len(self.pot_ingredients) >= 1:
                    self.pot_state = "start"

            elif holding == "bowl" and self.pot_state != "idle":
                bowl_state = f"bowl-{self.pot_state}"
                holding = bowl_state

                self.pot_ingredients.clear()
                self.pot_state = "idle"
                self.pot_timer = 0

            elif holding in ("bowl-start", "bowl-done", "bowl-burnt") and self.pot_state == "idle":
                soup_state = holding.split("-", 1)[1]
                self.pot_state = soup_state

                self.pot_ingredients = ["soup"]
                holding = "bowl"

        elif tile == "S":
            if holding == "bowl-done":
                served_correct = False

                for order in self.active_orders:
                    if not order["served"] and order["meal"] == "onion-soup":
                        order["served"] = True
                        self.completed_orders.append(order)
                        served_correct = True
                        break

                if served_correct:
                    self.score += 1
                    self.reward += 1
                    self.serving_state = "bowl-done"
                    self.feedback_text = "Correct order!"
                    self.feedback_color = (80, 220, 120)
                else:
                    self.reward -= 0.5
                    self.serving_state = "bowl-done"
                    self.feedback_text = "Wrong order!"
                    self.feedback_color = (220, 80, 80)

                holding = None

            elif holding in ("bowl-start", "bowl-burnt"):
                self.reward -= 0.25
                self.score += 0.5
                self.serving_state = holding
                holding = None
                self.feedback_text = "Undercooked / burnt soup!"
                self.feedback_color = (220, 80, 80)

        elif tile == "G" and holding is not None:
            holding = None

        elif tile == "#":
            key = (tx, ty)
            item_here = self.wall_items.get(key)

            if holding is None and item_here is not None:
                holding = item_here
                del self.wall_items[key]

            elif holding is not None and item_here is None:
                self.wall_items[key] = holding
                holding = None
        
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
        header_rect = (0, 0, self.grid_width * self.tile_size, self.header_size)
        draw.rect(screen, self.header_bg_color, header_rect)

        font1 = font.SysFont("arial", 24, bold=True)
        score_text = font1.render(f"Score: {self.score}", True, self.header_text_color)
        screen.blit(score_text, (10, 10))

        if self.feedback_text:
            font_feedback = font.SysFont("arial", 20)
            feedback_surf = font_feedback.render(self.feedback_text, True, self.feedback_color)
            screen.blit(feedback_surf, (10, 50))

        orders_to_show = [o for o in self.active_orders if not o.get("served", False)]
        orders_to_show = sorted(orders_to_show, key=lambda o: o["deadline"])

        if orders_to_show:
            font_orders = font.SysFont("arial", 18)
            start_x = 180          # where the first pill starts
            y = 10                 # vertical position in the header
            card_w = 170
            card_h = 30
            gap = 8

            for i, order in enumerate(orders_to_show[:4]):
                remaining = max(0, order["deadline"] - self.step_count)
                x = start_x + i * (card_w + gap)

                fill_color = (45, 45, 75)
                border_color = (110, 110, 170)

                if remaining < 300:
                    border_color = (200, 90, 90)   # red

                rect = Rect(x, y, card_w, card_h)
                draw.rect(screen, fill_color, rect, border_radius=8)
                draw.rect(screen, border_color, rect, width=2, border_radius=8)

                meal_name = order["meal"].replace("-", " ")
                seconds_left = remaining / 60.0
                text = f"{meal_name} ({seconds_left:.1f}s)"
                text_surf = font_orders.render(text, True, self.header_text_color)
                text_rect = text_surf.get_rect(center=rect.center)
                screen.blit(text_surf, text_rect)

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
                            pot_time = font1.render(f"{COOK_TIME - self.pot_timer}", True, (255, 255, 255))

                        elif self.pot_state == "done":
                            pot_time = font1.render(f"{BURN_TIME - self.pot_timer}", True, (0, 255, 0))

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
            sprite = self.item_sprites[item_name]
            screen.blit(sprite, (x * self.tile_size, y * self.tile_size + self.header_size))
        
        if self.serving_state != "idle":
            sprite = self.item_sprites[self.serving_state]
            serving_x, serving_y = find_char(self.level, "S")
            screen.blit(sprite, (serving_x * self.tile_size, serving_y * self.tile_size + self.header_size))

        # Agent 1
        dir_name = self._dir_to_name(self.agent1_dir)
        carry_name = self._carry_to_name(self.agent1_holding)
        if dir_name == "up" and carry_name != "empty":
            carry_name = "carry"

        sprite = self.agent1_sprites[(dir_name, carry_name)]
        screen.blit(sprite, (self.agent1_pos[0] * self.tile_size,
                            self.agent1_pos[1] * self.tile_size + self.header_size))

        # Agent 2
        dir_name = self._dir_to_name(self.agent2_dir)
        carry_name = self._carry_to_name(self.agent2_holding)
        if dir_name == "up" and carry_name != "empty":
            carry_name = "carry"
            
        sprite = self.agent2_sprites[(dir_name, carry_name)]
        screen.blit(sprite, (self.agent2_pos[0] * self.tile_size,
                            self.agent2_pos[1] * self.tile_size + self.header_size))
    
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