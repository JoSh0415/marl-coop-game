class CoopEnv:
    def __init__(self, level, tile_size=60, max_steps=100):
        self.level = level
        self.grid_width = len(self.level[0])
        self.grid_height = len(self.level)

        self.tile_size = tile_size
        self.max_steps = max_steps

        self.initial_agent1_pos = list(self.find_char(self.level, "A"))
        self.initial_agent2_pos = list(self.find_char(self.level, "B"))

        self.order_schedule = []

        self.reset()

    def reset(self):
        self.step_count = 0
        self.score = 0

        self.agent1_pos = list(self.initial_agent1_pos)
        self.agent2_pos = list(self.initial_agent2_pos)
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
        pass

    def render(self, screen):
        pass

    def find_char(grid, target):
        for y, row in enumerate(grid):
            for x, char in enumerate(row):
                if char == target:
                    return x, y
        return None