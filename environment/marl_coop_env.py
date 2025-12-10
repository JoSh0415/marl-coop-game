from pygame import *

from env import CoopEnv

init()

tile_size = 60
header_size = 80

LEVEL1 = [
    "###S#######",
    "#    #    #",
    "#    #    #",
    "#    #    P",
    "I    #    #",
    "# B  #  A #",
    "#    #    G",
    "##R#####J##",
]

ORDERS1 = [
    {"meal": "onion-soup", "onions": 1, "tomatoes": 0, "start": 0}, 
    {"meal": "tomato-soup", "onions": 0, "tomatoes": 1, "start": 1500}, 
    {"meal": "onion-tomato-soup", "onions": 1, "tomatoes": 1, "start": 3000}
]

grid_width = len(LEVEL1[0])
grid_height = len(LEVEL1)
width = grid_width * tile_size
height = grid_height * tile_size + header_size

screen = display.set_mode((width, height))
display.set_caption("MARL Cooperative Environment")

clock = time.Clock()

env = CoopEnv(LEVEL1, ORDERS1, tile_size=tile_size, max_steps=6600, order_time = 3600, header_size=header_size)
obs = env.reset()

running = True

# 0 stay, 1 up, 2 down, 3 left, 4 right, 5 interact
action1 = 0
action2 = 0

while running:
    for e in event.get():
        if e.type == QUIT:
            running = False
        if e.type == KEYDOWN:
            # Agent 1 (arrow keys)
            if e.key == K_UP:
                action1 = 1
            elif e.key == K_DOWN:
                action1 = 2
            elif e.key == K_LEFT:
                action1 = 3
            elif e.key == K_RIGHT:
                action1 = 4
            elif e.key == K_RETURN:
                action1 = 5

            # Agent 2 (WASD)
            if e.key == K_w:
                action2 = 1
            elif e.key == K_s:
                action2 = 2
            elif e.key == K_a:
                action2 = 3
            elif e.key == K_d:
                action2 = 4
            elif e.key == K_SPACE:
                action2 = 5

    obs, reward, done, info = env.step(action1, action2)

    if done:
        obs = env.reset()

    action1 = 0
    action2 = 0

    screen.fill((255, 255, 255))

    env.render(screen)

    display.flip()

    clock.tick(60)
