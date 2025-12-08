from pygame import *

init()

grid_width, grid_height = (11, 8)
tile_size = 60

width = grid_width * tile_size
height = grid_height * tile_size

screen = display.set_mode((width, height))
display.set_caption("MARL Cooperative Environment")

clock = time.Clock()

running = True

LEVEL = [
    "###S#######",
    "#    #    #",
    "#    #    #",
    "#    #    P",
    "I    #    #",
    "# A  #  B #",
    "#    #    #",
    "###########",
]

def find_char(grid, target):
    for y, row in enumerate(grid):
        for x, char in enumerate(row):
            if char == target:
                return x, y
    return None

agent1_pos = list(find_char(LEVEL, "A"))
agent2_pos = list(find_char(LEVEL, "B"))


while running:
    for e in event.get():
        if e.type == QUIT:
            running = False
        if e.type == KEYDOWN:
            # Agent 1 (arrow keys)
            a1_new_pos = None
            if e.key == K_UP:
                a1_new_pos = [agent1_pos[0], agent1_pos[1] - 1]
            elif e.key == K_DOWN:
                a1_new_pos = [agent1_pos[0], agent1_pos[1] + 1]
            elif e.key == K_LEFT:
                a1_new_pos = [agent1_pos[0] - 1, agent1_pos[1]]
            elif e.key == K_RIGHT:
                a1_new_pos = [agent1_pos[0] + 1, agent1_pos[1]]

            if a1_new_pos is not None:
                x, y = a1_new_pos
                if LEVEL[y][x] == " ":
                    agent1_pos = a1_new_pos

            # Agent 2 (WASD)
            a2_new_pos = None
            if e.key == K_w:
                a2_new_pos = [agent2_pos[0], agent2_pos[1] - 1]
            elif e.key == K_s:
                a2_new_pos = [agent2_pos[0], agent2_pos[1] + 1]
            elif e.key == K_a:
                a2_new_pos = [agent2_pos[0] - 1, agent2_pos[1]]
            elif e.key == K_d:
                a2_new_pos = [agent2_pos[0] + 1, agent2_pos[1]]

            if a2_new_pos is not None:
                x, y = a2_new_pos
                if LEVEL[y][x] == " ":
                    agent2_pos = a2_new_pos



    screen.fill((255, 255, 255))

    for y, row in enumerate(LEVEL):
        for x, char in enumerate(row):
            rect = Rect(x * tile_size, y * tile_size, tile_size, tile_size)

            colour = (20, 20, 20)  

            if char == "#":
                colour = (80, 80, 80)  # wall
            elif char == "I":
                colour = (200, 0, 0)   # ingredient
            elif char == "P":
                colour = (0, 0, 200)   # pot
            elif char == "S":
                colour = (0, 150, 0)   # serving station

            draw.rect(screen, colour, rect)

            draw.rect(screen, (50, 50, 50), rect, 1)

    a1_rect = Rect(agent1_pos[0] * tile_size, agent1_pos[1] * tile_size, tile_size, tile_size)
    draw.rect(screen, (0, 0, 255), a1_rect)

    # Agent 2
    a2_rect = Rect(agent2_pos[0] * tile_size, agent2_pos[1] * tile_size, tile_size, tile_size)
    draw.rect(screen, (255, 255, 0), a2_rect)


    display.flip()

    clock.tick(60)
