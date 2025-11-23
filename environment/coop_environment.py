from pygame import *

init()

width = 800
height = 600

screen = display.set_mode((width, height))
display.set_caption("OverCooked")

running = True

state = "game"

grid_size = (13, 8)

while running:
    for e in event.get():
        if e.type == QUIT:
            running = False

    if state == "menu":
        screen.fill((0, 0, 0))
    elif state == "game":
        screen.fill((255, 255, 255))

    display.flip()
