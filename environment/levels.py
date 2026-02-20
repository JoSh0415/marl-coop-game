# Legend:
# ' ' = Empty Floor
# '#' = Counter
# 'A' = Agent 1 Start Position
# 'B' = Agent 2 Start Position
# 'P' = Pot (Cooking Station)
# 'S' = Serving Window
# 'I' = Onion Dispenser
# 'J' = Tomato Dispenser
# 'R' = Bowl Rack
# 'G' = Garbage

LEVELS = {
    # 1. The Bottleneck
    # Simple kitchen layout featuring a horizontal counter with a single gap. 
    # Good for learning basic coordination and timing with constrained movement.
    # The gap acts as a high-conflict zone where both agents must avoid collisions.
    "level_1": [
        "#####S#####",
        "I         J",
        "#         #",
        "# A     B #",
        "##### #####",
        "#         #",
        "#         #",
        "##P##G##R##",
    ],

    # 2. The Partition
    # Separated layout with vertical counter dividing the space.
    # Forces agents to pass items over the counter and coordinate across a barrier.
    # This will test task partitioning as they cannot do each other's jobs.
    "level_2": [
        "###S#######",
        "#    #    #",
        "#    #    #",
        "#    #    P",
        "I    #    #",
        "# B  #  A #",
        "#    #    G",
        "##R#####J##",
    ],

    # 3. The Obstacle Course
    # An open room where the stations themselves (Pot/Rack) act as central pillars.
    # Tests dynamic navigation where the workspace is also the obstacle. 
    # Agents must learn non-linear paths to avoid collisions in the center.
    "level_3": [
        "#####S#####",
        "I         #",
        "#         #",
        "#    P    #",
        "#    #    #",
        "# A  R  B #",
        "#         J",
        "#####G#####",
    ],
}