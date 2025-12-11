Environment Info:

Actions: 
0 stay, 1 up, 2 down, 3 left, 4 right, 5 interact

Valid tiles and meaning: 
" " floor, # wall, I onion, J tomato, R bowl rack, P pot, S serving station, G garbage

Meals:
onion-soup 1 onion 0 tomato, tomato-soup 0 onion 1 tomato, onion-tomato-soup 1 onion 1 tomato

Rewards:
Every step -0.01, Correct completed meal +1, Correct undercooked/burnt meal -0.25, Incorrect meal -0.5, Failed order -1

Reset:
When max_steps is reached