# AlphaZero for any Board Game



## Goal of this Project 
This project started out by implementing DeepMind's AlphaZero algorithm for Connect Four. I decided to generalize it to use it for other games as well.

## Implement a New Game
In order to implent a new games thw following steps are necessary:
- create a new class that implements all methods defined in game.py.
- change the constants BOARD_WIDTH, BOARD_HEIGHT, ACTION_COUNT in globals.CONST. They define the width, the height and the total number of possible actions
- adapt all hyperparameters and configuration value in globals.py
- in the file MainSelfPlayTraining.py pass the class of the game as parameter to the az_main.py function
- run MainSelfPlayTraining.py