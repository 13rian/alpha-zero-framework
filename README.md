# AlphaZero for any Board Game



## Goal of this Project 
This project started out by implementing DeepMind's AlphaZero algorithm for Connect Four. I decided to generalize it to use it for other games as well.

## Implement a New Game
In order to implent a new games thw following steps are necessary:
- create a new class that implements all methods defined in game.py.
- create a class with all the configuration values and add a new instance of it globals.py like: config = configuration.Config()
- in the file MainSelfPlayTraining.py pass the class of the game as parameter to the az_main.py function
- run MainSelfPlayTraining.py


An example implementation can be found in games/tic_tac_toe. The game is implemented by the class TicTacToeBoard and the configuration values are defined in configuration.py. The training process can be interrupted at any time and resumed later on as all networks and training examples will be saved after training. If you want to change the game or start from scratch make sure to empty the networks folder and delete the training_data.pkl.