# AlphaZero for any Board Game



## Goal of this Project 
This project started out by implementing DeepMind's AlphaZero algorithm for Connect Four. I decided to generalize it to use it for other games as well. I made most of the tries for Connect4. Therefore check out [this](https://github.com/13rian/alpha-zero-connect4) repository for a detailed explanation on how the algorithm works and how it is implemented. This is just a more generalized version with currently 3 implemented games:  
- Tic Tac Toe  
- Connect4  
- Checkers  

## Implement a New Game
In order to implent a new games thw following steps are necessary:
- create a new class that implements all methods defined in game.py.
- create a class with all the configuration values
- create a new MainTraining_....py file with the correct config and board imports.
- Run this file until training is completed
- Create a new Mainealuation_....py file with the correct config imports.
- Run this evaluation file in order to let the different network play against each other.


An example implementation can be found in games/tic_tac_toe. The game is implemented by the class TicTacToeBoard and the configuration values are defined in configuration.py. The training process can be interrupted at any time and resumed later on as all networks and training examples will be saved after training. If you want to change the game or start from scratch make sure to empty the networks folder and delete the training_data.pkl.