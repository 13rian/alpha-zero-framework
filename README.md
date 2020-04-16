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


## Results
In all matches the following score system was used:  
- 0 game is lost  
- 0.5 game is drawn  
- 1 game is won  

Below the most impressive results of the experiments are shown. [Here](https://github.com/13rian/alpha-zero-connect4) more about the implementation, the used techniques and some detailed result for Connect4 can be read. 

### Tic Tac Toe
In the plot below the scores of the different network generations against a minimax player can be seen. The minimay player always plays the perfect moves. Sometimes there are more than one perfect move therefore to get a better statistics one of the moves were chosen at random. As training progresses the network increases its score against the optimally playing opponent and finally reaches a score of 0.5, which means that the network can draw all games and play perfectly. 

<img src="/results/tic_tac_toe/net_vs_minimax.png" alt="drawing" width="450"/>

The plot below shows matches between the different network generations and the last network generation. As expected the score saturates at 0.5 because the data point in the far right is just a self-play game of the best network. In one case Monte-Carlo tree search was used to make a move. The curve labeled with "net only" was created by choosing a move according to the networks policy of all possible moves. This is way faster and gives a similar result. This approach can be used to quickly check if the network has really learned something.

<img src="/results/tic_tac_toe/net_vs_best_net.png" alt="drawing" width="450"/>


During training the average moves played in a game will increase. The quality of the self-play games is increasing and more optimal games resulting in draws are played. This can clearly be seen in the plot below.

<img src="/results/tic_tac_toe/avg_moves.png" alt="drawing" width="450"/>


### Connect4
The plot below shows the prediction error of the network for random positions. For this test only winning and drawing positions were considered. To get the correct result a Connect4 solver was used. The predicion error saturates around 3% which means that the network would play a non-optimal move every 30 moves in average. 

<img src="/results/connect4/move_prediction_error.png" alt="drawing" width="450"/>


Similar to the plot for Tic Tac Toe the average moves during self-play clearly increases during training as the self-play games increase in quality. 


<img src="/results/connect4/avg_moves.png" alt="drawing" width="450"/>


The training loss and the policy loss are decreasing during training which does not always need to be the case. Generally a downwards trend should be visible though. 


<img src="/results/connect4/policy_loss.png" alt="drawing" width="450"/>

<img src="/results/connect4/value_loss.png" alt="drawing" width="450"/>



### Checkers
Checkers is way more complex than Connect4 and needs a long time to train. I did not find any Checkers solver nor did I program one so far. Below are the results of the different networks playing against the best network. The network is clearly learning something but at the moment I do not know how good it is because of the lack of comparison to an optimal player.  

<img src="/results/checkers/net_vs_best_net.png" alt="drawing" width="450"/>