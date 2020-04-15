import globals
from games.tic_tac_toe import configuration

globals.init_config(configuration)

import training
from games.tic_tac_toe import tic_tac_toe

if __name__ == '__main__':
    training.main_az(tic_tac_toe.TicTacToeBoard)
