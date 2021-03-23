import globals
from games.tic_tac_toe import configuration

globals.init_config(configuration)

import evaluation
from games.tic_tac_toe import tic_tac_toe


if __name__ == '__main__':
    evaluation.main_evaluation(tic_tac_toe.TicTacToeBoard, "games/tic_tac_toe/results")
