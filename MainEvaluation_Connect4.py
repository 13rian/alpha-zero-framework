import globals
from games.connect4 import configuration

globals.init_config(configuration)

import evaluation
from games.connect4 import connect4


if __name__ == '__main__':
    evaluation.main_evaluation(connect4.Connect4Board)
