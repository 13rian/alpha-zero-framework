import globals
from games.checkers import configuration

globals.init_config(configuration)

import evaluation
from games.checkers import checkers


if __name__ == '__main__':
    evaluation.main_evaluation(checkers.CheckersBoard, "games/checkers/results")
