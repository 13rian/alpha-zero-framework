import globals
from games.checkers import configuration

globals.init_config(configuration)

import training
from games.checkers import checkers

if __name__ == '__main__':
    training.main_az(checkers.CheckersBoard)
