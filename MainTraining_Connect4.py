import globals
from games.connect4 import configuration

globals.init_config(configuration)

import training
from games.connect4 import connect4

if __name__ == '__main__':
    training.main_az(connect4.Connect4Board)
