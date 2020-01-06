from pathlib import Path

# from games.tic_tac_toe import configuration
from games.connect4 import configuration
# from games.checkers import configuration


class CONST:
	WHITE = 0		# white player
	BLACK = 1 		# black player


# initialize the configuration
config = configuration.Config()

Path(config.save_dir).mkdir(parents=True, exist_ok=True)
