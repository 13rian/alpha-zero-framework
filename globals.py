class CONST:
	WHITE = 0		# white player
	BLACK = 1 		# black player


config = {}


def init_config(configuration):
	global config
	config = configuration.Config()
