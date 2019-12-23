import torch


class Config:
	# torch devices for training and evaluation
	evaluation_device = torch.device('cpu')			# the pytorch device that is used for evaluation
	training_device = torch.device('cuda') 			# the pytorch device that is used for training

	# board dimension
	board_width = 3 	    				 # the width of the board (number of columns)
	board_height = 3						 # the height of the board
	board_size = board_width * board_height  # the size of the board
	tot_actions = 9 						 # the number of all possible actions (moves)

	# self-play
	cycles = 100				# the number of alpha zero cycles
	episodes = 200 				# the number of games that are self-played in one cycle 2000
	epochs = 2  				# the number of times all training examples are passed through the network 10
	use_initial_data = False  	# true if the generation 0 network data should be used to have a larger experience buffer at start

	# hyperparameters
	mcts_sim_count = 20   		# the number of simulations for the monte-carlo tree search 800
	c_puct = 4 	 				# the higher this constant the more the mcts explores 4
	temp = 1  					# the temperature, controls the policy value distribution
	temp_threshold = 9  		# up to this move the temp will be temp, otherwise 0 (deterministic play)
	alpha_dirich = 1  			# alpha parameter for the dirichlet noise (0.03 - 0.3 az paper, 10/ avg n_moves) 0.3

	# network
	input_channels = 2			# the number of input channels
	filters = 16  				# the number of filters in the conv layers
	blocks = 2  				# number of residual blocks
	value_head_filters = 16   	# number of filters used in the value head
	policy_head_filters = 16  	# number of filters used in the policy head

	# training
	batch_size = 256  			# the batch size of the experience buffer for the neural network training
	cyclic_learning = False  	# true if cyclic learning should be applied
	learning_rate = 10 ** -4  	# the learning rate of the neural network
	min_lr = 10 ** -5  			# minimal learning rate of the cyclical learning process 10**-5.5
	max_lr = 10 ** -4  			# maximal learning rate of the cyclical learning process 10**-3.8
	weight_decay = 1e-4  		# weight decay to prevent overfitting, should be twice as large as L2 regularization const
	average_positions = False  	# true if the positions should be averaged before training
	min_window_size = 4 		# minimal size of the training window (number of cycles for the training data)
	max_window_size = 20   		# maximal size of the training window (number of cycles for the training data)
	initial_game_length = 7     # the initial estimated length of a game to fill the initial training data


	# pytorch dataset configurations
	data_set_params = {
		'batch_size': batch_size,
		'shuffle': True,
		'num_workers': 2,
		'pin_memory': True,
		'drop_last': True,
	}
