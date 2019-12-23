import random
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import utils
import data_storage
from globals import CONST
from games.tic_tac_toe import minimax


# The logger
utils.init_logger(logging.DEBUG, file_name="log/connect4.log")
logger = logging.getLogger('evaluation')

np.set_printoptions(suppress=True, precision=6)
plt.style.use('seaborn-dark-palette')             # define the styles for the plot


# set the random seed
random.seed(a=None, version=2)
np.random.seed(seed=None)

network_dir = "../../networks/"                   # directory in which the networks are saved


# create the minimax state dict
minimax.create_state_dict()


# load the network
path_list = os.listdir(network_dir)
path_list.sort(key=utils.natural_keys)


# define the parameters for the evaluation
torch_device = torch.device('cpu')     # torch device that is used for evaluation
game_count = 300                        # the number of games to play
mcts_sim_count = 20                     # the number of mcts simulations


# test the best network to quickly get a result
net_path = network_dir + path_list[-1]
net = data_storage.load_net(net_path, torch_device)
white_score = minimax.play_minimax_games(net, game_count, mcts_sim_count, CONST.WHITE)
black_score = minimax.play_minimax_games(net, game_count, mcts_sim_count, CONST.BLACK)
logger.debug("white score: {}, black: {}, network: {}".format(white_score, black_score, net_path))


# let the different networks play against a minimax player
generation = []
white_scores = []
black_scores = []
path_list = os.listdir(network_dir)
path_list.sort(key=utils.natural_keys)


# get the prediction error of all networks
for i in range(len(path_list)):
    generation.append(i)
    net_path = network_dir + path_list[i]
    net = data_storage.load_net(net_path, torch_device)

    white_score = minimax.play_minimax_games(net, game_count, mcts_sim_count, CONST.WHITE)
    white_scores.append(white_score)
    logger.info("white score vs minimax: {}, network: {}".format(white_score, net_path))

    black_score = minimax.play_minimax_games(net, game_count, mcts_sim_count, CONST.BLACK)
    black_scores.append(black_score)
    logger.info("black score vs minimax: {},  network: {}".format(black_score, net_path))


# save the results
np.save("white_scores.npy", np.array(white_scores))
np.save("black_scores.npy", np.array(black_scores))
np.save("net_generation.npy", np.array(generation))


# plot the white and the black score against the minimax player
fig1 = plt.figure(1)
plt.plot(generation, white_scores, label="white")
plt.plot(generation, black_scores, label="black")
axes = plt.gca()
axes.set_ylim([0, 0.6])
axes = plt.gca()
axes.grid(True, color=(0.9, 0.9, 0.9))
plt.legend()
plt.title("Score vs. Minimax Player")
plt.xlabel("Generation")
plt.ylabel("Score")
fig1.show()


plt.show()
