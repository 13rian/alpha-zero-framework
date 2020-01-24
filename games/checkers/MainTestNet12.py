import time
import os

import numpy as np

from utils import utils
from games.checkers import checkers
from globals import config
import data_storage
import evaluation
import mcts
from games.checkers import checkers


# load the best network
network_path = "../../training_data/checkers/networks/network_gen_191.pt"
net = data_storage.load_net(network_path, evaluation.torch_device)


# print the board
board = checkers.CheckersBoard()
board.print()


def network_recommendation(board, network):
    """
    calculates the network policy with mcts
    :return:
    """
    policy = mcts.mcts_policy(board, 200, network, 1, 0)
    return policy


def play_pdn(board, pdn_move):
    print("pdn-move test: ", pdn_move)
    board.play_pdn_move(pdn_move)


def play_net_move(board):
    policy = network_recommendation(board, net)
    action = np.argmax(np.array(policy))
    pdn_move = checkers.action_to_pdn(action, board.player)
    print("pdn-move net: ", pdn_move)
    board.play_pdn_move(pdn_move)



# example games as white
play_net_move(board)
play_pdn(board, "23-18")


play_net_move(board)
play_pdn(board, "24-19")


play_net_move(board)
play_pdn(board, "18-14")


play_net_move(board)
play_pdn(board, "21x14")


play_net_move(board)
play_pdn(board, "22-18")


play_net_move(board)



print(board.is_terminal())
# board.print()