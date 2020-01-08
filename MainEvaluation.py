import random
import os
import logging

import matplotlib.pyplot as plt
import numpy as np

from utils import utils
import evaluation
import data_storage
from globals import config
from games.tic_tac_toe import tic_tac_toe
from games.connect4 import connect4
from games.checkers import checkers


#@utils.profile
def main_evaluation(game_class):
    # configuration values
    game_count = 200        # the number of test games to play
    mcts_sim_count = 200     # the number of mcts simulations to perform
    temp = 0.3              # the temperature used to get the policy for the move selection, gives some randomness


    # the logger
    utils.init_logger(logging.DEBUG, file_name="log/app.log")
    logger = logging.getLogger('evaluation')

    # set the random seed
    random.seed(a=None, version=2)
    np.random.seed(seed=None)


    # load the network
    network_dir = config.save_dir + "/networks/"
    path_list = os.listdir(network_dir)
    path_list.sort(key=utils.natural_keys)


    # let all network play against the last generation without any mcts
    best_net_path = network_dir + path_list[-1]
    best_net = data_storage.load_net(best_net_path, evaluation.torch_device)



    generation = []
    prediction_score = []
    for i in range(len(path_list)):
        generation.append(i)
        net_path = network_dir + path_list[i]
        net = data_storage.load_net(net_path, evaluation.torch_device)
        score = evaluation.net_vs_net_prediction(net, best_net, game_count, game_class)
        prediction_score.append(score)

        logger.debug("prediction score: {}, network: {}".format(score, net_path))



    # let all network play against the last generation with mcts
    mcts_score = []
    path_list = [] # [path_list[0], path_list[-2]]
    for i in range(len(path_list)):
        net_path = network_dir + path_list[i]
        net = data_storage.load_net(net_path, evaluation.torch_device)
        score = evaluation.net_vs_net_mcts(net, best_net, mcts_sim_count, temp, game_count, game_class)
        mcts_score.append(score)

        logger.debug("mcts_score score: {}, network: {}".format(score, net_path))


    # save the results
    np.save("net_vs_net_pred.npy", np.array(prediction_score))
    np.save("net_vs_net_mcts.npy", np.array(mcts_score))
    np.save("net_vs_net_gen.npy", np.array(generation))


    # set the style of the plot
    plt.style.use('seaborn-dark-palette')


    # plot the prediction score
    fig1 = plt.figure(1)
    plt.plot(generation, prediction_score)
    axes = plt.gca()
    axes.set_ylim([0, 0.55])
    axes.grid(True, color=(0.9, 0.9, 0.9))
    plt.title("Prediction Score vs Best Network")
    plt.xlabel("Generation")
    plt.ylabel("Prediction Score")
    fig1.show()


    # # plot the mcts score
    # fig2 = plt.figure(2)
    # plt.plot(generation, mcts_score)
    # axes = plt.gca()
    # axes.set_ylim([0, 0.55])
    # axes.grid(True, color=(0.9, 0.9, 0.9))
    # plt.title("MCTS Prediction Score vs Best Network")
    # plt.xlabel("Generation")
    # plt.ylabel("MCTS Score")
    # fig2.show()

    plt.show()


if __name__ == '__main__':
    # main_evaluation(tic_tac_toe.TicTacToeBoard)
    # main_evaluation(connect4.Connect4Board)
    main_evaluation(checkers.CheckersBoard)


