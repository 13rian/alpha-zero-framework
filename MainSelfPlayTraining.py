import random
import time
import logging

import matplotlib.pyplot as plt
import numpy as np

from utils import utils
import alpha_zero_learning
import data_storage
from globals import config
import networks
from games.tic_tac_toe import tic_tac_toe
from games.connect4 import connect4
from games.checkers import checkers


#@utils.profile
def main_az(game_class):
    """
    trains an agent using the AlphaZero algorithm
    :param game_class: class of the implemented games
    :return:
    """

    # The logger
    utils.init_logger(logging.DEBUG, file_name="log/connect4.log")
    logger = logging.getLogger('main_training')

    # set the random seed
    random.seed(a=None, version=2)
    np.random.seed(seed=None)

    # create the storage object
    training_data = data_storage.load_data()

    # create the agent
    network = networks.ResNet()
    agent = alpha_zero_learning.Agent(network)

    if training_data.cycle == 0:
        logger.debug("create a new agent")
        training_data.save_data(agent.network)              # save the generation 0 network

        if config.use_initial_data:
            logger.debug("fill the experience buffer with some initial data")
            agent.experience_buffer.fill_with_initial_data()    # add training examples of untrained network

    else:
        # load the current network
        logger.debug("load an old network")
        agent.network = training_data.load_current_net()
        agent.experience_buffer = training_data.experience_buffer

    start_training = time.time()
    for i in range(training_data.cycle, config.cycles, 1):
        ###### self play and update: create some games data through self play
        logger.info("start playing games in cycle {}".format(i))
        avg_moves_played = agent.play_self_play_games(game_class, training_data.network_path)
        training_data.avg_moves_played.append(avg_moves_played)
        logger.debug("average moves played: {}".format(avg_moves_played))


        ###### training, train the training network and use the target network for predictions
        logger.info("start updates in cycle {}".format(i))
        loss_p, loss_v = agent.nn_update(i)
        training_data.policy_loss.append(loss_p)
        training_data.value_loss.append(loss_v)
        logger.debug("policy loss: {}".format(loss_p))
        logger.debug("value loss: {}".format(loss_v))


        ###### save the new network
        logger.info("save check point to file in cycle {}".format(i))
        training_data.cycle += 1
        training_data.experience_buffer = agent.experience_buffer
        training_data.save_data(agent.network)


    end_training = time.time()
    training_time = end_training - start_training
    logger.info("elapsed time whole training process {}".format(training_time))



    # save the results
    np.save("value_loss.npy", np.array(training_data.value_loss))
    np.save("policy_loss.npy", np.array(training_data.policy_loss))
    np.save("avg_moves.npy", np.array(training_data.avg_moves_played))


    # set the style of the plot
    plt.style.use('seaborn-dark-palette')

    # plot the value training loss
    fig1 = plt.figure(1)
    plt.plot(training_data.value_loss)
    axes = plt.gca()
    axes.grid(True, color=(0.9, 0.9, 0.9))
    plt.title("Average Value Training Loss")
    plt.xlabel("Generation")
    plt.ylabel("Value Loss")
    fig1.show()

    # plot the training policy loss
    fig2 = plt.figure(2)
    plt.plot(training_data.policy_loss)
    axes = plt.gca()
    axes.grid(True, color=(0.9, 0.9, 0.9))
    plt.title("Average Policy Training Loss")
    plt.xlabel("Generation")
    plt.ylabel("Policy Loss")
    fig2.show()

    # plot the average number of moves played in the self-play games
    fig3 = plt.figure(3)
    plt.plot(training_data.avg_moves_played)
    axes = plt.gca()
    axes.grid(True, color=(0.9, 0.9, 0.9))
    plt.title("Average Moves in Self-Play Games")
    plt.xlabel("Generation")
    plt.ylabel("Move Count")
    fig3.show()

    plt.show()


if __name__ == '__main__':
    # main_az(tic_tac_toe.TicTacToeBoard)
    # main_az(connect4.Connect4Board)
    main_az(checkers.CheckersBoard)


