import logging
import random
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from globals import CONST, config
import mcts
from mcts import MCTS
import data_storage
from utils import utils

torch_device = config.evaluation_device


class BoardWrapper:
    def __init__(self, board, white_network_number):
        self.board = board
        self.white_network_number = white_network_number      # number of the white player network


#@utils.profile
def main_evaluation(game_class, result_folder):
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
    best_net = data_storage.load_net(best_net_path, torch_device)



    generation = []
    prediction_score = []
    for i in range(len(path_list)):
        generation.append(i)
        net_path = network_dir + path_list[i]
        net = data_storage.load_net(net_path, torch_device)
        score = net_vs_net_prediction(net, best_net, game_count, game_class)
        prediction_score.append(score)

        logger.debug("prediction score: {}, network: {}".format(score, net_path))



    # let all network play against the last generation with mcts
    mcts_score = []
    path_list = []      # [path_list[0], path_list[-2]]
    for i in range(len(path_list)):
        net_path = network_dir + path_list[i]
        net = data_storage.load_net(net_path, torch_device)
        score = net_vs_net_mcts(net, best_net, mcts_sim_count, temp, game_count, game_class)
        mcts_score.append(score)

        logger.debug("mcts_score score: {}, network: {}".format(score, net_path))


    # save the results
    np.save(result_folder +"/net_vs_net_pred.npy", np.array(prediction_score))
    np.save(result_folder + "/net_vs_net_mcts.npy", np.array(mcts_score))
    np.save(result_folder + "/net_vs_net_gen.npy", np.array(generation))


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


def net_vs_net_prediction(net1, net2, game_count, game_class):
    """
    plays the two passed network against each other, in half of the games net1 is white and in the other half net2
    is white. to get the policy only the network prediction is used without any mcts simulations. the move to play
    is sampled from the policy distribution. playing deterministically makes no sense as the same games would just be
    repeated over and over again
    :param net1:            network 1
    :param net2:            network 2
    :param game_count:      the number of games to play in total, half of the games are played as white and the other half as black
    :param game_class:      the class of the game
    :return:                score of net1, the score is in the range of 0-1 where:
                            0:   loss
                            0.5: draw
                            1:   win
    """

    half_count = game_count // 2

    board_wrapper_list = []
    for i in range(2*half_count):
        if i < half_count:
            board_wrapper_list.append(BoardWrapper(game_class(), CONST.WHITE))    # net1 is white
        else:
            board_wrapper_list.append(BoardWrapper(game_class(), CONST.BLACK))    # net2 is white


    all_terminated = False
    while not all_terminated:
        batch_list1 = []
        batch_list2 = []
        idx_list1 = []
        idx_list2 = []

        for idx, board_wrapper in enumerate(board_wrapper_list):
            # skip finished games
            if board_wrapper.board.is_terminal():
                continue

            # get the white perspective
            sample, player = board_wrapper.board.white_perspective()
            if player == CONST.WHITE:
                if board_wrapper.white_network_number == 1:
                    batch_list1.append(sample)
                    idx_list1.append(idx)
                else:
                    batch_list2.append(sample)
                    idx_list2.append(idx)

            else:
                if board_wrapper.white_network_number == 1:
                    batch_list2.append(sample)
                    idx_list2.append(idx)
                else:
                    batch_list1.append(sample)
                    idx_list1.append(idx)


        # get the policy form the network
        if len(batch_list1) > 0:
            batch1 = torch.Tensor(batch_list1).to(torch_device)
            policy1, _ = net1(batch1)
            policy1 = policy1.detach().cpu().numpy()
        else:
            policy1 = None


        if len(batch_list2) > 0:
            batch2 = torch.Tensor(batch_list2).to(torch_device)
            policy2, _ = net2(batch2)
            policy2 = policy2.detach().cpu().numpy()
        else:
            policy2 = None


        if policy1 is not None:
            for i_batch in range(policy1.shape[0]):
                # set the illegal moves to 0
                policy = policy1[i_batch]
                idx = idx_list1[i_batch]
                illegal_moves = board_wrapper_list[idx].board.illegal_actions()
                policy[illegal_moves] = 0


                # choose an action according to the probability distribution of all legal actions
                policy /= np.sum(policy)
                action = np.random.choice(len(policy), p=policy)

                # execute the best legal action
                board_wrapper_list[idx].board.execute_action(action)


        if policy2 is not None:
            for i_batch in range(policy2.shape[0]):
                # set the illegal moves to 0
                policy = policy2[i_batch]
                idx = idx_list2[i_batch]
                illegal_moves = board_wrapper_list[idx].board.illegal_actions()
                policy[illegal_moves] = 0

                # choose an action according to the probability distribution of all legal actions
                policy /= np.sum(policy)
                action = np.random.choice(len(policy), p=policy)

                # execute the best legal action
                board_wrapper_list[idx].board.execute_action(action)


        # check if all boards are terminated
        all_terminated = True
        for board_wrapper in board_wrapper_list:
            if not board_wrapper.board.is_terminal():
                all_terminated = False
                break


    # calculate the score of network 1
    score = 0
    for board_wrapper in board_wrapper_list:
        reward = (board_wrapper.board.reward() + 1) / 2
        if board_wrapper.white_network_number == 1:
            score += reward                     # net1 is white
        else:
            score += (1 - reward)               # net1 is black

    score = score / (2*half_count)
    return score


class MCTSContextWrapper:
    def __init__(self, board, player1_color):
        self.board = board
        self.player1_color = player1_color      # color of the player 1 (net1)
        self.mcts_ctx1 = MCTS(board)                # mcts for player 1
        self.mcts_ctx2 = MCTS(board)                # mcts for player 2


    def mcts_info(self):
        """
        returns the information needed for the next mcts simulations
        :return:    the mcts object
                    1 or 2 depending on the network to use
        """
        if self.player1_color == 1:
            if self.board.current_player() == CONST.WHITE:
                return self.mcts_ctx1, 1
            else:
                return self.mcts_ctx2, 2
        else:
            if self.board.current_player() == CONST.WHITE:
                return self.mcts_ctx1, 2
            else:
                return self.mcts_ctx2, 1





def net_vs_net_mcts(net1, net2, mcts_sim_count, temp, game_count, game_class):
    """
    plays the two passed network against each other, in half of the games net1 is white and in the other half net2
    is white. to get the policy mcts is used.
    :param net1:            network 1
    :param net2:            network 2
    :param game_count:      the number of games to play in total, half of the games are played as white and the other half as black
    :param game_class:      the class of the game
    :return:                score of net1, the score is in the range of 0-1 where:
                            0:   loss
                            0.5: draw
                            1:   win
    """

    half_count = game_count // 2

    mcts_ctx_wrapper_list = []
    for i in range(2*half_count):
        if i < half_count:
            mcts_ctx_wrapper_list.append(MCTSContextWrapper(game_class(), 1))    # net1 is white
        else:
            mcts_ctx_wrapper_list.append(MCTSContextWrapper(game_class(), 2))    # net2 is white


    all_terminated = False
    while not all_terminated:
        # prepare the mcts context lists
        mcts_list1 = []         # mcts list where net1 needs to be used
        mcts_list2 = []         # mcts list where net2 needs to be used

        for idx, mcts_ctx_wrapper in enumerate(mcts_ctx_wrapper_list):
            # skip finished games
            if mcts_ctx_wrapper.board.is_terminal():
                continue

            mcts_ctx, net_number = mcts_ctx_wrapper.mcts_info()
            if net_number == 1:
                mcts_list1.append(mcts_ctx)
            else:
                mcts_list2.append(mcts_ctx)

        # run the mcts simulations
        mcts.run_simulations(mcts_list1, mcts_sim_count, net1, 0)
        mcts.run_simulations(mcts_list2, mcts_sim_count, net2, 0)


        # execute the move of the tree search
        for i_mcts_ctx, mcts_ctx in enumerate(mcts_list1):
            # skip terminated games
            if mcts_ctx.board.is_terminal():
                continue

            # choose the action according to the probability distribution
            policy = mcts_ctx.policy_from_state(mcts_ctx.board.state_id(), temp)
            action = np.random.choice(len(policy), p=policy)

            # execute the action on the board
            mcts_ctx.board.execute_action(action)


        for i_mcts_ctx, mcts_ctx in enumerate(mcts_list2):
            # skip terminated games
            if mcts_ctx.board.is_terminal():
                continue

            # choose the action according to the probability distribution
            policy = mcts_ctx.policy_from_state(mcts_ctx.board.state_id(), temp)
            action = np.random.choice(len(policy), p=policy)

            # execute the action on the board
            mcts_ctx.board.execute_action(action)


        # check if all boards are terminated
        all_terminated = True
        for mcts_ctx_wrapper in mcts_ctx_wrapper_list:
            if not mcts_ctx_wrapper.board.is_terminal():
                all_terminated = False
                break


    # calculate the score of network 1
    score = 0
    for mcts_ctx_wrapper in mcts_ctx_wrapper_list:
        reward = (mcts_ctx_wrapper.board.reward() + 1) / 2
        if mcts_ctx_wrapper.player1_color == 1:
            score += reward                 # net1 is white
        else:
            score += (1-reward)             # net1 is white


    score = score / (2*half_count)
    return score

