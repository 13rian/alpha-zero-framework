import time
import logging
import pickle
import os

import numpy as np

from globals import CONST
from games.tic_tac_toe import tic_tac_toe
import mcts

logger = logging.getLogger('minimax')

state_dict = {}                         # holds all states of the game, key: state_number, value: white score
state_dict_path = "state_dict.pkl"      # path to safe the minimax dict


def create_state_dict():
    """
    fills the state dict with the white score values
    :return:
    """
    global state_dict

    if len(state_dict) > 0:
        return

    # load the state dict form file if it is already present
    if os.path.exists(state_dict_path):
        with open(state_dict_path, 'rb') as input:
            state_dict = pickle.load(input)

        logger.debug("loaded the state dict from file")
        return

    logger.debug("start to fill the minimax state dict")
    start_time = time.time()
    board = tic_tac_toe.TicTacToeBoard()

    # fill in the first state
    state = board.state_id()

    # go through the whole game
    score = minimax(board, board.player, True)
    state_dict[state] = score
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug("elapsed time to fill the state dict: {}".format(elapsed_time))
    logger.debug("size of the state dict: {}".format(len(state_dict)))

    # dump the state dict to a file
    with open(state_dict_path, 'wb') as output:
        pickle.dump(state_dict, output, pickle.HIGHEST_PROTOCOL)


def minimax(board, player, fill_state_dict = False):
    """
    optimally solves tic tac toe in every board position. this is an implementation of the
    minimax algorithm. it fills the state dict with the seen states
    :param board:       the tic tac toe board
    :param player:      the current player
    :return:            the score of the current player
                         1: win
                         0: draw
                        -1: loss
    """
    if board.terminal:
        reward = board.reward()
        if player == CONST.BLACK:
            reward = -reward
        return reward

    move = -1
    score = -2

    for a in board.legal_actions():
        board_clone = board.clone()
        current_player = board_clone.player
        board_clone.execute_action(a)      # try the move
        move_score = -minimax(board_clone, board_clone.player, fill_state_dict)      # get the score for the opponent

        # fill the state dict
        if fill_state_dict:
            white_score = move_score if current_player == CONST.WHITE else -move_score
            state_number = board_clone.state_id()
            state_dict[state_number] = white_score

        if move_score > score:
            score = move_score
            move = a

    if move == -1:
        return 0

    return score


def play_minimax_games(net, game_count, mcts_sim_count, network_color):
    """
    returns the error percentage of the optimal move prediction by the network
    the network and the mcts are used to predict the move to play
    :param net:                 the network
    :param game_count:          the number of games to play
    :param mcts_sim_count:      the number of monte carlo simulations
    :param network_color:       the color of the network
    :return:                    the score of the network vs the minimax player
    """
    mcts_list = [mcts.MCTS(tic_tac_toe.TicTacToeBoard()) for _ in range(game_count)]
    player = CONST.WHITE

    all_terminated = False
    while not all_terminated:
        # make a move with the az agent
        if player == network_color:
            # run all mcts simulations
            mcts.run_simulations(mcts_list, mcts_sim_count, net, 0)

            # paly the best move suggested by the mcts policy
            for i_mcts_ctx, mcts_ctx in enumerate(mcts_list):
                # skip terminated games
                if mcts_ctx.board.is_terminal():
                    continue

                policy = mcts_list[i_mcts_ctx].policy_from_state(mcts_ctx.board.state_id(), 0)
                move = np.where(policy == 1)[0][0]
                mcts_ctx.board.execute_action(move)

        # make an optimal minimax move
        else:
            for mcts_ctx in mcts_list:
                # skip terminated games
                if mcts_ctx.board.is_terminal():
                    continue

                move = mcts_ctx.board.minimax_move()
                mcts_ctx.board.execute_action(move)

        # swap the player
        player = CONST.WHITE if player == CONST.BLACK else CONST.BLACK

        # check if all games are terminated
        all_terminated = True
        for mcts_ctx in mcts_list:
            if not mcts_ctx.board.is_terminal():
                all_terminated = False
                break


    # extract the score from all boards
    tot_score = 0
    for mcts_ctx in mcts_list:
        score = mcts_ctx.board.white_score() if network_color == CONST.WHITE else mcts_ctx.board.black_score()
        tot_score += score

    tot_score /= game_count
    return tot_score
