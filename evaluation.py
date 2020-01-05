import torch
import numpy as np

from globals import CONST, config
import mcts
from mcts import MCTS

torch_device = config.evaluation_device


class BoardWrapper:
    def __init__(self, board, white_network_number):
        self.board = board
        self.white_network_number = white_network_number      # number of the white player network


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

