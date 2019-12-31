import logging
import pickle
from operator import itemgetter

import torch
from torch.utils import data
import numpy as np

from globals import CONST, config
import mcts
from mcts import MCTS
import data_storage


logger = logging.getLogger('az_learning')


class Agent:
    def __init__(self, network):
        """
        :param network:         alpha zero network that is used for training and evaluation
        """
        self.network = network                                  # the network
        self.experience_buffer = ExperienceBuffer()             # buffer that saves all experiences

        # activate the evaluation mode of the networks
        self.network = data_storage.net_to_device(self.network, config.evaluation_device)
        self.network.eval()


    def play_self_play_games(self, game_class, network_path):
        """
        plays some games against itself and adds the experience into the experience buffer
        :param game_class       the class of the implemented games
        :param network_path     the path of the current network
        :return:                the average nu,ber of moves played in a games
        """
        # execute the self-play
        self_play_results = __self_play_worker__(game_class, network_path, config.episodes)


        # add the training examples to the experience buffer
        self.experience_buffer.add_new_cycle()
        tot_moves_played = len(self_play_results) / game_class.symmetry_count()  # account for symmetric boards,
        self.experience_buffer.add_data(self_play_results)

        avg_game_length = tot_moves_played / config.episodes
        return avg_game_length


    def nn_update(self, generation):
        """
        updates the neural network by picking a random batch form the experience replay
        :param generation:      the network generation (number of iterations so far)
        :return:                average policy and value loss over all mini batches
        """
        # setup the data set
        training_generator = self.experience_buffer.prepare_data(generation)
        step_size = training_generator.dataset.__len__() // (2 * config.batch_size)
        logger.info("training data prepared, step size: {}".format(step_size))


        # activate the training mode
        self.network = data_storage.net_to_device(self.network, config.training_device)
        if config.cyclic_learning:
            self.network.update_scheduler(step_size)  # update the scheduler
        self.network.train()

        avg_loss_p = 0
        avg_loss_v = 0
        tot_batch_count = 0
        for epoch in range(config.epochs):
            # training
            for state_batch, value_batch, policy_batch in training_generator:
                # send the data to the gpu
                state_batch = state_batch.to(config.training_device, dtype=torch.float)
                value_batch = value_batch.unsqueeze(1).to(config.training_device, dtype=torch.float)
                policy_batch = policy_batch.to(config.training_device, dtype=torch.float)

                # execute the training step with one batch
                if config.cyclic_learning:
                    loss_p, loss_v = self.network.train_cyclical_step(state_batch, policy_batch, value_batch)
                else:
                    loss_p, loss_v = self.network.train_step(state_batch, policy_batch, value_batch)

                avg_loss_p += loss_p
                avg_loss_v += loss_v
                tot_batch_count += 1

        # calculate the mean of the loss
        avg_loss_p /= tot_batch_count
        avg_loss_v /= tot_batch_count

        # activate the evaluation mode
        self.network = data_storage.net_to_device(self.network, config.evaluation_device)
        self.network.eval()

        return avg_loss_p.item(), avg_loss_v.item()


class SelfPlayDataSet(data.Dataset):
    def __init__(self, training_data):
        """
        data set for the neural network training
        :param training_data:      list of dicts with state, value and policy
        """
        self.training_data = training_data


    def __len__(self):
        return len(self.training_data)


    def __getitem__(self, index):
        """
        returns a sample with the passed index
        :param index:   index of the sample
        :return:        state, value, policy
        """

        return self.training_data[index].get("state"), self.training_data[index].get("value"), self.training_data[index].get("policy")



class ExperienceBuffer:
    def __init__(self):
        self.training_cycles = []       # holds a list with the training data of different cycles

        # to save the current training data
        self.data_size = 0
        self.state = None
        self.policy = None
        self.value = None


    def fill_with_initial_data(self):
        """
        fills the experience buffer with initial training data from an untrained network
        this helps to prevents early overfitting of the network
        :return:
        """

        with open("initial_training_data.pkl", 'rb') as input:
            initial_data = pickle.load(input)

        for i in range(config.min_window_size):
            self.add_new_cycle()

            start_idx = i*config.episode_count*config.initial_game_length
            end_idx = (i+1)*config.episode_count*config.initial_game_length
            data = initial_data[start_idx:end_idx]
            self.add_data(data)



    def add_new_cycle(self):
        """
        adds a new training cycle to the training cycle list
        :return:
        """
        self.training_cycles.append([])


    def add_data(self, training_examples):
        """
        adds the passed training examples to the experience buffer
        :param training_examples:    list containing the training examples
        :return:
        """

        self.training_cycles[-1] += training_examples


    def window_size_from_generation(self, generation):
        """
        returns the size of the window that is used for training
        :param generation:  the network generation
        :return:            the size of the training window
        """
        # some initilal data is used
        if config.use_initial_data:
            window_size = config.min_window_size + generation // 2
            if window_size > config.max_window_size:
                window_size = config.max_window_size

            return window_size

        # no initial data is used
        window_size = max(config.min_window_size + (generation - config.min_window_size + 1) // 2, config.min_window_size)
        if window_size > config.max_window_size:
            window_size = config.max_window_size

        return window_size


    def prepare_data(self, generation):
        """
        prepares the training data for training
        :param generation:      the generation of the network (number of iterations so far)
        :return:                torch training generator for training
        """
        # calculate the size of the window
        window_size = self.window_size_from_generation(generation)
        logger.debug("window_size: {}".format(window_size))

        # get rid of old data
        while len(self.training_cycles) > window_size:
            self.training_cycles.pop(0)

        # get the whole training data
        training_data = []
        for sample in self.training_cycles:
            training_data += sample

        # average the positions (early positions are overrepresented)
        if config.average_positions:
            training_data = self.__average_positions__(training_data)



        # create the training set
        training_set = SelfPlayDataSet(training_data)
        training_generator = data.DataLoader(training_set, **config.data_set_params)
        return training_generator


    def __average_positions__(self, training_data):
        """
        calculates the average over same positions, since connect4 only has a few reasonable starting
        lines the position at the beginning are overrepresented in the data set.
        :param training_data:   list of training samples
        :return:                list of averaged training samples that does not contain position duplicates
        """
        training_data = sorted(training_data, key=itemgetter('state_id'))

        training_data_avg = []
        state_id = training_data[0].get("state_id")
        state = training_data[0].get("state")
        position_count = 0
        policy = 0
        value = 0
        for position in training_data:
            if state_id == position.get("state_id"):
                policy = np.add(policy, position.get("policy"))
                value += position.get("value")
                position_count += 1

            else:
                policy = np.divide(policy, np.sum(policy))     # normalize the policy

                averaged_sample = {
                    "state_id": state_id,
                    "state": state,
                    "policy": policy,
                    "value": value / position_count
                }
                training_data_avg.append(averaged_sample)

                state_id = position.get("state_id")
                state = position.get("state")
                policy = position.get("policy")
                value = position.get("value")
                position_count = 1

        # add the last example as well
        if position_count > 1:
            policy = np.divide(policy, np.sum(policy))  # normalize the policy

            averaged_sample = {
                "state_id": state_id,
                "state": state,
                "policy": policy,
                "value": value / position_count
            }
            training_data_avg.append(averaged_sample)

        size_reduction = (len(training_data) - len(training_data_avg)) / len(training_data)
        logger.debug("size reduction due to position averaging: {}".format(size_reduction))
        return training_data_avg


def __self_play_worker__(game_class, network_path, game_count):
    """
    plays a number of self play games
    :param game_class:          the class of the implemented games
    :param network_path:        path of the network
    :param game_count:          the number of self-play games to play
    :return:                    a list of dictionaries with all training examples
    """
    # load the network
    net = data_storage.load_net(network_path, config.evaluation_device)

    training_expl_list = []

    # initialize the mcts object for all games
    mcts_list = [MCTS(game_class()) for _ in range(game_count)]

    # initialize the lists that keep track of the games
    player_list = [[] for _ in range(game_count)]
    state_list = [[] for _ in range(game_count)]
    state_id_list = [[] for _ in range(game_count)]
    policy_list = [[] for _ in range(game_count)]


    move_count = 0
    all_terminated = False
    while not all_terminated:
        # =========================================== execute one mcts simulations for all boards
        mcts.run_simulations(mcts_list, config.mcts_sim_count, net, config.alpha_dirich)


        # ===========================================  get the policy from the mcts
        temp = 0 if move_count >= config.temp_threshold else config.temp

        for i_mcts_ctx, mcts_ctx in enumerate(mcts_list):
            # skip terminated games
            if mcts_ctx.board.is_terminal():
                continue

            policy = mcts_list[i_mcts_ctx].policy_from_state(mcts_ctx.board.state_id(), temp)

            # add regular board
            state, player = mcts_ctx.board.white_perspective()
            state_id = mcts_ctx.board.state_id()
            state_list[i_mcts_ctx].append(state)
            state_id_list[i_mcts_ctx].append(state_id)
            player_list[i_mcts_ctx].append(player)
            policy_list[i_mcts_ctx].append(policy)


            # add symmetric boards
            board_symmetries, policy_symmetries = mcts_ctx.board.symmetries(policy)
            if board_symmetries is not None:
                for board_sym, policy_sym in zip(board_symmetries, policy_symmetries):
                    state_s, player_s = board_sym.white_perspective()
                    state_id_s = board_sym.state_id()
                    state_list[i_mcts_ctx].append(state_s)
                    state_id_list[i_mcts_ctx].append(state_id_s)
                    player_list[i_mcts_ctx].append(player_s)

                    policy_list[i_mcts_ctx].append(policy_sym)


            # sample from the policy to determine the move to play
            action = np.random.choice(len(policy), p=policy)
            mcts_ctx.board.execute_action(action)

        move_count += 1


        # ===========================================  check if there are still boards with running games
        all_terminated = True
        for mcts_ctx in mcts_list:
            if not mcts_ctx.board.is_terminal():
                all_terminated = False
                break


    # =========================================== add the training example
    for i_mcts_ctx, mcts_ctx in enumerate(mcts_list):
        reward = mcts_ctx.board.training_reward()
        for i_player, player in enumerate(player_list[i_mcts_ctx]):
            value = reward if player == CONST.WHITE else -reward

            # save the training example
            training_expl_list.append({
                "state": state_list[i_mcts_ctx][i_player],
                "state_id": state_id_list[i_mcts_ctx][i_player],
                "player": player,
                "policy": policy_list[i_mcts_ctx][i_player],
                "value": value
            })


    # free up some resources
    del net
    del mcts_list
    torch.cuda.empty_cache()

    return training_expl_list