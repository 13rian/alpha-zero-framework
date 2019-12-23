import random
import torch

import numpy as np

from utils import utils
from globals import CONST, Config
import game

# mask that defines the location of the upper bits of the board
upper_row_mask = (1 << 5) + (1 << 12) + (1 << 19) + (1 << 26) + (1 << 33) + (1 << 40) + (1 << 47)


class Connect4Board(game.GameBoard):
    """
    the position is represented like this
    5  12 19 26 32 40 47
    4  11 18 25 31 39 46
    3  10 17 24 30 38 45
    2  9  16 23 29 37 44
    1  8  15 22 28 36 43
    0  7  14 21 27 35 42
    ...
    on integer will represent the set disks of one player and the other integer is a mask with all moves
    """

    def __init__(self):
        super().__init__()
        self.position = 0                           # bit mask with the disks of the current player
        self.disk_mask = 0                          # bit mask with all disks of the games
        self.move_count = 0                         # the number of moves played
        
        self.player = CONST.WHITE                   # disk of the player to move
        self.terminal = False                       # is the games finished
        self.score = 0                              # -1 if black wins, 0 if it is a tie and 1 if white wins


    ###########################################################################################################
    #                                   methods that need to be implemented                                   #
    ###########################################################################################################

    def is_terminal(self):
        return self.terminal


    def current_player(self):
        return self.player


    def symmetries(self, policy):
        board_m = self.mirror()
        policy_m = np.flip(policy)

        return [board_m], [policy_m]


    @staticmethod
    def symmetry_count():
        return 1


    def white_perspective(self):
        """
        returns the board from the white perspective. If it is white's move the normal board representation is returned.
        if it is black's move the white and the black pieces are swapped.
        the board is rotated by 90° in order to avoid the expensive rotation operation, the disks are played from the
        right side.
        :return:
        """

        white_board = self.int_to_board(self.get_white_position())
        black_board = self.int_to_board(self.get_black_position())
        if self.player == CONST.WHITE:
            bit_board = np.stack((white_board, black_board), axis=0)
        else:
            bit_board = np.stack((black_board, white_board), axis=0)

        return bit_board, self.player


    def state_id(self):
        """
        uses the cantor pairing function to create one unique id for the state form the two integers representing the
        board state
        """
        # state = "{}_{}".format(self.position, self.disk_mask)
        state = self.position + self.disk_mask
        return state


    def play_move(self, column):
        """
        plays the passed move on the board
        :param column:    integer that defines the column in which the disk is played
        :return:
        """
        column = int(column)
        # old_mask = self.disk_mask
        self.position = self.position ^ self.disk_mask                              # get position of the opponent
        self.disk_mask = self.disk_mask | (self.disk_mask + (1 << (column * 7)))

        self.move_count += 1
        self.check_win(self.position ^ self.disk_mask)
        self.player = self.move_count % 2

        if self.move_count == 42:
            self.terminal = True


    def legal_moves(self):
        legal_moves = []
        for i in range(Config.board_width):
            top_mask = (1 << (Config.board_height - 1)) << i * (Config.board_height + 1)
            if (self.disk_mask & top_mask) == 0:
                legal_moves.append(i)

        return legal_moves


    def illegal_moves(self):
        illegal_moves = []
        for idx, val in enumerate([5, 12, 19, 26, 33, 40, 47]):
            if self.disk_mask & (1 << val):
                illegal_moves.append(idx)
        return illegal_moves


    def reward(self):
        """
        :return:    -1 if black has won
                    0 if the games is drawn or the games is still running
                    1 if white has won
        """
        return self.score


    def training_reward(self):
        """
        returns the reward for training
        :return:
        """
        return self.score

        # the reward below prefers quick wins over slow wins
        # return self.score * (1.18 - (9*self.move_count) / 350)



    ###########################################################################################################
    #                                               helper methds                                             #
    ###########################################################################################################
    def mirror(self):
        """
        returns a position that is mirrored on the y - axis
        :return:
        """
        mirrored_position = self.clone()
        mirrored_position.position = self.mirror_board_number(self.position)
        mirrored_position.disk_mask = self.mirror_board_number(self.disk_mask)
        return mirrored_position


    def mirror_board_number(self, number):
        """
        mirrors the passed number representing a board
        :param number:  any number representing some board configuration (position, board mask, disk maksk etc)
        :return:
        """
        mirrored_number = 0

        # left half of the board
        for col in range((Config.board_width+1) // 2 - 1):
            mirrored_number += (number & column_mask(col)) << ((Config.board_width - (2 * col + 1)) * (Config.board_height + 1))

        # right half of the board
        for col in range((Config.board_width+1) // 2 - 1):
            mirrored_number += (number & column_mask(Config.board_width - col - 1)) >> ((Config.board_width - (2 * col + 1)) * (Config.board_height + 1))

        # center row
        if Config.board_width % 2 != 0:
            col = Config.board_width // 2
            mirrored_number += (number & column_mask(col))

        return mirrored_number


    def from_board_matrix(self, board):
        """
        creates the bit board from the passed board representation
        :param board:   games represented as one board
        :return:
        """
        self.disk_mask = 0
        self.position = 0

        count = 0
        self.move_count = 0
        for x in range(7):
            for y in range(5, -1, -1):
                if board[y, x]:
                    self.disk_mask += 1 << (count + x)
                    self.move_count += 1
                if board[y, x] == CONST.WHITE + 1:
                    self.position += 1 << (count + x)

                count += 1

        self.player = self.move_count % 2

        if self.player == CONST.BLACK:
            self.position = self.position ^ self.disk_mask

        # check the games states
        self.check_win(self.position)


    def from_position(self, position, disk_mask):
        """
        creates a position from the integer representations
        :param position:    position of the current player
        :param disk_mask:   all disks that are set
        :return:
        """

        self.position = position
        self.disk_mask = disk_mask
        self.move_count = utils.popcount(disk_mask)
        self.player = self.move_count % 2

        # check the games states
        self.check_win(self.position)


    def print(self):
        """
        prints the current board configuration
        :return:
        """
        print(self.to_board_matrix())


    def to_board_matrix(self):
        """
        :return:  human readable games board representation
        """

        white_board = self.int_to_board(self.get_white_position())
        white_board = np.rot90(white_board, 1)
        black_board = self.int_to_board(self.get_black_position())
        black_board = np.rot90(black_board, 1)
        board = np.add(white_board * (CONST.WHITE + 1), black_board * (CONST.BLACK + 1))
        return board


    def int_to_board(self, number):
        """
        creates the 6x7 bitmask that is represented by the passed integer
        the board will be rotated by 90° in order to avoid the expensive
        rotation operation
        :param number:      move on the board
        :return:            6x7 matrix representing the board
        """

        mask = 0x03F
        row_number = 0
        for i in range(7):
            row = number & (mask << (i*7))
            row_number = row_number + (row << i)

        byte_arr = np.array([row_number], dtype=np.uint64).view(np.uint8)
        board_mask = np.unpackbits(byte_arr).reshape(-1, 8)[0:7, ::-1][:, 0:6]
        return board_mask


    def get_white_position(self):
        white_position = self.position if self.player == CONST.WHITE else self.position ^ self.disk_mask
        return white_position


    def get_black_position(self):
        black_position = self.position if self.player == CONST.BLACK else self.position ^ self.disk_mask
        return black_position


    def check_win(self, position):
        """
        checks if the current player has won the games
        :param position:    the position that is checked
        :return:
        """
        if self.four_in_a_row(position):
            self.terminal = True
            self.score = 1 if self.player == CONST.WHITE else -1


    def is_legal_move(self, move):
        if move in self.legal_moves():
            return True
        else:
            return False


    def four_in_a_row(self, position):
        """
        checks if the passed player has a row of four
        :param position:    the position that is checked
        :return:
        """

        # vertical
        temp = position & (position >> 1)
        if temp & (temp >> 2):
            return True

        # horizontal
        temp = position & (position >> 7)
        if temp & (temp >> 14):
            return True

        # diagonal /
        temp = position & (position >> 8)
        if temp & (temp >> 16):
            return True

        # diagonal \
        temp = position & (position >> 6)
        if temp & (temp >> 12):
            return True

        # no row of 4 found
        return False


    def set_player_white(self):
        self.player = CONST.WHITE
        

    def set_player_black(self):
        self.player = CONST.BLACK


    def white_score(self):
        reward = self.reward()
        return (reward + 1) / 2


    def black_score(self):
        reward = self.reward()
        return (-reward + 1) / 2


    def network_prediction(self, net, torch_device):
        batch, _ = self.white_perspective()
        batch = torch.Tensor(batch).unsqueeze(0).to(torch_device)
        return net(batch)



def column_mask(col):
    """
    bitmask with all cells of the passed column
    :param col:     board column
    :return:
    """
    return ((1 << Config.board_height) - 1) << col * (Config.board_height + 1)