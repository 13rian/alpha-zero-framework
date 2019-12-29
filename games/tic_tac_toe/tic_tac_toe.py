import numpy as np

from globals import CONST
import game
from games.tic_tac_toe import minimax


class TicTacToeBoard(game.GameBoard):
    """
    each player gets a separate board representation
    0  1  2
    3  4  5
    6  7  8
    if a stone is set for a player the bit string will have a 1 on the correct position
    a move is defined by a number, e.g. 4 (this represents setting a stone on the board position 4)
    """

    def __init__(self):
        self.white_player = 0
        self.black_player = 0

        self.player = CONST.WHITE       # disk of the player to move
        self.terminal = False           # is the games finished
        self.score = 0                  # -1 if black wins, 0 if it is a tie and 1 if white wins
        self.legal_moves_list = []           # holds all legal moves of the current board position

        # calculate all legal moves and disks to flip
        self.__calc_legal_moves__()


    ###########################################################################################################
    #                                   methods that need to be implemented                                   #
    ###########################################################################################################

    def is_terminal(self):
        return self.terminal


    def current_player(self):
        return self.player


    # def symmetries(self, policy):
    #     return None, None
    #
    #
    # @staticmethod
    # def symmetry_count():
    #     return 1


    def white_perspective(self):
        white_board = self.int_to_board(self.white_player)
        black_board = self.int_to_board(self.black_player)
        if self.player == CONST.WHITE:
            bit_board = np.stack((white_board, black_board), axis=0)
            player = CONST.WHITE
        else:
            bit_board = np.stack((black_board, white_board), axis=0)
            player = CONST.BLACK

        return bit_board, player


    def state_id(self):
        state = "{}_{}".format(self.white_player, self.black_player)
        return state


    def execute_action(self, move):
        """
        plays the passed move on the board
        :param move:    integer that defines the position to set the stone
        :return:
        """
        # if move not in self.legal_moves:
        #     print("move not in list")

        # set the token
        if self.player == CONST.WHITE:
            self.white_player = self.white_player + (1 << move)
        else:
            self.black_player = self.black_player + (1 << move)

        # check if the player won
        self.check_win()

        # swap the active player and calculate the legal moves
        self.swap_players()
        self.__calc_legal_moves__()


    def legal_actions(self):
        return self.legal_moves_list


    def illegal_actions(self):
        """
        returns a list of illegal moves
        :return:
        """
        # define the mask with all legal moves
        move_mask = self.white_player & self.black_player

        illegal_moves = []
        for move in range(9):
            if (1 << move) & move_mask > 0:
                illegal_moves.append(move)

        return illegal_moves


    def reward(self):
        """
        :return:    -1 if black has won
                    0 if the games is drawn or the games is still running
                    1 if white has won
        """

        if not self.terminal:
            return 0
        else:
            return self.score


    def training_reward(self):
        """
        :return:    -1 if black has won
                    0 if the games is drawn or the games is still running
                    1 if white has won
        """

        if not self.terminal:
            return 0
        else:
            return self.score


    ###########################################################################################################
    #                                               helper methods                                            #
    ###########################################################################################################
    def from_board_matrix(self, board):
        """
        creates the bit board from the passed board representation
        :param board:   games represented as one board
        :return:
        """

        white_board = board == CONST.WHITE
        white_board = white_board.astype(int)
        self.white_player = self.board_to_int(white_board)

        black_board = board == CONST.BLACK
        black_board = black_board.astype(int)
        self.black_player = self.board_to_int(black_board)

        # calculate all legal moves and disks to flip
        self.__calc_legal_moves__()

        # check the games states
        self.swap_players()
        self.check_win()
        self.swap_players()


    def print(self):
        """
        prints the current board configuration
        :return:
        """

        # create the board representation form the bit strings
        print(self.get_board_matrix())


    def get_board_matrix(self):
        """
        :return:  human readable games board representation
        """

        white_board = self.int_to_board(self.white_player)
        black_board = self.int_to_board(self.black_player)
        board = np.add(white_board * 1, black_board * 2)
        return board


    def int_to_board(self, number):
        """
        creates the 3x3 bitmask that is represented by the passed integer
        :param number:      move on the board
        :return:            x3 matrix representing the board
        """

        number = (number & 7) + ((number & 56) << 5) + ((number & 448) << 10)
        byte_arr = np.array([number], dtype=np.uint32).view(np.uint8)
        board_mask = np.unpackbits(byte_arr).reshape(-1, 8)[0:3, ::-1][:, 0:3]
        return board_mask


    def board_to_int(self, mask):
        """
        converts the passed board mask (3x3) to an integer
        :param mask:    binary board representation 3x3
        :return:        integer representing the passed board
        """
        bit_arr = np.reshape(mask, -1).astype(np.uint32)
        number = bit_arr.dot(1 << np.arange(bit_arr.size, dtype=np.uint32))
        return int(number)


    def move_to_board_mask(self, move):
        """
        :param move:    integer defining a move on the board
        :return:        the move represented as a mask on the 3x3 board
        """

        mask = 1 << move
        board_mask = self.int_to_board(mask)
        return board_mask


    def check_win(self):
        """
        checks if the current player has won the games
        :return:
        """

        if self.three_in_a_row(self.player):
            self.terminal = True
            self.score = 1 if self.player == CONST.WHITE else -1


    def swap_players(self):
        self.player = CONST.WHITE if self.player == CONST.BLACK else CONST.BLACK


    def white_score(self):
        reward = self.reward()
        return (reward + 1) / 2


    def black_score(self):
        reward = self.reward()
        return (-reward + 1) / 2


    def __calc_legal_moves__(self):
        # define the mask with all legal moves
        move_mask = bit_not(self.white_player ^ self.black_player, 9)  # this is basically an xnor (only 1 if both are 0)

        self.legal_moves_list = []
        for move in range(9):
            if (1 << move) & move_mask > 0:
                self.legal_moves_list.append(move)

        # if there are no legal moves the games is drawn
        if len(self.legal_moves_list) == 0:
            self.terminal = True


    def three_in_a_row(self, player):
        """
        checks if the passed player has a row of three
        :param player:      the player for which 3 in a row is checked
        :return:
        """

        board = self.white_player if player == CONST.WHITE else self.black_player

        # horizontal check
        if board & 7 == 7 or board & 56 == 56 or board & 448 == 448:
            return True

            # vertical check
        if board & 73 == 73 or board & 146 == 146 or board & 292 == 292:
            return True

        # diagonal check /
        if board & 84 == 84:
            return True

        # diagonal check \
        if board & 273 == 273:
            return True

        # nothing found
        return False


    def minimax_move(self):
        """
        returns the optimal minimax move, if there are more than one optimal moves, a ranodm one is
        picked
        :return:
        """

        # get the white score for all legal moves
        score_list = np.empty(len(self.legal_moves_list))
        for idx, move in enumerate(self.legal_moves_list):
            board_clone = self.clone()
            board_clone.execute_action(move)
            state = board_clone.state_id()
            white_score = minimax.state_dict.get(state)
            score_list[idx] = white_score

        # find the indices of the max score for white and the min score for black
        if self.player == CONST.WHITE:
            move_indices = np.argwhere(score_list == np.amax(score_list))
        else:
            move_indices = np.argwhere(score_list == np.amin(score_list))

        move_indices = move_indices.squeeze(axis=1)
        best_moves = np.array(self.legal_moves_list)[move_indices]
        best_move = np.random.choice(best_moves, 1)
        return int(best_move)



def bit_not(n, bit_length):
    """
    defines the logical not operation
    :param n:            the number to which the not operation is applied
    :param bit_length:   the length of the bit to apply the not operation
    :return:
    """
    return (1 << bit_length) - 1 - n
