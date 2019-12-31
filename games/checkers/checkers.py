import random

import numpy as np

import game
from globals import CONST
from utils import utils
from games.checkers.configuration import Config


SKIP_BITS = 0x804020100                     # bits that are not used for the board representation
ALL_BITS = 0xfffffffff                      # defines all bits
BAKWARDS_BIT = (1 << 50)                    # bit that is set for backwards moves

WHITE_BACKRANK_BITS = 0xf                   # bits that define white's backrank
BLACK_BACKRANK_BITS = 0x780000000           # bits that define black's backrank


#################################################################################################################
#                                             all checkers moves                                                #
#################################################################################################################
# define all possible directions
directions = [(-1, 1), (-1, -1), (1, -1), (1, 1)]
capture_dirs = [(-2, 2), (-2, -2), (2, -2), (2, 2)]


def add_tuples(a, b):
    return (a[0] + b[0], a[1] + b[1])


def is_valid_index(idx):
    if idx[0] > 7 or idx[1] > 7 or idx[0] < 0 or idx[1] < 0:
        return False
    else:
        return True


def all_pdn_moves():
    """
    returns a list with all possible moves that can occur in a game situation. the notation is like pdn, the start and
    the destination square are separated by a minus for normal moves and by a x for a capture. capture sequences are
    not considered in this list. only one capture step is defined. if multiple captures are possible they are executed
    until the player needs to make a new decision. this is a special game state where the player needs
    to make two moves in a row.
    :return:
    """

    # create the board with the move notation on it
    board = np.zeros((8, 8))

    count = 1
    for row in range(7, -1, -1):
        for col in range(7, -1, -1):
            if row % 2 == 0 and col % 2 == 1:
                board[row][col] = count
                count += 1

            if row % 2 == 1 and col % 2 == 0:
                board[row][col] = count
                count += 1

    # find all moves with no captures
    moves = []
    for row in range(7, -1, -1):
        for col in range(7, -1, -1):
            # skip squares that are not part of the state space
            if board[row][col] < 1:
                continue

            for direction in directions:
                # create the new index
                from_pos = (row, col)
                to_pos = add_tuples(from_pos, direction)
                if not is_valid_index(to_pos):
                    continue

                moves.append("{}-{}".format(int(board[from_pos]), int(board[to_pos])))

    # all capture moves
    for row in range(7, -1, -1):
        for col in range(7, -1, -1):
            # skip squares that are not part of the state space
            if board[row][col] < 1:
                continue

            for direction in capture_dirs:
                # create the new index
                from_pos = (row, col)
                to_pos = add_tuples(from_pos, direction)
                if not is_valid_index(to_pos):
                    continue

                moves.append("{}x{}".format(int(board[from_pos]), int(board[to_pos])))

    return moves


def all_moves(pdn_moves):
    """
    returns a list of moves in the bit board representation. a move is defined by an integer that has two bits turned
    on, the start and the end square. if it is a backwards move (from the white perspective) the backwards bit
    is set as well in order to have a unique move definition
    :param pdn_moves:    list of all pdn moves
    :return:
    """
    moves = []

    for pdn_move in pdn_moves:
        # convert normal moves to the integer representation
        if "-" in pdn_move:
            pos_str = pdn_move.split("-")
            from_square = int(pos_str[0]) - 1
            from_square = from_square + from_square // 8
            to_square = int(pos_str[1]) - 1
            to_square = to_square + to_square // 8

            move = (1 << from_square) + (1 << to_square)
            if from_square > to_square:
                move += BAKWARDS_BIT

            moves.append(move)

        # convert capture moves to the integer representation
        if "x" in pdn_move:
            pos_str = pdn_move.split("x")
            from_square = int(pos_str[0]) - 1
            from_square = from_square + from_square // 8
            to_square = int(pos_str[1]) - 1
            to_square = to_square + to_square // 8

            move = (1 << from_square) + (1 << to_square)
            if from_square > to_square:
                move += BAKWARDS_BIT

            move *= -1
            moves.append(move)

    return moves


def create_move_dicts(pdn_moves, moves):
    """
    creates two dictionaries to convert pdn_moves to moves. the pdn_moves and the moves need to have the same order
    :param pdn_moves:   a list of pdn-moves
    :param moves:       a list of moves
    :return:
    """
    pdn_dict = {}
    move_dict = {}

    for pdn_move, move in zip(pdn_moves, moves):
        pdn_dict[pdn_move] = move
        move_dict[move] = pdn_move

    return pdn_dict, move_dict


def move_action_dict(moves):
    """
    returns a lookup table with the move as key and the policy index as value
    :param moves:      all possible moves
    :return:
    """
    lookup_table = {}
    for i, label in enumerate(moves):
        lookup_table[label] = i

    return lookup_table


PDN_MOVES = all_pdn_moves()
MOVES = all_moves(PDN_MOVES)
PDN_MOVE_DICT, MOVE_PDN_DICT = create_move_dicts(PDN_MOVES, MOVES)
MOVE_ACTION_DICT = move_action_dict(MOVES)
ALL_MOVE_COUNT = len(PDN_MOVES)



#################################################################################################################
#                                               move conversions                                                #
#################################################################################################################
def reverse_bits(n, bit_count):
    """
    reversed the order of the bits in the passed number
    :param n:               number of which the bits are reversed
    :param bit_count:      the number of bits that are used for the passed number
    :return:
    """
    rev = 0

    # traversing bits of 'n' from the right
    for _ in range(bit_count):
        rev <<= 1
        if n & 1:
            rev = rev ^ 1

        n >>= 1
    return rev


def mirror_move(move):
    """
    converts a move for one player to a move for the other player
    :param move:    the move to convert
    :return:
    """
    # change the sign to a positive sign
    is_capture = move < 0
    if is_capture:
        move *= -1

    # remove the backwards bit
    is_backwards = move & BAKWARDS_BIT
    move &= ALL_BITS

    # reverse the bits of the move
    reversed_move = reverse_bits(move, 35)

    # forward move become backwards move and vice versa
    if not is_backwards:
        reversed_move += BAKWARDS_BIT

    # set the capture information
    if is_capture:
        reversed_move *= -1

    return reversed_move


def action_to_move(action, player):
    """
    converts the passed action to a checkers move. as the action is always defined from the white perspective
    it needs to be mirrored for the black player
    :param action:      move policy vector
    :param player:      the current player
    :return:            move that can be played on the board
    """
    # convert the action to a checkers move
    move = MOVES[action]

    # mirror the move for the black player
    if player == CONST.BLACK:
        move = mirror_move(move)

    return move


def move_to_action(move, player):
    """
    returns the action that corresponds to the passed move to play
    :param move:    checkers move
    :param player:  current player
    :return:
    """
    # mirror the the moves for the black player since the board is always viewed from the white perspective
    if player == CONST.BLACK:
        move = mirror_move(move)

    action = MOVE_ACTION_DICT[move]
    return action


def action_to_pdn(action, player):
    move = action_to_move(action, player)
    pdn_move = MOVE_PDN_DICT[move]
    return pdn_move


def pdn_to_action(pdn_move, player):
    move = PDN_MOVE_DICT[pdn_move]
    action = move_to_action(move, player)
    return action


#################################################################################################################
#                                           checkers game board                                                 #
#################################################################################################################
class CheckersBoard(game.GameBoard):
    """
    each player gets a separate matrix that represents the board. for faster calculations the games is represented with
    4 integers 2 for the white and the black men and 2 for the white and the black kings. the board is represented
    by turnend on bits as follows:
    board representation:
    0  32 0  31 0  30 0  29
    28 0  27 0  26 0  25 0
    0  24 0  23 0  22 0  21
    20 0  19 0  18 0  17 0
    0  16 0  15 0  14 0  13
    12 0  11 0  10 0  9  0
    0  8  0  7  0  6  0  5
    4  0  3  0  2  0  1  0
    every second row one bit is skipped in order to have the same bit shift for all captures
    """

    def __init__(self):
        self.white_disks = 0x1eff           # white pieces
        self.white_kings = 0                # white pieces that are allowed to move backwards
        self.black_disks = 0x7fbc00000      # black pieces
        self.black_kings = 0                # white pieces that are allowed to move backwards

        self.empty = SKIP_BITS ^ ALL_BITS ^ (self.white_disks | self.black_disks)  # holds all empty squares

        self.player = CONST.WHITE                   # current player
        self.move_count = 0                         # the number of moves played so far
        self.no_progress_count = 0                  # counts the half moves after the last men moved or the last piece was captured
        self.terminal = False                       # is the game finished
        self.score = 0                              # -1 if black wins, 0 if it is a tie and 1 if white wins

        self.mandatory_captures = []            # list with all possible capture moves



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
        if self.player == CONST.WHITE:
            white_men_board = int_to_board(self.white_disks ^ self.white_kings)
            white_kings_board = int_to_board(self.white_kings)
            black_men_board = int_to_board(self.black_disks ^ self.black_kings)
            black_kings_board = int_to_board(self.black_kings)

            bit_board = np.stack((white_men_board, white_kings_board, black_men_board, black_kings_board), axis=0)

        else:
            # the network sees all from the white perspective so the board needs to be rotated by 180°
            white_men_board = int_to_mirrored_board(self.white_disks ^ self.white_kings)
            white_kings_board = int_to_mirrored_board(self.white_kings)
            black_men_board = int_to_mirrored_board(self.black_disks ^ self.black_kings)
            black_kings_board = int_to_mirrored_board(self.black_kings)

            bit_board = np.stack((black_men_board, black_kings_board, white_men_board, white_kings_board), axis=0)

        return bit_board, self.player


    def state_id(self):
        state = "{}_{}_{}_{}".format(self.white_disks, self.white_kings, self.black_disks, self.black_kings)
        return state


    def execute_action(self, action):
        """
        executes the passed move on the baord, a move is represented by an integer
        that has the start and the end bit turned on
        :param action:      the action to play on the checkers board
        """
        # create a move from the passed action
        move = action_to_move(action, self.player)

        # increment the no progress count, it will be set to 0 later on if either a man moved or a piece as captured
        self.no_progress_count += 1

        # execute the capture
        if move < 0:
            self.no_progress_count = 0      # reset the no progress count as a piece was captured
            is_capture = True

            # remove the backwards bit
            move *= -1
            move &= ALL_BITS

            # the average of the bit position is the captured piece
            captured_disk = int(1 << sum(i for (i, b) in enumerate(bin(move)[::-1]) if b == '1')//2)

            if self.player == CONST.WHITE:
                # remove the captured opponent disk
                if self.black_disks & captured_disk:
                    self.black_disks ^= captured_disk
                if self.black_kings & captured_disk:
                    self.black_kings ^= captured_disk

            else:
                # remove the captured opponent disk
                if self.white_disks & captured_disk:
                    self.white_disks ^= captured_disk
                if self.white_kings & captured_disk:
                    self.white_kings ^= captured_disk

        else:
            is_capture = False
            move &= ALL_BITS  # remove the backwards bit

        # execute the move
        if self.player == CONST.WHITE:
            # set the disk at the new position
            if self.white_disks & move:
                self.white_disks ^= move
            if self.white_kings & move:
                self.white_kings ^= move

            # reset the no progress count if a man moved
            if (self.white_disks ^ self.white_kings) & move:
                self.no_progress_count = 0

            # check if white made a king
            to_square = move & self.white_disks
            if to_square & BLACK_BACKRANK_BITS:
                self.white_kings |= to_square

        else:
            # set the disk at the new position
            if self.black_disks & move:
                self.black_disks ^= move
            if self.black_kings & move:
                self.black_kings ^= move

            # reset the no progress count if a man moved
            if (self.black_disks ^ self.black_kings) & move:
                self.no_progress_count = 0

            # check if black made a king
            to_square = move & self.black_disks
            if to_square & WHITE_BACKRANK_BITS:
                self.black_kings |= to_square

        # update the empty squares
        self.empty = SKIP_BITS ^ ALL_BITS ^ (self.white_disks | self.black_disks)


        # check if there are still possible capture moves
        if is_capture:
            self.mandatory_captures = self.captures_from_position(to_square)
            capture_count = len(self.mandatory_captures)

            # only one mandatory capture, execute it
            if capture_count == 1:
                capture_action = move_to_action(self.mandatory_captures[0], self.player)
                self.execute_action(capture_action)
                return

            # more than one capture, the same player needs to decide which capture to execute
            elif capture_count > 1:
                return


        # no mandatory captures, change the player
        self.move_count += 1
        self.player = self.move_count % 2

        # find the new capture moves
        self.mandatory_captures = self.all_captures()


        # check if the game has ended
        self.check_terminal()


    def legal_actions(self):
        """
        returns a list with all legal actions
        :return:
        """
        # check if there are some captures
        if len(self.mandatory_captures) > 0:
            if self.player == CONST.WHITE:
                actions = [MOVE_ACTION_DICT[move] for move in self.mandatory_captures]
            else:
                actions = [MOVE_ACTION_DICT[mirror_move(move)] for move in self.mandatory_captures]
            return actions

        # no captures possible, find normal moves
        if self.player == CONST.WHITE:
            rf = (self.empty >> 4) & self.white_disks       # right forward move
            lf = (self.empty >> 5) & self.white_disks       # left forward move
            rb = (self.empty << 4) & self.white_kings       # right backwards move
            lb = (self.empty << 5) & self.white_kings       # left backwards move

        else:
            rf = (self.empty >> 4) & self.black_kings       # right forward move
            lf = (self.empty >> 5) & self.black_kings       # left forward move
            rb = (self.empty << 4) & self.black_disks       # right backwards move
            lb = (self.empty << 5) & self.black_disks       # left backwards move

        moves = [0x11 << i for (i, bit) in enumerate(bin(rf)[::-1]) if bit == '1']
        moves += [0x21 << i for (i, bit) in enumerate(bin(lf)[::-1]) if bit == '1']
        moves += [BAKWARDS_BIT + (0x11 << (i - 4)) for (i, bit) in enumerate(bin(rb)[::-1]) if bit == '1']
        moves += [BAKWARDS_BIT + (0x21 << (i - 5)) for (i, bit) in enumerate(bin(lb)[::-1]) if bit == '1']

        if self.player == CONST.WHITE:
            actions = [MOVE_ACTION_DICT[move] for move in moves]
        else:
            actions = [MOVE_ACTION_DICT[mirror_move(move)] for move in moves]   # mirror the moves for black

        return actions


    def illegal_actions(self):
        """
        returns a list of illegal actions
        :return:
        """
        legal_actions = self.legal_actions()
        illegal_actions = [a for a in range(Config.tot_actions)]
        for legal_action in legal_actions:
            illegal_actions.remove(legal_action)

        return illegal_actions


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
    def play_pdn_move(self, pdn_move):
        """
        plays the passed pdn_move
        :param pdn_move:    pdn move to play
        :return:
        """
        action = pdn_to_action(pdn_move, self.player)
        self.execute_action(action)


    def check_terminal(self):
        """
        checks if the game is terminal
        :return:
        """
        # check if the current player can move
        if len(self.legal_actions()) == 0:
            self.score = 1 if self.player == CONST.BLACK else -1
            self.terminal = True
            return

        # the game is drawn if during the last 40 moves no men was moved or no piece was captured
        if self.no_progress_count >= 80:
            self.score = 0
            self.terminal = True
            return

        # rule out some basic endgames to shorten the games
        popcount_kings_w = utils.popcount(self.white_kings)
        popcount_kings_b = utils.popcount(self.black_kings)

        # no men
        if (self.white_disks ^ self.white_kings) == 0 and (self.black_disks ^ self.black_kings) == 0:
            # same king count which is not larger than 3 is a draw
            if popcount_kings_w <= 3 and popcount_kings_w == popcount_kings_b:
                self.score = 0
                self.terminal = True
                return

            if popcount_kings_w <= 3 and popcount_kings_b <= 3:
                if popcount_kings_w > popcount_kings_b:
                    self.score = 1
                    self.terminal = True
                    return

                if popcount_kings_b > popcount_kings_w:
                    self.score = -1
                    self.terminal = True
                    return




    def captures_from_position(self, position_mask):
        """
        returns a list of all possible captures form the passed position
        :param position_mask:   disk position to analyze
        :return:                list of all possible capture moves
        """
        # the rules for a successful capture are empty at the destination and an opponent disk inbetween
        # the start and the end square
        if self.player == CONST.WHITE:
            # captures for men
            rfc = (self.empty >> 8) & (self.black_disks >> 4) & position_mask        # right forward capture
            lfc = (self.empty >> 10) & (self.black_disks >> 5) & position_mask       # left forward capture

            # captures for kings
            if position_mask & self.black_kings:
                rbc = (self.empty << 8) & (self.black_disks << 4) & position_mask    # right backwards capture
                lbc = (self.empty << 10) & (self.black_disks << 5) & position_mask   # left backwards capture
            else:
                rbc = 0
                lbc = 0

        else:
            # captures for men
            rbc = (self.empty << 8) & (self.white_disks << 4) & position_mask       # right backwards capture
            lbc = (self.empty << 10) & (self.white_disks << 5) & position_mask      # left backwards capture

            # captures for kings
            if position_mask & self.white_kings:
                rfc = (self.empty >> 8) & (self.white_disks >> 4) & position_mask   # right forward capture
                lfc = (self.empty >> 10) & (self.white_disks >> 5) & position_mask  # left forward capture
            else:
                rfc = 0
                lfc = 0

        capture_moves = []
        if (rfc | lfc | rbc | lbc) != 0:
            capture_moves += [-0x101 << i for (i, bit) in enumerate(bin(rfc)[::-1]) if bit == '1']
            capture_moves += [-0x401 << i for (i, bit) in enumerate(bin(lfc)[::-1]) if bit == '1']
            capture_moves += [-BAKWARDS_BIT - (0x101 << (i - 8)) for (i, bit) in enumerate(bin(rbc)[::-1]) if bit == '1']
            capture_moves += [-BAKWARDS_BIT - (0x401 << (i - 10)) for (i, bit) in enumerate(bin(lbc)[::-1]) if bit == '1']

        return capture_moves


    def all_captures(self):
        """
        returns a list of all captures of the current position
        :return:
        """

        if self.player == CONST.WHITE:
            rfc = (self.empty >> 8) & (self.black_disks >> 4) & self.white_disks        # right forward capture
            lfc = (self.empty >> 10) & (self.black_disks >> 5) & self.white_disks       # left forward capture
            rbc = (self.empty << 8) & (self.black_disks << 4) & self.white_kings        # right backwards capture
            lbc = (self.empty << 10) & (self.black_disks << 5) & self.white_kings       # left backwards capture

        else:
            rfc = (self.empty >> 8) & (self.white_disks >> 4) & self.black_kings        # right forward capture
            lfc = (self.empty >> 10) & (self.white_disks >> 5) & self.black_kings       # left forward capture
            rbc = (self.empty << 8) & (self.white_disks << 4) & self.black_disks        # right backwards capture
            lbc = (self.empty << 10) & (self.white_disks << 5) & self.black_disks       # left backwards capture

        capture_moves = []
        if (rfc | lfc | rbc | lbc) != 0:
            capture_moves += [-0x101 << i for (i, bit) in enumerate(bin(rfc)[::-1]) if bit == '1']
            capture_moves += [-0x401 << i for (i, bit) in enumerate(bin(lfc)[::-1]) if bit == '1']
            capture_moves += [-BAKWARDS_BIT - (0x101 << (i - 8)) for (i, bit) in enumerate(bin(rbc)[::-1]) if bit == '1']
            capture_moves += [-BAKWARDS_BIT - (0x401 << (i - 10)) for (i, bit) in enumerate(bin(lbc)[::-1]) if bit == '1']

        return capture_moves


    def random_action(self):
        """
        returns a random legal action
        :return:
        """
        return random.choice(self.legal_actions())


    def print(self):
        """
        prints out the board in human readable form
        :return:
        """
        white_men = self.white_disks ^ self.white_kings
        black_men = self.black_disks ^ self.black_kings

        board = np.zeros((8, 8))
        pos = -1
        for row in range(7, -1, -1):
            for col in range(7, -1, -1):
                if row % 2 == 0 and col % 2 == 1:
                    pos += 1

                elif row % 2 == 1 and col % 2 == 0:
                    pos += 1

                else:
                    # field not used
                    continue

                shift = pos + pos // 8
                cell = 1 << shift
                if cell & white_men:
                    board[row][col] = 1

                elif cell & self.white_kings:
                    board[row][col] = 2

                elif cell & black_men:
                    board[row][col] = 3

                elif cell & self.black_kings:
                    board[row][col] = 4

        # print(board)

        str_board = ""
        for row in range(8):
            for col in range(8):
                square = board[row][col]
                if square == 1:
                    str_board += "w  "
                elif square == 2:
                    str_board += "W  "
                elif square == 3:
                    str_board += "b  "
                elif square == 4:
                    str_board += "B  "
                else:
                    str_board += "0  "

            str_board += "\n"

        print(str_board)


def int_to_board(number):
    """
    converts the passed integer to the board representation
    :param number:      integer to convert to one channel of the neural network input
    :return:
    """
    board = np.zeros((8, 8))
    pos = -1
    for row in range(7, -1, -1):
        for col in range(7, -1, -1):
            if row % 2 == 0 and col % 2 == 1:
                pos += 1

            elif row % 2 == 1 and col % 2 == 0:
                pos += 1

            else:
                continue    # field not used

            shift = pos + pos // 8
            cell = 1 << shift
            if cell & number:
                board[row][col] = 1

    return board


def int_to_mirrored_board(number):
    """
    converts the passed integer to the board representation that is rotated
    :param number:      integer to convert to one channel of the neural network input
    :return:
    """
    board = np.zeros((8, 8))
    pos = -1
    for row in range(8):
        for col in range(8):
            if row % 2 == 0 and col % 2 == 1:
                pos += 1

            elif row % 2 == 1 and col % 2 == 0:
                pos += 1

            else:
                continue    # field not used

            shift = pos + pos // 8
            cell = 1 << shift
            if cell & number:
                board[row][col] = 1

    return board
