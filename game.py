import copy


class GameBoard:
    """
    class that contains the complete games logic, this is the only place AlphaZero requires human knowledge
    (basic rules of the games to learn)
    """

    def clone(self):
        """
        returns a new board with the same state
        :return:
        """
        board = copy.deepcopy(self)
        return board


    def is_terminal(self):
        """
        returns true if the position is terminal and false if the games is still running
        :return:
        """
        pass


    def current_player(self):
        """
        returns the current player, needs to be CONST.WHITE or CONST.BLACK
        :return:
        """
        pass


    def symmetries(self, policy):
        """
        :param policy:      the policy of the original board
        returns a list of symmetric boards and policies or None if there are no symmetries or if symmetric positions
        should not be used for the training. the list should not include the original board. the boards and the
        policies need to have the same order in order to be able to map them to each other
        :return:    list of boards
                    list of policy
        """
        return None, None


    @staticmethod
    def symmetry_count():
        """
        returns the number of symmetries that are created by symmetric_boards(). the symmetry count is always 1 larger
        than the size of the list returned by symmetric_boards(). if there is no symmetry this count should be 1
        :return:
        """
        return 1

    
    def white_perspective(self):
        """
        returns the board from the white perspective. If it is white's move the normal board representation is returned.
        if it is black's move the white and the black pieces are swapped.
        :return:    the matrix representation of the board
                    the current player (CONST.WHITE or CONST.BLACK)
        """
        pass
    
    
    def state_id(self):
        """
        returns a unique state id of the current board
        :return:
        """
        pass


    def play_move(self, move):
        """
        plays the passed move on the board
        :param move:    integer that defines the column in which the disk is played
        :return:
        """
        pass


    def illegal_moves(self):
        """
        returns a list of all legal moves
        :return:
        """
        pass


    def illegal_moves(self):
        """
        returns a list of all illegal moves
        :return:
        """
        pass


    def reward(self):
        """
        returns the reward of the games
        :return:    -1 if black has won
                    0 if the games is drawn or the games is still running
                    1 if white has won
        """
        pass


    def training_reward(self):
        """
        returns the reward for training, this is normally the same method as self.reward()
        :return:
        """
        pass
