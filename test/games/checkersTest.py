import unittest

from games.checkers import checkers
from globals import CONST


class TestCheckers(unittest.TestCase):
    def setUp(self):
        self.test_game = TestGame()


    def test_game(self):
        self.test_game.white_has_one_legal_capture()
        self.assertEqual(self.test_game.current_player(), CONST.WHITE)
        self.assertEqual(len(self.test_game.legal_pdn_moves()), 1)
        self.assertTrue("x" in self.test_game.legal_pdn_moves()[0])

        self.test_game.black_has_two_captures()
        self.assertEqual(self.test_game.current_player(), CONST.BLACK)
        self.assertEqual(len(self.test_game.legal_pdn_moves()), 2)
        self.assertTrue("x" in self.test_game.legal_pdn_moves()[0])
        self.assertTrue("x" in self.test_game.legal_pdn_moves()[1])

        self.test_game.black_has_a_king()
        self.assertTrue("B" in self.test_game.board.to_string())

        self.test_game.black_can_make_king_move()

        self.test_game.black_king_can_be_captured()
        self.assertEqual(len(self.test_game.legal_pdn_moves()), 1)
        self.assertTrue("x" in self.test_game.legal_pdn_moves()[0])

        self.test_game.black_king_captured()
        self.assertTrue("B" not in self.test_game.board.to_string())

        self.test_game.white_one_branch_capture()
        self.assertEqual(self.test_game.current_player(), CONST.WHITE)
        self.assertEqual(len(self.test_game.legal_pdn_moves()), 1)
        self.assertTrue("x" in self.test_game.legal_pdn_moves()[0])

        self.test_game.white_needs_to_decide_at_branch()
        self.assertEqual(self.test_game.current_player(), CONST.WHITE)
        self.assertEqual(len(self.test_game.legal_pdn_moves()), 2)
        self.assertTrue("x" in self.test_game.legal_pdn_moves()[0])
        self.assertTrue("x" in self.test_game.legal_pdn_moves()[1])

        self.test_game.white_has_king()
        self.assertTrue("W" in self.test_game.board.to_string())

        self.test_game.black_can_capture_whites_king()
        self.assertEqual(self.test_game.current_player(), CONST.BLACK)
        self.assertEqual(len(self.test_game.legal_pdn_moves()), 1)
        self.assertTrue("x" in self.test_game.legal_pdn_moves()[0])

        self.test_game.no_white_king_anymore()
        self.assertTrue("W" not in self.test_game.board.to_string())

        self.test_game.black_one_double_capture_and_branch_capture()
        self.assertEqual(self.test_game.current_player(), CONST.BLACK)
        self.assertEqual(len(self.test_game.legal_pdn_moves()), 2)
        self.assertTrue("x" in self.test_game.legal_pdn_moves()[0])
        self.assertTrue("x" in self.test_game.legal_pdn_moves()[1])

        self.test_game.black_needs_to_decide_at_branch()
        self.assertEqual(self.test_game.current_player(), CONST.BLACK)
        self.assertEqual(len(self.test_game.legal_pdn_moves()), 2)
        self.assertTrue("x" in self.test_game.legal_pdn_moves()[0])
        self.assertTrue("x" in self.test_game.legal_pdn_moves()[1])


        self.test_game.black_two_captures_and_king_capture()
        self.assertEqual(len(self.test_game.legal_pdn_moves()), 3)
        self.assertTrue("x" in self.test_game.legal_pdn_moves()[0])
        self.assertTrue("x" in self.test_game.legal_pdn_moves()[1])
        self.assertTrue("x" in self.test_game.legal_pdn_moves()[2])



class TestGame:
    def __init__(self):
        self.board = checkers.CheckersBoard()

    def legal_pdn_moves(self):
        legal_actions = self.board.legal_actions()
        return [checkers.action_to_pdn(legal_action, self.board.player) for legal_action in legal_actions]

    def current_player(self):
        return self.board.current_player()

    def white_has_one_legal_capture(self):
        self.board.play_pdn_move("11-15")
        self.board.play_pdn_move("24-19")

    def black_has_two_captures(self):
        self.board.play_pdn_move("15x24")

    def black_has_a_king(self):
        self.board.play_pdn_move("28x19")
        self.board.play_pdn_move("8-11")
        self.board.play_pdn_move("27-24")
        self.board.play_pdn_move("3-8")
        self.board.play_pdn_move("32-27")
        self.board.play_pdn_move("11-15")
        self.board.play_pdn_move("21-17")
        self.board.play_pdn_move("10-14")
        self.board.play_pdn_move("17x10")

    def black_can_make_king_move(self):
        self.board.play_pdn_move("6-10")

    def black_king_can_be_captured(self):
        self.board.play_pdn_move("3-7")

    def black_king_captured(self):
        self.board.play_pdn_move("2x11")

    def white_one_branch_capture(self):
        self.board.play_pdn_move("25-21")
        self.board.play_pdn_move("1-6")
        self.board.play_pdn_move("30-25")
        self.board.play_pdn_move("11-16")
        self.board.play_pdn_move("23-18")

    def white_needs_to_decide_at_branch(self):
        self.board.play_pdn_move("16x23")

    def white_has_king(self):
        self.board.play_pdn_move("23x32")

    def black_can_capture_whites_king(self):
        self.board.play_pdn_move("18x11")
        self.board.play_pdn_move("8x15")
        self.board.play_pdn_move("24-20")
        self.board.play_pdn_move("32-27")

    def no_white_king_anymore(self):
        self.board.play_pdn_move("31x24")

    def black_one_double_capture_and_branch_capture(self):
        self.board.play_pdn_move("4-8")
        self.board.play_pdn_move("21-17")
        self.board.play_pdn_move("8-11")
        self.board.play_pdn_move("25-21")
        self.board.play_pdn_move("9-13")
        self.board.play_pdn_move("29-25")
        self.board.play_pdn_move("6-9")
        self.board.play_pdn_move("26-23")
        self.board.play_pdn_move("15-19")


    def black_needs_to_decide_at_branch(self):
        self.board.play_pdn_move("24x15")


    def black_two_captures_and_king_capture(self):
        self.board.play_pdn_move("15x8")
        self.board.play_pdn_move("9-14")
        self.board.play_pdn_move("8-3")
        self.board.play_pdn_move("5-9")
        self.board.play_pdn_move("3-7")
        self.board.play_pdn_move("14-18")






if __name__ == '__main__':
    unittest.main()