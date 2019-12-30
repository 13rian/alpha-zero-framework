import time

from games.checkers import checkers


board = checkers.CheckersBoard()
board.print()


print("number of pdn-moves: ", len(checkers.PDN_MOVES))
print("number of moves: ", len(checkers.MOVES))



# play a test game to test various moves
board = checkers.CheckersBoard()
board.play_pdn_move("11-15")
board.print()
print(" ")

# one capture after the move played
board.play_pdn_move("24-19")
board.print()
print(" ")
legal_actions = board.legal_actions()
print(legal_actions, checkers.action_to_pdn(legal_actions[0], board.player))

# two captures after the move played
board.play_pdn_move("15x24")
board.print()
print(" ")
legal_actions = board.legal_actions()
print(legal_actions, checkers.action_to_pdn(legal_actions[0], board.player), checkers.action_to_pdn(legal_actions[1], board.player))


board.play_pdn_move("28x19")
board.print()
print(" ")

board.play_pdn_move("8-11")
board.print()
print(" ")

board.play_pdn_move("27-24")
board.print()
print(" ")

board.play_pdn_move("3-8")
board.print()
print(" ")

board.play_pdn_move("32-27")
board.print()
print(" ")

board.play_pdn_move("11-15")
board.print()
print(" ")

board.play_pdn_move("21-17")
board.print()
print(" ")

# after this move is played white will have one capture and black two double captures, both leading to a king at the
# end of the sequence. it is blacks move so black needs to choose between one of the two captures
board.play_pdn_move("10-14")
board.print()
print(" ")
legal_actions = board.legal_actions()
print(legal_actions, checkers.action_to_pdn(legal_actions[0], board.player), checkers.action_to_pdn(legal_actions[1], board.player))


# black should have a king after executing the move
board.play_pdn_move("17x10")
board.print()
print(" ")


# black should have a legal move with the king by now
board.play_pdn_move("6-10")
board.print()
print(" ")
legal_actions = board.legal_actions()
print(legal_actions)


# black moves with the king so that white can capture it, white has only one capture
board.play_pdn_move("3-7")
board.print()
print(" ")
legal_actions = board.legal_actions()
print(legal_actions)


# black king should be captured after this move
board.play_pdn_move("2x11")
board.print()
print(" ")

board.play_pdn_move("25-21")
board.print()
print(" ")

board.play_pdn_move("1-6")
board.print()
print(" ")

board.play_pdn_move("30-25")
board.print()
print(" ")

board.play_pdn_move("11-16")
board.print()
print(" ")


# after this move white has one capture that branches
board.play_pdn_move("23-18")
board.print()
print(" ")
legal_actions = board.legal_actions()
print(legal_actions, checkers.action_to_pdn(legal_actions[0], board.player))

# after executing the capture it is again white's move as there are two possible capture paths
board.play_pdn_move("16x23")
board.print()
print(" ")
print("player: ", board.player)
legal_actions = board.legal_actions()
print(legal_actions, checkers.action_to_pdn(legal_actions[0], board.player), checkers.action_to_pdn(legal_actions[1], board.player))


# white executes one of the captures and gets a king
board.play_pdn_move("23x32")
board.print()
print(" ")


board.play_pdn_move("18x11")
board.print()
print(" ")

board.play_pdn_move("8x15")
board.print()
print(" ")


board.play_pdn_move("24-20")
board.print()
print(" ")


# black has now the opportunity to capture white's king
board.play_pdn_move("32-27")
board.print()
print(" ")
legal_actions = board.legal_actions()
print(legal_actions, checkers.action_to_pdn(legal_actions[0], board.player))


# after this move the white king should be gone from the board
board.play_pdn_move("31x24")
board.print()
print(" ")


board.play_pdn_move("4-8")
board.print()
print(" ")


board.play_pdn_move("21-17")
board.print()
print(" ")


board.play_pdn_move("8-11")
board.print()
print(" ")


board.play_pdn_move("25-21")
board.print()
print(" ")

board.play_pdn_move("9-13")
board.print()
print(" ")


board.play_pdn_move("29-25")
board.print()
print(" ")


board.play_pdn_move("6-9")
board.print()
print(" ")

board.play_pdn_move("26-23")
board.print()
print(" ")


# black has one double capture and one capture that branches
board.play_pdn_move("15-19")
board.print()
print(" ")
legal_actions = board.legal_actions()
print(legal_actions, checkers.action_to_pdn(legal_actions[0], board.player), checkers.action_to_pdn(legal_actions[1], board.player))


# make the start of the branching capture and it should still be black's move with two captures to execute
board.play_pdn_move("24x15")
board.print()
print(" ")
print("player: ", board.player)
legal_actions = board.legal_actions()
print(legal_actions, checkers.action_to_pdn(legal_actions[0], board.player), checkers.action_to_pdn(legal_actions[1], board.player))


# execute the second capture and it is white's move again
board.play_pdn_move("15x8")
board.print()
print(" ")
print("player: ", board.player)


board.play_pdn_move("9-14")
board.print()
print(" ")

# black makes a king
board.play_pdn_move("8-3")
board.print()
print(" ")


board.play_pdn_move("5-9")
board.print()
print(" ")


board.play_pdn_move("3-7")
board.print()
print(" ")


# after this move black has the opportunity to capture a piece with the king and two other captures as well
board.play_pdn_move("14-18")
board.print()
print(" ")
legal_actions = board.legal_actions()
print(legal_actions, checkers.action_to_pdn(legal_actions[0], board.player), checkers.action_to_pdn(legal_actions[1], board.player), checkers.action_to_pdn(legal_actions[2], board.player))


# execute the double capture with the king
board.play_pdn_move("7x14")
board.print()
print(" ")




# play a few random games to test the performance
n_games = 1000
start = time.time()
for i in range(n_games):
    board = checkers.CheckersBoard()
    while not board.terminal:
        move = board.random_action()
        board.execute_action(move)
    # if board.score == 0:
    #     board.print()

print("time for one game: ", (time.time() - start) / n_games)
