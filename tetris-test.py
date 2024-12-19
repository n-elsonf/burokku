from tetris import Tetris
import unittest

# # Test 4 Row Clears
# game = Tetris()
# game.set_curr(6)
# game.play(1, True, delay=0.0001)
# game.set_curr(6)
# game.play(3, True, delay=0.0001)
# game.set_curr(6)
# game.play(5, True, delay=0.0001)
# game.set_curr(6)
# game.play(7, True, delay=0.0001)
# game.set_curr(6)
# game.play(1, True, delay=0.0001)
# game.set_curr(6)
# game.play(3, True, delay=0.0001)
# game.set_curr(6)
# game.play(5, True, delay=0.0001)
# game.set_curr(6)
# game.play(7, True, delay=0.0001)
# game.set_curr(5)
# game.play(8, True, delay=0.0001)
# game.set_curr(5)
# game.play(9, True, delay=0.0001)


class TestClass(unittest.TestCase):
    board1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
              [0, 1, 1, 1, 1, 1, 1, 0, 0, 1],
              [0, 1, 1, 0, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    
    board2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]]

    def test_calculate_bumpiness(self):
        game = Tetris()
        game.set_board(TestClass.board1)
        board = game.board
        self.assertEqual(6, Tetris.calculate_bumpiness(game, board))

    def test_calculate_aggregated_height(self):
        game = Tetris()
        game.set_board(TestClass.board1)
        board = game.board
        self.assertEqual(48, Tetris.calculate_aggregated_height(game, board))

    def test_calculate_holes(self):
        game = Tetris()
        game.set_board(TestClass.board1)
        board = game.board
        self.assertEqual(2, Tetris.calculate_holes(game, board))

    def test_clean_rows(self):
        game = Tetris()
        game.set_board(TestClass.board1)
        board = game.board
        rows_cleaned, _ = Tetris.clean_rows(game, board)
        self.assertEqual(2, rows_cleaned)
    
    def test_get_board_properties(self):
        game = Tetris()
        game.set_curr(0)
        game.rotate_piece(1)
        game.set_board(TestClass.board2)
        game.play(8)
        board =  game.board
        self.assertEqual([4, 0, 0, 0], Tetris.get_board_properties(game, board))
        


if __name__ == "__main__":
    unittest.main()
