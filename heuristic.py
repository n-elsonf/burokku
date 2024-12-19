from tetris import Tetris
import numpy as np


class TetrisAI:

    def __init__(self, game):
        self.game = game

    def evaluate_board(self):
        ''' Evaluates the current board state using a weighted heuristic function '''

        board = self.game.board

        agg_height = self.game.calculate_aggregated_height(board)
        holes = self.game.calculate_holes(board)
        bumpiness = self.game.calculate_bumpiness(board)
        cleared, _= self.game.clean_rows(board)

        # Heuristic weights
        score = (-0.5 * agg_height) + (-0.7 * holes) + \
            (-0.3 * bumpiness) + (1.0 * cleared)

        return score

    def best_move(self):
        '''
        Finds the best move by performing the heuristic search of one shape with all four rotations. Place shape in every move possible and compute the best score out of every moves"
        '''

        best_score = float('-inf')
        best_position = None
        best_rotation = 0

        # Itearate all 4 rotations including 0 rotation
        for rotation in range(4):

            rotated_piece = np.rot90(self.game.curr_piece, rotation)

            # Place piece in every column and calculate the board state

            for col in range(Tetris.BOARD_WIDTH):
                # Init temporary board as default before evaluating
                temp_board = self.game.board.copy()
                pos = [col, 0]

                # Drop piece until collision
                while not self.game.check_collision(rotated_piece, pos):
                    pos[1] += 1
                pos[1] -= 1  # Move back up after the collision

                # Check if final position is valid
                if not self.game.check_collision(rotated_piece, pos):
                    self.game.board = self.game.add_piece(rotated_piece, pos)

                    # Evaluate the board state after adding the piece
                    score = self.evaluate_board()

                    # Reset board state
                    self.game.board = temp_board

                    # If this move results in a better score, update the best move
                    if score > best_score:
                        best_score = score
                        best_position = (col, pos[1])
                        best_rotation = rotation

        return best_position, best_rotation

    def ai_play(self, render=False, delay=None):
        ''' Executes the AI’s best move '''
        best_position, best_rotation = self.best_move()

        # Rotate the piece to the best rotation
        self.game.rotate_piece(best_rotation)

        # Move the piece to the best position
        x_pos, _ = best_position
        self.game.play(x_pos, True)


# Run AI
if __name__ == "__main__":
    game = Tetris()

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
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1]]
    
    game.set_board = board1
    ai = TetrisAI(game)

    while not game.game_over:
        ai.ai_play(render=False)
    print("Game Over. Final Score:", game.score)
