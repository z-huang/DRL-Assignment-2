
from collections import defaultdict
import copy
import math
import pickle
import numpy as np


def rot90(pattern, board_size):
    return [(y, board_size - 1 - x) for x, y in pattern]


def horizontal_flip(pattern, board_size):
    return [(board_size - 1 - x, y) for x, y in pattern]


class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        symmetries = []
        for _ in range(4):
            symmetries.append(pattern)
            symmetries.append(horizontal_flip(pattern, self.board_size))
            pattern = rot90(pattern, self.board_size)
        return symmetries

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        values = []
        for i, sym_group in enumerate(self.symmetry_patterns):
            for pattern in sym_group:
                feature = self.get_feature(board, pattern)
                values.append(self.weights[i][feature])
        return np.mean(values)

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for i, sym_group in enumerate(self.symmetry_patterns):
            for pattern in sym_group:
                feature = self.get_feature(board, pattern)
                self.weights[i][feature] += alpha * delta

    def get_best_action(self, env):
        values = []
        for a in range(4):
            if not env.is_move_legal(a):
                v = -1
            else:
                temp_env = env.clone()
                next_state, new_score, _, _ = temp_env.do_action(a)
                v = new_score + self.value(next_state)
            values.append(v)
        action = np.argmax(values)
        return action

    def save_weights(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)

    def load_weights(self, path):
        with open(path, 'rb') as f:
            self.weights = pickle.load(f)
