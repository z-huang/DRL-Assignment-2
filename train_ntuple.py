import copy
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle
import os
from game2048 import Game2048Env

WIN_SIZE = 100
CHECKPOINT_PATH = 'td-8x6.pkl'
HISTORY_PATH = 'history_8x6.pkl'
GRAPH_PATH = 'reward_8x6.png'


# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------
def rot90(pattern, board_size):
    return [(y, board_size - 1 - x) for x, y in pattern]

def horizontal_flip(pattern, board_size):
    return [(board_size - 1 - x, y) for x, y in pattern]

def has_multiple_max_values(arr):
    max_value = np.max(arr)
    max_count = np.sum(arr == max_value)
    return max_count > 1

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
        return symmetries# return symmetries
        # return list(set(tuple(sorted(p)) for p in symmetries))

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            # return int(math.log(tile, 2))
            return tile.item().bit_length() - 1

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
                temp_env = copy.deepcopy(env)
                next_state, _, _, _ = temp_env.do_action(a)
                v = temp_env.score + self.value(next_state)
            values.append(v)
        action = np.argmax(values)
        return action

    def save_weights(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)

    def load_weights(self, path):
        with open(path, 'rb') as f:
            self.weights = pickle.load(f)


def td_learning(
    env: Game2048Env,
    approximator: NTupleApproximator,
    num_episodes=50000,
    alpha=0.01,
    gamma=0.99,
    epsilon=0.1,
):
    """
    Trains the 2048 agent using TD-Learning.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
    """
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, 'rb') as f:
            final_scores = pickle.load(f)
    else:
        final_scores = []
    success_flags = []

    with tqdm(range(num_episodes)) as pbar:
        for episode in pbar:
            state = env.reset()
            trajectory = []  # Store trajectory data if needed
            previous_score = 0
            done = False
            max_tile = np.max(state)

            while not done:
                legal_moves = [a for a in range(4) if env.is_move_legal(a)]
                if not legal_moves:
                    break
                # TODO: action selection
                # Note: TD learning works fine on 2048 without explicit exploration, but you can still try some exploration methods.
                if random.random() < epsilon:
                    action = random.choice(legal_moves)
                else:
                    action = approximator.get_best_action(env)

                s1, _, _, _ = env.do_action(action)
                s2, _, done, _ = env.spawn_tile()
                reward = env.score - previous_score
                max_tile = max(max_tile, np.max(s1))

                # TODO: Store trajectory or just update depending on the implementation
                trajectory.append((s1, reward))

                state = s2
                previous_score = env.score

            # TODO: If you are storing the trajectory, consider updating it now depending on your implementation.
            for (s1, _), (s2, r) in reversed(list(zip(trajectory[:-1], trajectory[1:]))):
                delta = r + gamma * approximator.value(s2) - approximator.value(s1)
                approximator.update(s1, delta, alpha)
    
            final_scores.append(env.score)
            success_flags.append(1 if max_tile >= 2048 else 0)

            pbar.set_postfix({
                'avg score': int(np.mean(final_scores[-WIN_SIZE:])),
                'success rate': np.sum(success_flags[-WIN_SIZE:]) / WIN_SIZE,
                'max_tile': max_tile
            })

            if (episode + 1) % 100 == 0:
                approximator.save_weights(CHECKPOINT_PATH)
                with open(HISTORY_PATH, 'wb') as f:
                    pickle.dump(final_scores, f)
                moving_avg = [np.mean(final_scores[i:i+WIN_SIZE])
                              for i in range(len(final_scores) - WIN_SIZE + 1)]
                plt.plot(final_scores, alpha=0.5)
                plt.plot(range(WIN_SIZE - 1, len(final_scores)),
                         moving_avg,
                         label='Moving Average',
                         color='red')
                plt.xlabel('Episodes')
                plt.ylabel('Score')
                plt.title('TD Learning')
                plt.savefig(GRAPH_PATH)
                plt.clf()

    return final_scores


# TODO: Define your own n-tuple patterns
patterns = [
    # 5x6-tuple
    # [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    # [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    # [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
    # [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
    # [(2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1)],

    # 8x6 tuple
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)],

    # 4-tuple
    # [(0, 0), (0, 1), (0, 2), (0, 3)],
    # [(1, 0), (1, 1), (1, 2), (1, 3)],
    # [(0, 0), (1, 0), (2, 0), (3, 0)],
    # [(0, 1), (1, 1), (2, 1), (3, 1)],
    # [(0, 0), (0, 1), (1, 0), (1, 1)],
    # [(0, 1), (0, 2), (1, 1), (1, 2)],
    # [(1, 1), (1, 2), (2, 1), (2, 2)],
]

approximator = NTupleApproximator(board_size=4, patterns=patterns)
env = Game2048Env()

if os.path.exists(CHECKPOINT_PATH):
    approximator.load_weights(CHECKPOINT_PATH)
final_scores = td_learning(env, approximator, num_episodes=100000, alpha=0.05, gamma=1.0, epsilon=0.0)
approximator.save_weights(CHECKPOINT_PATH)
