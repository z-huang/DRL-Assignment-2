import gym
from gym import spaces
import random
import numpy as np


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True

        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board.copy()

    @classmethod
    def restore(cls, board, score):
        env = cls()
        env.board = board.copy()
        env.score = score
        return env

    def clone(self):
        return Game2048Env.restore(self.board, self.score)

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        new_row = row[row != 0]
        # new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        new_row = np.concatenate((new_row, np.zeros(self.size - len(new_row), dtype=int)))
        return new_row

    def merge(self, row):
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board.copy(), self.score, done, {}

    def do_action(self, action):
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            self.move_up()
        elif action == 1:
            self.move_down()
        elif action == 2:
            self.move_left()
        elif action == 3:
            self.move_right()

        done = self.is_game_over()

        return self.board.copy(), self.score, done, {}

    def spawn_tile(self):
        self.add_random_tile()
        done = self.is_game_over()

        return self.board.copy(), self.score, done, {}

    def simulate_row_move(self, row):
        new_row = row[row != 0]
        # new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        new_row = np.concatenate((new_row, np.zeros(self.size - len(new_row), dtype=int)))
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        new_row = new_row[new_row != 0]
        # new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        new_row = np.concatenate((new_row, np.zeros(self.size - len(new_row), dtype=int)))
        return new_row

    def is_move_legal(self, action):
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")
        return not np.array_equal(self.board, temp_board)

    def render(self):
        color_map = {
            0:    "\033[48;5;231m\033[30m",   # 白底 + 黑字
            2:    "\033[48;5;230m\033[30m",   # 淺米色
            4:    "\033[48;5;223m\033[30m",   # 淺橙
            8:    "\033[48;5;209m\033[30m",   # 橘
            16:   "\033[48;5;208m\033[30m",   # 深橘
            32:   "\033[48;5;202m\033[30m",   # 紅橘
            64:   "\033[48;5;196m\033[97m",   # 紅 + 白字
            128:  "\033[48;5;220m\033[30m",   # 金黃
            256:  "\033[48;5;214m\033[30m",
            512:  "\033[48;5;208m\033[97m",
            1024: "\033[48;5;226m\033[30m",
            2048: "\033[48;5;190m\033[30m",
        }
        reset = "\033[0m"

        for row in self.board:
            for x in row:
                color = color_map.get(x, "\033[48;5;240m\033[97m")  # default: 灰底 + 白字
                value_str = f"{x:5d}" if x != 0 else "     "
                print(f"{color} {value_str} {reset}", end='')
            print()
        print()
        print('Score:', self.score)