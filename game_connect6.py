import random
from typing import List, Tuple
import numpy as np
import gym
from gym import spaces


class Connect6Env(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, board_size=19, neighbor_radius=2):
        super().__init__()
        self.size = board_size
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False

        self.stones_per_turn = 1
        self.stones_played_in_turn = 0

        self.last_move = None
        self.winner = None

        self.neighbor_radius = neighbor_radius
        self.active_area = set()

        # Action space: place a stone at (row, col)
        self.action_space = spaces.MultiDiscrete([self.size, self.size])
        # Observation space: board state
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.size, self.size), dtype=np.int8)

    def reset(self):
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        self.stones_per_turn = 1
        self.stones_played_in_turn = 0
        self.last_move = None
        self.winner = None
        self.active_area.clear()
        return self.board.copy(), {}

    def clone(self):
        env = Connect6Env(self.size, self.neighbor_radius)
        env.board = self.board.copy()
        env.turn = self.turn
        env.game_over = self.game_over
        env.stones_per_turn = self.stones_per_turn
        env.stones_played_in_turn = self.stones_played_in_turn
        env.last_move = self.last_move
        env.winner = self.winner
        env.active_area = self.active_area.copy()
        return env

    def step(self, action):
        """action: (row, col) tuple representing one stone placement"""
        if self.game_over:
            return self.board.copy(), 0.0, True, False, {"error": "Game over"}

        row, col = action
        if not (0 <= row < self.size and 0 <= col < self.size):
            return self.board.copy(), -1.0, True, False, {"error": "Out of bounds"}
        if self.board[row, col] != 0:
            return self.board.copy(), -1.0, True, False, {"error": "Position occupied"}

        self.board[row, col] = self.turn
        self.stones_played_in_turn += 1

        radius = self.neighbor_radius
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = row + dr, col + dc
                if (0 <= nr < self.size and
                    0 <= nc < self.size and
                        self.board[nr, nc] == 0):
                    self.active_area.add((nr, nc))
        self.active_area.discard((row, col))

        self.winner = self.check_win()
        if self.winner:
            self.game_over = True
            reward = 1.0 if self.winner == self.turn else -1.0
            return self.board.copy(), reward, True, False, {"winner": self.winner}

        if self.stones_played_in_turn >= self.stones_per_turn:
            self.turn = 3 - self.turn
            self.stones_played_in_turn = 0
            self.stones_per_turn = 2

        self.last_move = action

        return self.board.copy(), 0.0, False, False, {}

    @property
    def next_turn(self):
        if self.stones_played_in_turn + 1 >= self.stones_per_turn:
            return 3 - self.turn
        else:
            return self.turn

    def check_win(self, return_coords=False):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 水平、垂直、對角線、反對角線
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        coords = [(rr, cc)]
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                            coords.append((rr, cc))
                        if count >= 6:
                            if return_coords:
                                # 去掉最後一個超出邊界的座標
                                return current_color, coords[:-1]
                            else:
                                return current_color
        if return_coords:
            return 0, []
        else:
            return 0

    def render(self, action=None):
        _, win_coords = self.check_win(return_coords=True)

        print("\n  Turn: ", "Black" if self.turn == 1 else "White")
        print("  Stones this turn: ", self.stones_played_in_turn,
              "/", self.stones_per_turn)
        print("   " + " ".join(self.index_to_label(i)
              for i in range(self.size)))

        win_set = set(win_coords) if win_coords else set()

        for r in range(self.size - 1, -1, -1):
            row_str = f"{r+1:2} "
            for c in range(self.size):
                piece = self.board[r, c]
                coord = (r, c)
                if piece == 0:
                    mark = "."
                elif piece == 1:
                    color = "\033[1;93m" if coord in win_set or coord == action else "\033[0;31m"
                    mark = f"{color}X\033[0m"
                elif piece == 2:
                    color = "\033[1;93m" if coord in win_set or coord == action else "\033[0;34m"
                    mark = f"{color}O\033[0m"
                row_str += f"{mark} "
            print(row_str)
        print()

    def index_to_label(self, col):
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skip 'I'

    @property
    def empty_positions(self) -> List[Tuple[int, int]]:
        return list(zip(*np.where(self.board == 0)))

    def get_neighboring_empty_positions(self, radius) -> List[Tuple[int, int]]:
        return list(self.active_area)
        candidates = set()

        stones = np.argwhere(self.board != 0)

        for r, c in stones:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.size and
                        0 <= nc < self.size and
                            self.board[nr, nc] == 0):
                        candidates.add((nr, nc))

        return list(candidates)

    def max_line_length(self, player) -> int:
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        max_len = 0

        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != player:
                    continue
                for dr, dc in directions:
                    # 確保是這條線的起點
                    prev_r, prev_c = r - dr, c - dc
                    if (0 <= prev_r < self.size and
                        0 <= prev_c < self.size and
                            self.board[prev_r, prev_c] == player):
                        continue

                    # 計算這條線的連子數量
                    count = 0
                    nr, nc = r, c
                    for _ in range(6):
                        if (0 <= nr < self.size and
                            0 <= nc < self.size and
                                self.board[nr, nc] == player):
                            count += 1
                            nr += dr
                            nc += dc
                        else:
                            break

                    # 往兩邊擴展，看總共可用格數（最多 6）
                    extendable = count

                    # 往後延伸
                    tr, tc = nr, nc
                    for _ in range(6 - count):
                        if (0 <= tr < self.size and
                            0 <= tc < self.size and
                                self.board[tr, tc] in (0, player)):
                            extendable += 1
                            tr += dr
                            tc += dc
                        else:
                            break

                    # 往前延伸
                    tr, tc = r - dr, c - dc
                    for _ in range(6 - count):
                        if (0 <= tr < self.size and
                            0 <= tc < self.size and
                                self.board[tr, tc] in (0, player)):
                            extendable += 1
                            tr -= dr
                            tc -= dc
                        else:
                            break

                    # 如果整條線有機會成為六連，才計入 max_len
                    if extendable >= 6:
                        max_len = max(max_len, count)

        return max_len


if __name__ == '__main__':
    env = Connect6Env()
    obs, _ = env.reset()

    done = False
    while not done:
        env.render()
        action = random.choice(env.empty_positions)
        obs, reward, done, _, info = env.step(action)

    env.render()
