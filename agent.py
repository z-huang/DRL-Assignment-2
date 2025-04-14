
import random
import time
from typing import Literal

import numpy as np
from game_connect6 import Connect6Env
from mcts_6 import MCTS, MCTSAgent


class RandomAgent:
    def get_action(self, env: Connect6Env):
        if env.last_move:
            last_r, last_c = env.last_move
            potential_moves = [
                (r, c)
                for r in range(max(0, last_r - 2), min(env.size, last_r + 3))
                for c in range(max(0, last_c - 2), min(env.size, last_c + 3))
                if env.board[r, c] == 0
            ]
        else:
            potential_moves = [
                (r, c)
                for r in range(env.size)
                for c in range(env.size)
                if env.board[r, c] == 0
            ]

        return random.choice(potential_moves)


class RuleBasedAgent:
    def __init__(
        self,
        strength: Literal[None, 'weak', 'slightly_weak'] = None
    ):
        self.strength = strength

    def get_action(self, env: Connect6Env):
        env = env.clone()

        if np.count_nonzero(env.board) == 0:
            middle = env.size // 2
            return (middle, middle + 1)

        my_color = env.turn
        opponent_color = 3 - my_color
        empty_positions = env.empty_positions

        if self.strength == 'weak':
            if random.uniform(0, 1) < 0.2:
                return random.choice(empty_positions)

        # 1. Winning move
        for r, c in empty_positions:
            env.board[r, c] = my_color
            if env.check_win() == my_color:
                env.board[r, c] = 0
                return (r, c)
            env.board[r, c] = 0

        # 2. Block opponent's winning move
        for r, c in empty_positions:
            env.board[r, c] = opponent_color
            if env.check_win() == opponent_color:
                env.board[r, c] = 0
                return (r, c)
            env.board[r, c] = 0

        if self.strength == 'slightly_weak':
            if random.uniform(0, 1) < 0.15:
                return random.choice(empty_positions)

        # 3. Attack: prioritize strong formations
        best_move = None
        best_score = 0
        for r, c in empty_positions:
            score = self.evaluate_position(r, c, my_color, env)
            if score > best_score:
                best_score = score
                best_move = (r, c)

        # 4. Defense: prevent opponent from forming strong positions
        for r, c in empty_positions:
            opponent_score = self.evaluate_position(r, c, opponent_color, env)
            if opponent_score > best_score:
                best_score = opponent_score
                best_move = (r, c)

        # 5. Execute best move
        if best_move:
            return best_move

        # 6. Default move: play near last opponent move
        if env.last_move:
            last_r, last_c = env.last_move
            potential_moves = [
                (r, c)
                for r in range(max(0, last_r - 2), min(env.size, last_r + 3))
                for c in range(max(0, last_c - 2), min(env.size, last_c + 3))
                if env.board[r, c] == 0
            ]
            if potential_moves:
                return random.choice(potential_moves)

        # 7. Random move as fallback
        return random.choice(empty_positions)

    def evaluate_position(self, r, c, color, env: Connect6Env):
        """Evaluates the strength of a position based on alignment potential."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        score = 0

        for dr, dc in directions:
            count = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < env.size and 0 <= cc < env.size and env.board[rr, cc] == color:
                count += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while 0 <= rr < env.size and 0 <= cc < env.size and env.board[rr, cc] == color:
                count += 1
                rr -= dr
                cc -= dc

            if count >= 5:
                score += 10000
            elif count == 4:
                score += 5000
            elif self.strength == 'slightly_weak':
                if count == 3:
                    score += 1000
                elif count == 2:
                    score += 100

        return score


if __name__ == '__main__':
    env = Connect6Env()
    win_count = 0
    n_episodes = 10

    for episode in range(n_episodes):
        env.reset()
        agent1 = MCTSAgent(
            mcts=MCTS(
                c_puct=1.41 * 10,
                rollout_depth=6
            ),
            iterations=80
        )
        # agent2 = RandomAgent()
        agent2 = RuleBasedAgent('slightly_weak')

        while not env.game_over:
            if env.turn == 1:
                action = agent1.get_action(env)
            else:
                action = agent2.get_action(env)
            env.step(action)
            env.render(action)
            print(win_count, '/', episode)

        if env.winner == 1:
            win_count += 1

        time.sleep(1)

    print(win_count, '/', n_episodes, win_count / n_episodes)
