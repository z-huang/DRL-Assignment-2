import math
from typing import Any, Tuple

import numpy as np
from tqdm import tqdm
from game_connect6 import Connect6Env


def evaluate_position(env: Connect6Env, r, c, player):
    """Evaluate both attack and defense for a given move, considering extendable lines."""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    score = 1
    opponent = 3 - player
    stones_left = env.stones_per_turn - env.stones_played_in_turn

    def evaluate_for(p):
        yield 1  # ensure not empty
        for dr, dc in directions:
            count = 1
            extendable = 1

            # Forward
            rr, cc = r + dr, c + dc
            while (0 <= rr < env.size and
                   0 <= cc < env.size and
                   env.board[rr, cc] == p):
                count += 1
                extendable += 1
                rr += dr
                cc += dc
            while (0 <= rr < env.size and
                   0 <= cc < env.size and
                   env.board[rr, cc] in (0, p) and
                   extendable < 6):
                extendable += 1
                rr += dr
                cc += dc

            # Backward
            rr, cc = r - dr, c - dc
            while (0 <= rr < env.size and
                   0 <= cc < env.size and
                   env.board[rr, cc] == p):
                count += 1
                extendable += 1
                rr -= dr
                cc -= dc
            while (0 <= rr < env.size and
                   0 <= cc < env.size and
                   env.board[rr, cc] in (0, p) and
                   extendable < 6):
                extendable += 1
                rr -= dr
                cc -= dc

            if extendable >= 6:
                yield count

    env.board[r, c] = player
    my_count = max(evaluate_for(player))
    env.board[r, c] = opponent
    oppo_count = max(evaluate_for(opponent))
    env.board[r, c] = 0

    if my_count == 6:
        return 20000
    elif my_count == 5 and stones_left == 2:
        return 20000
    elif oppo_count == 6:
        return 10000
    elif oppo_count == 5 and stones_left == 1:
        return 10000

    if my_count == 5:
        score += 5000
    elif my_count == 4:
        score += 500
    elif my_count == 3:
        score += 100
    elif my_count == 2:
        score += 10

    if oppo_count == 4:
        score += 1000
    elif oppo_count == 3:
        score += 100

    return score


def evaluate_position(env: Connect6Env, r, c, player):
    """Evaluate both attack and defense for a given move, considering extendable lines."""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    score = 1
    opponent = 3 - player
    stones_left = env.stones_per_turn - env.stones_played_in_turn

    def evaluate_for(p):
        for dr, dc in directions:
            count = 1
            extendable = 1

            # Forward
            rr, cc = r + dr, c + dc
            while (0 <= rr < env.size and
                   0 <= cc < env.size and
                   env.board[rr, cc] == p):
                count += 1
                extendable += 1
                rr += dr
                cc += dc
            while (0 <= rr < env.size and
                   0 <= cc < env.size and
                   env.board[rr, cc] in (0, p) and
                   extendable < 6):
                extendable += 1
                rr += dr
                cc += dc

            # Backward
            rr, cc = r - dr, c - dc
            while (0 <= rr < env.size and
                   0 <= cc < env.size and
                   env.board[rr, cc] == p):
                count += 1
                extendable += 1
                rr -= dr
                cc -= dc
            while (0 <= rr < env.size and
                   0 <= cc < env.size and
                   env.board[rr, cc] in (0, p) and
                   extendable < 6):
                extendable += 1
                rr -= dr
                cc -= dc

            if extendable >= 6:
                yield count

    env.board[r, c] = player
    for count in evaluate_for(player):
        if count >= 6:
            score += 20000
        elif count == 5:
            score += 5000
        elif count == 4:
            score += 500
        elif count == 3:
            score += 100
        elif count == 2:
            score += 10

    env.board[r, c] = opponent
    for count in evaluate_for(opponent):
        if count >= 6:
            score += 20000  # very urgent to block!
        elif count == 5:
            score += 10000
        elif count == 4:
            score += 1000
        elif count == 3:
            score += 100

    env.board[r, c] = 0
    return score


def score_fn(env: Connect6Env, player):
    if env.winner == player:
        return 10000
    if env.winner == 3 - player:
        return -10000

    max_line_length = env.max_line_length(player)
    if max_line_length == 5:
        return 1000
    elif max_line_length == 4:
        return 400
    elif max_line_length == 3:
        return 100
    elif max_line_length == 2:
        return 30
    elif max_line_length == 1:
        return 10
    else:
        return 0


def rollout_policy(env: Connect6Env):
    candidate_moves = env.get_neighboring_empty_positions(radius=2)
    scores = []

    for move in candidate_moves:
        r, c = move
        score = evaluate_position(env, r, c, env.turn)
        scores.append(score)

    return {move: score for move, score in zip(candidate_moves, scores)}


def policy(env: Connect6Env):
    candidate_moves = env.get_neighboring_empty_positions(radius=2)
    # return {move: 1/len(candidate_moves) for move in candidate_moves}
    scores = []

    for move in candidate_moves:
        r, c = move
        score = evaluate_position(env, r, c, env.turn)
        scores.append(score)

    total_score = sum(scores)
    return {move: score / total_score for move, score in zip(candidate_moves, scores)}


def value_fn(env: Connect6Env, player):
    my_score = score_fn(env, player)
    opp_score = score_fn(env, 3 - player)

    return my_score - opp_score


class Node:
    def __init__(
        self,
        player,
        parent=None,
        action=None,
        prior=0.0,
    ):
        self.player = player
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children: dict[Any, Node] = {}
        self.visits = 0
        self.total_reward = 0.0

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def Q(self):
        return self.total_reward / (self.visits + 1e-6)

    def __repr__(self):
        return f'Node(action={self.action}, prior={self.prior}, q={self.Q}, is_leaf={self.is_leaf})'


class MCTS:
    def __init__(
        self,
        policy=policy,
        rollout_policy=rollout_policy,
        value_fn=value_fn,
        c_puct=1.41,
        rollout_depth=100,
    ):
        self.policy = policy
        self.rollout_policy = rollout_policy
        self.value_fn = value_fn
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth

    def select_child(self, node: Node):
        # PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        # where Q(s,a) = child.total_reward / child.visits.
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            Q = child.total_reward / (child.visits + 1e-6)
            U = self.c_puct * child.prior * \
                math.sqrt(node.visits) / (1 + child.visits)
            score = Q + U

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def rollout(self, env: Connect6Env, depth):
        player = env.turn
        for _ in range(depth):
            if env.game_over:
                break
            action_probs = self.rollout_policy(env)
            # action = random.choices(
            #     list(action_probs.keys()),
            #     weights=list(action_probs.values()), k=1
            # )[0]
            action = max(action_probs, key=lambda a: action_probs[a])
            env.step(action)

        return self.value_fn(env, player)

    def backpropagate(self, node: Node, reward):
        # TODO: Propagate the reward up the tree, updating visit counts and total rewards.
        player = node.player
        while node is not None:
            node.visits += 1
            if node.player == player:
                node.total_reward += reward
            else:
                node.total_reward -= reward
            node = node.parent

    def expand(self, node: Node, env: Connect6Env):
        action_probs = self.policy(env)
        for action, prob in action_probs.items():
            if action not in node.children:
                node.children[action] = Node(
                    action=action,
                    player=env.turn,
                    prior=prob,
                    parent=node
                )

    def run_simulation(self, root: Node, env: Connect6Env):
        env = env.clone()

        node = root

        # Selection
        while not node.is_leaf:
            node = self.select_child(node)
            env.step(node.action)

        # Expansion
        self.expand(node, env)

        # Rollout
        reward = self.rollout(env, self.rollout_depth)

        # Backpropagation
        self.backpropagate(node, reward)

    def best_action(self, root: Node):
        visit_dist = {
            action: child.visits / root.visits
            for action, child in root.children.items()
        }
        return max(visit_dist, key=lambda a: visit_dist[a]), visit_dist


class MCTSAgent:
    def __init__(self, mcts: MCTS, iterations: int):
        self.mcts = mcts
        self.iterations = iterations

    def get_action(self, env: Connect6Env) -> Tuple[int, int]:
        if np.count_nonzero(env.board) == 0:
            middle = env.size // 2
            return (middle, middle + 1)

        root = Node(player=env.turn)
        threshold = self.iterations / 2

        for i in tqdm(range(self.iterations)):
            self.mcts.run_simulation(root, env)
            if i + 1 >= threshold and root.children:
                max_visited_node = max(
                    root.children.values(),
                    key=lambda x: x.visits
                )
                if max_visited_node.visits > self.iterations / 2:
                    break

        action, visit_dist = self.mcts.best_action(root)
        return action
