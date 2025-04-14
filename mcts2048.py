import copy
import math
import random
from typing import Any

import numpy as np
from game2048 import Game2048Env


def pop_random_element(lst):
    """Pop and return a random element from the list."""
    if not lst:
        raise IndexError("Cannot pop from an empty list.")
    idx = random.randrange(len(lst))
    return lst.pop(idx)


class TD_MCTS_Node:
    def __init__(self, state: np.ndarray, score, parent=None, action=None, is_chance_node=False):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children: dict[Any, TD_MCTS_Node] = {}
        self.visits = 0
        self.total_reward = 0.0
        self.is_chance_node = is_chance_node
        # List of untried actions based on the current state's legal moves
        if is_chance_node:
            self.untried_actions = []
            self.empty_cells = list(zip(*np.where(state == 0)))
        else:
            env = Game2048Env.restore(state, score)
            self.untried_actions = [a for a in range(4)
                                    if env.is_move_legal(a)]
            self.empty_cells = []

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


class TD_MCTS:
    def __init__(
        self,
        approximator,
        iterations=500,
        exploration_constant=1.41,
        rollout_depth=10,
        v_norm=80000,
    ):
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.v_norm = v_norm

    def select_child(self, node):
        best_child = None
        best_score = float('-inf')

        for child in node.children.values():
            exploit = (child.total_reward / child.visits)
            explore = self.c * \
                math.sqrt(math.log(node.visits + 1) / child.visits)
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def rollout(self, env: Game2048Env, depth):
        for _ in range(depth):
            if env.is_game_over():
                break
            legal_actions = [a for a in range(4) if env.is_move_legal(a)]
            action = random.choice(legal_actions)
            env.step(action)

        v = env.score

        if not env.is_game_over():
            for a in range(4):
                if not env.is_move_legal(a):
                    continue
                temp_env = env.clone()
                after_state, score, _, _ = temp_env.do_action(a)
                v = max(v, score + self.approximator.value(after_state))

        return v

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def expand_normal_state(self, node):
        action = node.untried_actions.pop()
        sim_env = Game2048Env.restore(node.state, node.score)
        after_state, new_score, _, _ = sim_env.do_action(action)
        after_node = TD_MCTS_Node(
            state=after_state,
            score=new_score,
            parent=node,
            action=action,
            is_chance_node=True
        )
        node.children[action] = after_node
        return after_node

    def expand_afterstate(self, node):
        if not node.empty_cells:
            return node

        i, j = random.choice(node.empty_cells)
        s = node.state.copy()
        s[i][j] = 2 if random.random() < 0.9 else 4
        key = (i, j, s[i][j])
        if key not in node.children:
            child_node = TD_MCTS_Node(
                state=s,
                score=node.score,
                parent=node
            )
            node.children[key] = child_node

        return node.children[key]

    def run_simulation(self, root: TD_MCTS_Node):
        node = root

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            node = self.expand_afterstate(node)

        # TODO: Expansion: If the node is not terminal, expand an untried action.
        if node.untried_actions:
            node = self.expand_normal_state(node)
            node = self.expand_afterstate(node)

        env = Game2048Env.restore(node.state, node.score)
        rollout_reward = self.rollout(env, self.rollout_depth) / self.v_norm
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = []
        for action in range(4):
            if total_visits == 0 or action not in root.children:
                distribution.append(0)
            else:
                distribution.append(root.children[action].visits / total_visits)
        best_action = np.argmax(distribution)
        return best_action, distribution
