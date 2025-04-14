import pickle
import sys
import numpy as np
from game2048 import Game2048Env
from mcts2048 import TD_MCTS, TD_MCTS_Node
from ntuple import NTupleApproximator

patterns = [
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)],
]

approximator = NTupleApproximator(board_size=4, patterns=patterns)

print('Loading checkpoint...')
approximator.load_weights('td-8x6.pkl')
td_mcts = TD_MCTS(
    approximator,
    iterations=100,
    exploration_constant=1.41,
    rollout_depth=0,
    v_norm=25000
)


def get_action(state: np.ndarray, score):
    root = TD_MCTS_Node(state, 0)

    if score < 40000:
        td_mcts.iterations = 100
    elif score < 70000:
        td_mcts.iterations = 200
    elif score < 100000:
        td_mcts.iterations = 400
    elif score < 130000:
        td_mcts.iterations = 800

    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    action, visit_dist = td_mcts.best_action_distribution(root)
    # print(f'Score: {score}', "\r" , end=' ')

    return action


if __name__ == '__main__':
    env = Game2048Env()
    state = env.reset()
    env.render()

    done = False
    while not done:
        action = get_action(state, env.score)
        state, reward, done, _ = env.step(action)
        env.render()
