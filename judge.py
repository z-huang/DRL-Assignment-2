import numpy as np
from game2048 import Game2048Env
import student_agent

env = Game2048Env()
scores = []

for i in range(10):
    print(f'=== Running trial {i + 1} ===')
    state = env.reset()
    env.render()

    done = False
    while not done:
        action = student_agent.get_action(state, env.score)
        state, reward, done, _ = env.step(action)
        env.render()

    scores.append(env.score)
    print(f'Final score: {env.score}')
    input()

print(f'Average score: {np.mean(scores):.2f}')
