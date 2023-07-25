import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from env import Maze
from utils import plot_policy, plot_values, test_agent

def value_iteration(policy_probs, state_values, theta=1e-6, gamma=0.99):
    delta = float("inf")
    while delta > theta:
        delta = 0
        for row in range(5):
            for col in range(5):
                old_value = state_values[(row, col)]
                action_probs = None
                max_qsa = float("-inf")
                for action in range(4):
                    next_state, reward, _, _ = env.simulate_step((row, col), action)
                    qsa = reward + gamma * state_values[next_state]

                    if qsa > max_qsa:
                      max_qsa = qsa
                      action_probs = np.zeros(4)
                      action_probs[action] = 1.

                state_values[(row, col)] = max_qsa
                policy_probs[(row, col)] = action_probs

                delta = max(delta, abs(max_qsa - old_value))


def policy(state):
    return policy_probs[state]

if __name__ == '__main__':
    env = Maze()
    policy_probs = np.full((5,5,4),0.25)
    state_values = np.zeros((5,5))
    value_iteration(policy_probs,state_values)
    
    frame = env.render('mode=rgb_array')
    plot_values(state_values,frame)
    plot_policy(policy_probs,frame)
    test_agent(env,policy)
    plt.show()