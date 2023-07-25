import sys
sys.path.append('../')  # Adds the parent directory to the system path

import numpy as np
import matplotlib.pyplot as plt
from env import Maze
from utils import plot_policy, plot_values, test_agent

def policy_evaluation(policy_probs,state_values,theta=1e-6,gamma=0.99):
    delta = float('inf')
    while delta > theta:
        delta = 0

        for row in range(5):
            for col in range(5):
                old_value = state_values[(row,col)]
                new_value = 0.
                action_probabilities = policy_probs[(row,col)]

                for action,prob in enumerate(action_probabilities):
                    next_state,reward,_,_ = env.simulate_step((row,col),action)
                    new_value += prob * (reward + gamma * state_values[next_state])

                state_values[(row,col)] = new_value

                delta = max(delta,abs(old_value-new_value))

def policy_improvement(policy_probs,state_values,gamma=0.99):

    policy_stable = True

    for row in range(5):
        for col in range(5):
            old_action = policy_probs[(row,col)].argmax()

            new_action = None
            max_qsa = float('-inf')

            for action in range(4):
                next_state,reward,_,_ = env.simulate_step((row,col),action)
                qsa = reward + gamma * state_values[next_state]

                if qsa > max_qsa:
                    new_action = action
                    max_qsa = qsa

            action_probs = np.zeros(4)
            action_probs[new_action] = 1.
            policy_probs[(row,col)] = action_probs

            if new_action != old_action:
                policy_stable = False

    return policy_stable

def policy_iteration(policy_probs,state_values,theta=1e-6,gamma=0.99):
    policy_stable = False

    while not policy_stable:
        policy_evaluation(policy_probs,state_values,theta,gamma)
        policy_stable = policy_improvement(policy_probs,state_values,gamma)
        
    plot_values(state_values,frame)
    plot_policy(policy_probs,frame)

def policy(state):
    return policy_probs[state]

if __name__ == '__main__':
    env = Maze()
    policy_probs = np.full((5,5,4),0.25)
    state_values = np.zeros((5,5))
    frame = env.render('mode=rgb_array')
    policy_iteration(policy_probs,state_values)
    test_agent(env,policy)
    plt.show()