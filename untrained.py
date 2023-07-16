import numpy as np
import matplotlib.pyplot as plt

from env import Maze
from utils import plot_policy, plot_values, test_agent

def policy(state):
    return policy_probs[state]

if __name__ == '__main__':
    env = Maze()
    frame = env.render('mode=rgb_array')
    plt.axis('off')
    plt.imshow(frame)
    policy_probs = np.full((5,5,4),0.25)
    test_agent(env,policy,episodes=1)
    plot_policy(policy_probs,frame)
    plt.show()