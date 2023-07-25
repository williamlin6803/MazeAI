import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np
from env import Maze
from collections import deque
from utils import test_dqagent

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.net(x)

def decay_epsilon(episode, start, end, decay):
    return end + (start - end) * np.exp(-1. * episode / decay)

def train_dqn(env):
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 50
    learning_rate = 0.001
    target_update = 10
    memory_size = 1000  
    batch_size = 16  
    episodes = 50 

    epsilon = epsilon_start
    input_dim = 2
    output_dim = env.action_space.n

    dqn = DQN(input_dim, output_dim)
    target_dqn = DQN(input_dim, output_dim)
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()

    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    replay_memory = deque(maxlen=memory_size)

    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            epsilon = decay_epsilon(episode, epsilon_start, epsilon_end, epsilon_decay)

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                action = torch.argmax(dqn(state_tensor)).item()

            next_state, reward, done, _ = env.step(action)
            replay_memory.append((state, action, reward, next_state, done))

            if len(replay_memory) > batch_size:
                minibatch = random.sample(replay_memory, batch_size)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(
                    lambda x: torch.tensor(x, dtype=torch.float32), zip(*minibatch))
                action_batch = action_batch.long()

                q_values = dqn(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)
                next_q_values = target_dqn(next_state_batch).max(1)[0]
                target = reward_batch + gamma * next_q_values * (~done_batch.to(torch.int)).to(torch.float)

                loss = criterion(q_values, target.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

        if episode % target_update == 0:
            target_dqn.load_state_dict(dqn.state_dict())

    return dqn

if __name__ == "__main__":
    env = Maze()
    frame = env.render('mode=rgb_array')
    plt.axis('off')
    plt.imshow(frame)
    trained_dqn = train_dqn(env)
    test_dqagent(env, trained_dqn)
    plt.show()