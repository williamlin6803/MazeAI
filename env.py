from typing import Tuple, Dict, Optional, Iterable

import numpy as np
import gym
from gym import spaces
from gym.error import DependencyNotInstalled
import pygame
from pygame import gfxdraw

"""
    Description:
        The environment consists of a grid of (size x size) positions. The agent
        starts at (row=0, col=0). The goal is always at (row=size-1, col=size-1).
        
    Actions:
        Num     Action
        0       Move up
        1       Move right
        2       Move down
        3       Move left
        
    Reward:
        Agent will receive reward of -1.0 until it reaches the goal.
        
    Episode termination:
        The episode terminates when the agent reaches the goal state.
"""
class Maze(gym.Env):
    """
        @ param exploring_starts: A boolean indicating whether the agent should restart at a random location.
        @ param shaped_rewards: A boolean indicating whether the environment should shape the rewards.
        @ param size: An integer representing the size of the maze. It will be of shape (size x size).
        @ effects: Initializes the maze environment 
        @ return: None
    """
    def __init__(self, exploring_starts: bool = False, shaped_rewards: bool = False, size: int = 5) -> None:
        super().__init__()
        self.exploring_starts = exploring_starts
        self.shaped_rewards = shaped_rewards
        self.state = (size - 1, size - 1)
        self.goal = (size - 1, size - 1)
        self.maze = self._create_maze(size=size)
        self.distances = self._compute_distances(self.goal, self.maze)
        self.action_space = spaces.Discrete(n=4)
        self.action_space.action_meanings = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: "LEFT"}
        self.observation_space = spaces.MultiDiscrete([size, size])

        self.screen = None
        self.agent_transform = None


    """
        @ param action: integer representing the action to take 
        @ effects: takes specified action in the environment
        @ return: a tuple containing the updated state, reward, completion status, and additional information
    """
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        reward = self.compute_reward(self.state, action)
        self.state = self._get_next_state(self.state, action)
        done = self.state == self.goal
        info = {}
        return self.state, reward, done, info
    
    
    """
        @ effects: resets environment
        @ return: a tuple containing the initial position of the agent
    """
    def reset(self) -> Tuple[int, int]:
        if self.exploring_starts:
            while self.state == self.goal:
                self.state = tuple(self.observation_space.sample())
        else:
            self.state = (0, 0)
        return self.state

    """
        @ para mode: string representing the mode to render the environment in
        @ effects: renders the environment in rgb arrays via numpy
        @ return: a numpy array or None
    """
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:

        screen_size = 600
        scale = screen_size / 5

        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((screen_size, screen_size))

        surf = pygame.Surface((screen_size, screen_size))
        surf.fill((22, 36, 71))


        for row in range(5):
            for col in range(5):

                state = (row, col)
                for next_state in [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]:
                    if next_state not in self.maze[state]:

                        # Add the geometry of the edges and walls (i.e. the boundaries between
                        # adjacent squares that are not connected).
                        row_diff, col_diff = np.subtract(next_state, state)
                        left = (col + (col_diff > 0)) * scale - 2 * (col_diff != 0)
                        right = ((col + 1) - (col_diff < 0)) * scale + 2 * (col_diff != 0)
                        top = (5 - (row + (row_diff > 0))) * scale - 2 * (row_diff != 0)
                        bottom = (5 - ((row + 1) - (row_diff < 0))) * scale + 2 * (row_diff != 0)

                        gfxdraw.filled_polygon(surf, [(left, bottom), (left, top), (right, top), (right, bottom)], (255, 255, 255))

        # Add the geometry of the goal square to the viewer.
        left, right, top, bottom = scale * 4 + 10, scale * 5 - 10, scale - 10, 10
        gfxdraw.filled_polygon(surf, [(left, bottom), (left, top), (right, top), (right, bottom)], (40, 199, 172))

        # Add the geometry of the agent to the viewer.
        agent_row = int(screen_size - scale * (self.state[0] + .5))
        agent_col = int(scale * (self.state[1] + .5))
        gfxdraw.filled_circle(surf, agent_col, agent_row, int(scale * .6 / 2), (228, 63, 90))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))

        return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))