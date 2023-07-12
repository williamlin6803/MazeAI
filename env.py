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
        @ param: None
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
        @ param mode: string representing the mode to render the environment in
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
    
    """
        @ param: None
        @ effects: closes the environment
        @ return: None
    """
    def close(self) -> None:
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            
    """
        @ param state: the state of the agent prior to taking the action
        @ param action: the action taken by the agent
        @ effects: computes the reward attained by taking action 'a' at state 's'
        @ return: a float representing the reward signal received by the agent
    """
    def compute_reward(self, state: Tuple[int, int], action: int) -> float:
        next_state = self._get_next_state(state, action)
        if self.shaped_rewards:
            return - (self.distances[next_state] / self.distances.max())
        return - float(state != self.goal)
    
    """
        @ param state: state of the agent prior to taking the action
        @ param action: the action to simulate the step with
        @ effects: simulates taking an action in the environment
        @ return: a tuple containing the next state, reward, completion status, and additional information
    """
    def simulate_step(self, state: Tuple[int, int], action: int):
        reward = self.compute_reward(state, action)
        next_state = self._get_next_state(state, action)
        done = next_state == self.goal
        info = {}
        return next_state, reward, done, info 

    """
        @ param state: state of the agent prior to taking the action
        @ param action: move performed by the agent
        @ effects: gets the next state after the agent performs action 'a' in state 's'
        @ return: state instance representing the new state
    """
    def _get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        if action == 0:
            next_state = (state[0] - 1, state[1])
        elif action == 1:
            next_state = (state[0], state[1] + 1)
        elif action == 2:
            next_state = (state[0] + 1, state[1])
        elif action == 3:
            next_state = (state[0], state[1] - 1)
        else:
            raise ValueError("Action value not supported:", action)
        if next_state in self.maze[state]:
            return next_state
        return state
    
    """
        @ param size: number of elements of each side in the square grid
        @ effects: creates representation of the maze as a dictionary; keys: states available to agent & values: lists of adjacent states
        @ return: adjacency list dictionary.
    """
    @staticmethod
    def _create_maze(size: int) -> Dict[Tuple[int, int], Iterable[Tuple[int, int]]]:
        maze = {(row, col): [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
                for row in range(size) for col in range(size)}

        left_edges = [[(row, 0), (row, -1)] for row in range(size)]
        right_edges = [[(row, size - 1), (row, size)] for row in range(size)]
        upper_edges = [[(0, col), (-1, col)] for col in range(size)]
        lower_edges = [[(size - 1, col), (size, col)] for col in range(size)]
        walls = [
            [(1, 0), (1, 1)], [(2, 0), (2, 1)], [(3, 0), (3, 1)],
            [(1, 1), (1, 2)], [(2, 1), (2, 2)], [(3, 1), (3, 2)],
            [(3, 1), (4, 1)], [(0, 2), (1, 2)], [(1, 2), (1, 3)],
            [(2, 2), (3, 2)], [(2, 3), (3, 3)], [(2, 4), (3, 4)],
            [(4, 2), (4, 3)], [(1, 3), (1, 4)], [(2, 3), (2, 4)],
        ]

        obstacles = upper_edges + lower_edges + left_edges + right_edges + walls

        for src, dst in obstacles:
            maze[src].remove(dst)

            if dst in maze:
                maze[dst].remove(src)

        return maze
    
    """
        @ param goal: tuple representing the location of the goal in a two-dimensional grid
        @ param maze: dictionary holding the adjacency lists of all locations in the two-dimensional grid.
        @ effects: computes the distance to the goal from all other positions in the maze using Dijkstra's algorithm.
        @ return: A (H x W) numpy array holding the minimum number of moves for each position
        to reach the goal.
    """
    @staticmethod
    def _compute_distances(goal: Tuple[int, int], maze: Dict[Tuple[int, int], Iterable[Tuple[int, int]]]) -> np.ndarray:
        distances = np.full((5, 5), np.inf)
        visited = set()
        distances[goal] = 0.

        while visited != set(maze):
            sorted_dst = [(v // 5, v % 5) for v in distances.argsort(axis=None)]
            closest = next(x for x in sorted_dst if x not in visited)
            visited.add(closest)

            for neighbour in maze[closest]:
                distances[neighbour] = min(distances[neighbour], distances[closest] + 1)
        return distances

env = Maze()
env.reset()
env.render(mode='rgb_array')