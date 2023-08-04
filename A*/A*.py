import sys
sys.path.append('../')  # Adds the parent directory to the system path
import heapq
import matplotlib.pyplot as plt
from env import Maze
import cv2

def manhattan_distance(start, goal):
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

def a_star_search(env): 
    start = env.reset()
    goal = env.goal
    frontier = [(manhattan_distance(start, goal), 0, start, [])]  # Priority, Cost, State, Path
    explored = set()
    
    while frontier:
        _, cost, current_state, path = heapq.heappop(frontier)
        
        if current_state == goal:
            return path
        
        explored.add(current_state)
        
        for action in range(env.action_space.n):
            next_state, _, _, _ = env.simulate_step(current_state, action)
            new_cost = cost + 1  # Cost for each step is 1
            heuristic = manhattan_distance(next_state, goal)
            
            if next_state not in explored:
                heapq.heappush(frontier, (new_cost + heuristic, new_cost, next_state, path + [action]))

    return []  # Return empty path if no solution found

def test_agent(env, delay):
    reached_end_state = False
    while not reached_end_state:
        state = env.reset()
        done = False
        path = a_star_search(env)
        for action in path:
            next_state, _, done, _ = env.step(action)
            frame = env.render(mode='rgb_array')
            cv2.imshow('Maze', frame)
            if cv2.waitKey(int(delay * 1000)) & 0xFF == ord('q'):
                break
            state = next_state

        if done:
            reached_end_state = True
        if cv2.waitKey(int(delay * 1000)) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    env = Maze()
    test_agent(env, delay=0.1)


