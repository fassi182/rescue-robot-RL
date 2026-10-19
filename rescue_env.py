import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RescueRobotEnv(gym.Env):
    """
    Custom Gymnasium Environment for Rescue Robot.
    Grid Dimensions: 5x5
    Actions: 0: Up, 1: Down, 2: Left, 3: Right, 4: Pickup, 5: Deliver[cite: 2]
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode="rgb_array"):
        super(RescueRobotEnv, self).__init__()
        
        self.grid_size = 5
        self.render_mode = render_mode
        
        # Action space: 4 directions + Pickup + Deliver[cite: 2]
        self.action_space = spaces.Discrete(6)
        
        # State space bounds: (robot_x, robot_y, carrying, victim1_rescued, victim2_rescued)[cite: 2]
        # carrying: 0 (No), 1 (Yes)
        # victim_status: 0 (Not Rescued), 1 (Rescued/Delivered)
        self.observation_space = spaces.MultiDiscrete([5, 5, 2, 2, 2])
        
        # Fixed locations
        self.base_hospital = (0, 0)
        self.obstacles = {(1, 1), (2, 3), (3, 1)}
        self.victim_locations = [(0, 4), (4, 4)]
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.robot_pos = [0, 0]
        self.carrying = 0
        self.victims_rescued = [0, 0]
        self.steps_taken = 0
        self.max_steps = 100
        
        state = self._get_obs()
        info = {}
        return state, info

    def _get_obs(self):
        return np.array([
            self.robot_pos[0],
            self.robot_pos[1],
            self.carrying,
            self.victims_rescued[0],
            self.victims_rescued[1]
        ], dtype=np.int32)

    def step(self, action):
        self.steps_taken += 1
        reward = -1  # Step penalty for shortest path search[cite: 2]
        terminated = False
        truncated = False

        r, c = self.robot_pos

        # Actions: 0-3 Movement[cite: 2]
        if action == 0 and r > 0: # Up
            next_pos = (r - 1, c)
        elif action == 1 and r < self.grid_size - 1: # Down
            next_pos = (r + 1, c)
        elif action == 2 and c > 0: # Left
            next_pos = (r, c - 1)
        elif action == 3 and c < self.grid_size - 1: # Right
            next_pos = (r, c + 1)
        else:
            next_pos = (r, c)

        # Movement Execution
        if action in [0, 1, 2, 3]:
            if next_pos in self.obstacles:
                reward = -5  # Penalty for running into obstacle[cite: 2]
            else:
                self.robot_pos = list(next_pos)

        # Action 4: Pickup Victim[cite: 2]
        elif action == 4:
            if self.carrying == 1:
                reward = -5  # Already carrying
            else:
                picked = False
                for idx, v_pos in enumerate(self.victim_locations):
                    if tuple(self.robot_pos) == v_pos and self.victims_rescued[idx] == 0:
                        self.carrying = 1
                        picked = True
                        reward = 10  # Reward for successfully picking up victim[cite: 2]
                        break
                if not picked:
                    reward = -5

        # Action 5: Deliver to Hospital[cite: 2]
        elif action == 5:
            if tuple(self.robot_pos) == self.base_hospital and self.carrying == 1:
                self.carrying = 0
                for idx in range(len(self.victims_rescued)):
                    if self.victims_rescued[idx] == 0:
                        self.victims_rescued[idx] = 1
                        break
                reward = 30  # High reward for hospital delivery[cite: 2]
            else:
                reward = -5

        # Check Termination
        if all(self.victims_rescued) and self.carrying == 0:
            reward += 50  # Mission completion bonus[cite: 2]
            terminated = True
        elif self.steps_taken >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        """RGB Array visualization frame generator for GIF rendering."""
        grid = np.ones((self.grid_size, self.grid_size, 3), dtype=np.uint8) * 255  # White background
        
        # Draw Obstacles (Black)
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = [0, 0, 0]
            
        # Draw Base/Hospital (Blue)
        grid[self.base_hospital[0], self.base_hospital[1]] = [0, 0, 255]
        
        # Draw Unrescued Victims (Red)
        for idx, v_pos in enumerate(self.victim_locations):
            if self.victims_rescued[idx] == 0:
                grid[v_pos[0], v_pos[1]] = [255, 0, 0]
                
        # Draw Robot (Green)
        grid[self.robot_pos[0], self.robot_pos[1]] = [0, 255, 0]
        
        # Upscale frame for clearer video/GIF
        return np.repeat(np.repeat(grid, 40, axis=0), 40, axis=1)