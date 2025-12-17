"""
Obstacle Generator for Frankie Obstacle Avoidance Simulation
Generates random cylindrical obstacles in the workspace.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np


class Obstacle:
    """Represents a cylindrical obstacle."""
    
    def __init__(self, position: Tuple[float, float], radius: float, height: float):
        self.position = np.array([position[0], position[1]], dtype=float)
        self.radius = float(radius)
        self.height = float(height)
    
    def contains_point(self, point: Tuple[float, float], safety_margin: float = 0.0) -> bool:
        """Check if a 2D point is inside the obstacle."""
        dist = np.hypot(point[0] - self.position[0], point[1] - self.position[1])
        return dist <= (self.radius + safety_margin)


def obstacles_to_world_grid(
    obstacles: List[Obstacle],
    world_bounds: Tuple[float, float, float, float],
    cell_size: float = 0.25,
    robot_safety_margin: float = 0.80,
) -> np.ndarray:
    """
    Convert obstacles to a world grid for the controller.
    Returns a numpy array where 1 = obstacle, 0 = free space.
    Grid is indexed as [row, col] where row increases downward (y decreases) 
    and col increases rightward (x increases).
    
    Args:
        obstacles: List of Obstacle objects
        world_bounds: Tuple of (min_x, max_x, min_y, max_y)
        cell_size: Size of each grid cell in meters
        robot_safety_margin: Additional margin around obstacles for robot safety
    
    Returns:
        numpy array where 1 = obstacle, 0 = free space
    """
    min_x, max_x, min_y, max_y = world_bounds
    
    width = max_x - min_x
    height = max_y - min_y
    
    cols = int(np.ceil(width / cell_size))
    rows = int(np.ceil(height / cell_size))
    
    # Initialize grid as free space (0)
    world_grid = np.zeros((rows, cols), dtype=np.int32)
    
    # Origin is at top-left (min_x, max_y)
    origin_x = min_x
    origin_y = max_y
    
    # Mark obstacles in grid
    for obs in obstacles:
        # Check all cells that might intersect with this obstacle
        obs_x, obs_y = obs.position[0], obs.position[1]
        effective_radius = obs.radius + robot_safety_margin
        
        # Calculate bounding box of obstacle in grid coordinates
        min_col = int(np.floor((obs_x - effective_radius - origin_x) / cell_size))
        max_col = int(np.ceil((obs_x + effective_radius - origin_x) / cell_size))
        min_row = int(np.floor((origin_y - (obs_y + effective_radius)) / cell_size))
        max_row = int(np.ceil((origin_y - (obs_y - effective_radius)) / cell_size))
        
        # Clamp to grid bounds
        min_col = max(0, min_col)
        max_col = min(cols - 1, max_col)
        min_row = max(0, min_row)
        max_row = min(rows - 1, max_row)
        
        # Mark cells that intersect with obstacle
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                # Convert grid cell center to world coordinates
                cell_center_x = origin_x + (c + 0.5) * cell_size
                cell_center_y = origin_y - (r + 0.5) * cell_size
                
                # Check if cell center is inside obstacle
                dist = np.hypot(cell_center_x - obs_x, cell_center_y - obs_y)
                if dist <= effective_radius:
                    world_grid[r, c] = 1
    
    return world_grid


def get_obstacles(
    world_bounds: Tuple[float, float, float, float],
    num_obstacles: int = 12,
    robot_base_pos: Tuple[float, float] = (0.0, 0.0),
    wall_position: Tuple[float, float] = (3.5, 0.0),
    min_distance_from_base: float = 1.5,
    min_distance_from_wall: float = 1.2,
    min_passage_width: float = 1.8,
    radius_range: Tuple[float, float] = (0.15, 0.25),
    height_range: Tuple[float, float] = (0.5, 1.1),
    rng: np.random.Generator = None,
) -> List[Obstacle]:
    """
    Generate random cylindrical obstacles in the workspace.
    
    Args:
        world_bounds: Tuple of (min_x, max_x, min_y, max_y) defining workspace bounds
        num_obstacles: Number of obstacles to generate
        robot_base_pos: Position of robot base (x, y) to avoid placing obstacles too close
        wall_position: Position of wall (x, y) to avoid placing obstacles too close
        min_distance_from_base: Minimum distance from robot base
        min_distance_from_wall: Minimum distance from wall
        min_passage_width: Minimum passage width between obstacles
        radius_range: Tuple of (min_radius, max_radius) for obstacle radius
        height_range: Tuple of (min_height, max_height) for obstacle height
        rng: Random number generator (if None, creates a new one)
    
    Returns:
        List of Obstacle objects
    """
    if rng is None:
        rng = np.random.default_rng()
    
    min_x, max_x, min_y, max_y = world_bounds
    obstacles = []
    
    for i in range(num_obstacles):
        max_attempts = 100
        for attempt in range(max_attempts):
            # Generate obstacle position based on region
            if i < 4:
                # First region: closer to start
                obs_x = float(rng.uniform(0.5, 2.0))
                obs_y = float(rng.uniform(-2.0, 2.0))
            elif i < 8:
                # Second region: middle area
                obs_x = float(rng.uniform(2.2, 3.0))
                obs_y = float(rng.uniform(-2.0, 2.0))
            else:
                # Third region: anywhere in workspace
                obs_x = float(rng.uniform(min_x + 0.8, max_x - 0.8))
                obs_y = float(rng.uniform(min_y + 0.8, max_y - 0.8))
            
            obs_radius = float(rng.uniform(radius_range[0], radius_range[1]))
            obs_height = float(rng.uniform(height_range[0], height_range[1]))
            
            # Check distances
            dist_from_robot = np.hypot(obs_x - robot_base_pos[0], obs_y - robot_base_pos[1])
            dist_from_wall = np.hypot(obs_x - wall_position[0], obs_y - wall_position[1])
            
            # Check collision with other obstacles
            too_close = False
            for other_obs in obstacles:
                dist = np.hypot(obs_x - other_obs.position[0], obs_y - other_obs.position[1])
                min_dist = obs_radius + other_obs.radius + min_passage_width
                if dist < min_dist:
                    too_close = True
                    break
            
            # Accept obstacle if it meets all constraints
            if (dist_from_robot >= min_distance_from_base and 
                dist_from_wall >= min_distance_from_wall and 
                not too_close):
                obstacle = Obstacle((obs_x, obs_y), obs_radius, obs_height)
                obstacles.append(obstacle)
                break
    
    return obstacles

