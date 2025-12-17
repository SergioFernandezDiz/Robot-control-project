"""
Maze environment builder for robot simulations.

This module provides classes and functions for building maze environments
in Swift 3D visualization, including walls, floors, and scene construction.
"""

import numpy as np
import random
import spatialgeometry as sg
import spatialmath as sm
import swift

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.maze_generator import get_maze


def to_world_pos(x, y, world_min_x=-10, world_min_y=-10):
    """
    Convert grid coordinates to world coordinates.
    
    Args:
        x: Grid x coordinate
        y: Grid y coordinate
        world_min_x: Minimum world x coordinate
        world_min_y: Minimum world y coordinate
    
    Returns:
        Tuple of (world_x, world_y) as integers
    """

    return int(x + world_min_x), int(y + world_min_y)


class Wall3D:
    """Represents a 3D wall with bounding box and visual object."""
    
    def __init__(self, min_xyz, max_xyz, visual_obj):
        """
        Initialize a 3D wall.
        
        Args:
            min_xyz: Minimum corner coordinates (x, y, z)
            max_xyz: Maximum corner coordinates (x, y, z)
            visual_obj: Swift visual object (Cuboid)
        """

        self.min = np.array(min_xyz, dtype=float)
        self.max = np.array(max_xyz, dtype=float)
        self.visual = visual_obj

    def bounds(self):
        """Get the bounding box of the wall."""
        return self.min, self.max


class Env:
    """
    Environment manager for Swift 3D visualization.
    
    Manages walls, floors, and other scene elements in a Swift environment.
    """
    
    def __init__(self,x_range=(-10, 10),y_range=(-10, 10),z=0.0,wall_height=1.5,swift_env=None):
        """
        Initialize the environment.
        
        Args:
            x_range: Tuple of (min_x, max_x) for world bounds
            y_range: Tuple of (min_y, max_y) for world bounds
            z: Base z coordinate
            wall_height: Default wall height
            swift_env: Swift environment 
        """

        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z = z
        self.wall_height = wall_height
        self.swift = swift_env if swift_env is not None else swift.Swift()
        self.swift.launch(reload=True)
        self.walls = []

    def add_wall(self, center, size, color="gray", texture=None):
        """
        Add a wall to the environment.
        
        Args:
            center: Center position (x, y, z)
            size: Size (width, depth, height)
            color: Wall color 
            texture: Optional texture 
        
        Returns:
            The created Wall3D object
        """ 

        cx, cy, cz = center
        sx, sy, sz = size
        if texture is None:
            wall = sg.Cuboid(scale=size, color=color)
        else:
            wall = sg.Cuboid(scale=size, texture=texture)
        wall.T = sm.SE3(cx, cy, cz)
        self.swift.add(wall)
        min_xyz = (cx - sx/2, cy - sy/2, cz - sz/2)
        max_xyz = (cx + sx/2, cy + sy/2, cz + sz/2)
        wall_3d = Wall3D(min_xyz, max_xyz, wall)
        self.walls.append(wall_3d)
        return wall_3d


def coords_to_map(x, y, min_x=-10, min_y=-10):
    """
    Convert world coordinates to map/grid coordinates.
    
    Args:
        x: World x coordinate
        y: World y coordinate
        min_x: Minimum world x coordinate
        min_y: Minimum world y coordinate
    
    Returns:
        Tuple of (map_x, map_y)
    """

    return x - min_x, y - min_y


def find_bulb_valid_pos(maze, near=(0, 0), box_size=4):
    """
    Find a valid position for a bulb near a given position.
    
    Args:
        maze: 2D numpy array representing the maze (0=free, 1=wall)
        near: Center position to search around (row, col)
        box_size: Size of search box around center
    
    Returns:
        Tuple of (row, col) 
    """

    for x in range(near[0] - box_size, near[0] + box_size + 1):
        if x < 0 or x >= len(maze):
            continue
        for y in range(near[1] - box_size, near[1] + box_size + 1):
            if y < 0 or y >= len(maze[0]):
                continue
            if maze[x][y] == 0:
                return x, y
    return None


def add_floor(env, grid_shape, thickness=0.1, color="lightgray"):
    """
    Add a floor to the environment.
    
    Args:
        env: Env instance
        grid_shape: Shape of the grid (rows, cols)
        thickness: Floor thickness
        color: Floor color
    """

    h, w = grid_shape
    sx = h
    sy = w
    sz = thickness
    cx = h / 2.0
    cy = w / 2.0
    cz = -sz
    env.add_wall((cx-2, cy-2, cz+.08), (sx+4, sy+4, sz), color=color)


def place_walls_from_grid(env, grid, wall_height=2.0):
    """
    Place walls in the environment based on a grid.
    
    Args:
        env: Env instance
        grid: 2D numpy array (1=wall, 0=free)
        wall_height: Height of walls
    """

    h, w = grid.shape
    for x in range(h):
        for y in range(w):
            if grid[x, y] == 1:
                cx = x
                cy = y
                cz = wall_height / 2.0
                env.add_wall((cx, cy, cz), (1.0, 1.0, wall_height))


def find_bulb_pos(grid, start_pos, min_distance=2.0, max_attempts=1000):
    """
    Find a valid bulb position with minimum distance from start.
    
    Args:
        grid: 2D numpy array (0=free, 1=obstacle)
        start_pos: Start position (row, col)
        min_distance: Minimum distance from start
        max_attempts: Maximum attempts (not used, kept for compatibility)
    
    Returns:
        Tuple of (row, col) if found, None otherwise
    """

    h, w = grid.shape
    free_cells = [(x, y) for x in range(h) for y in range(w) if grid[x, y] == 0]
    random.shuffle(free_cells)
    sx, sy = start_pos
    min_dist_sq = min_distance * min_distance
    for x, y in free_cells:
        dx = x - sx
        dy = y - sy
        if dx*dx + dy*dy >= min_dist_sq:
            return (x, y)
    return None


def build_maze_scene(env, max_tries=3, gap=8, show_path=False, wall_height=2.0):
    """
    Build a complete maze scene in the environment.
    
    This function generates a maze, places walls and floors, and returns
    the grid and key positions.
    
    Args:
        env: Env instance
        max_tries: Maximum attempts to generate valid maze
        gap: Gap size between maze and end area
        show_path: If True, visualize start (blue) and end (red) positions
        wall_height: Height of walls
    
    Returns:
        Tuple of (grid, end_pos, start_pos)
        - grid: 2D numpy array representing the world
        - end_pos: End position (x, y)
        - start_pos: Start position (x, y)
    
    Raises:
        RuntimeError: If unable to build valid maze after max_tries attempts
    """

    last_error = None

    if hasattr(env, "walls"):
        for w in env.walls:
            try:
                env.swift.remove(w.visual)
            except Exception:
                pass
        env.walls = []

    for _ in range(max_tries):
        maze_full = np.array(get_maze(10, 10))
        if maze_full.shape[0] <= 2 or maze_full.shape[1] <= 2:
            raise ValueError("Maze too small!")

        maze_core = maze_full[1:-1, 1:-1]

        h_core, w_core = maze_core.shape

        grid_h = 4 + h_core
        grid_w = w_core + gap + 1
        grid = np.zeros((grid_h, grid_w), dtype=int)

        start_core = find_bulb_valid_pos(maze_core, near=(grid_h//2, 0), box_size=4)
        grid[:, grid_w-1] = 1

        maze_start_col = 1 + gap
        for x in range(h_core):
            for y in range(w_core):
                if maze_core[x, y] == 1:
                    grid[x + 2, y] = 1
                elif maze_core[x, y] == 2:
                    grid[x + 2, y] = 2

        end_core = (grid_w//2, len(maze_core)+2)

        sx_core, sy_core = start_core
        ex_core, ey_core = end_core

        start_x = sx_core
        start_y = sy_core
        end_x = ex_core
        end_y = ey_core

        add_floor(env, grid.shape, thickness=0.1)
        place_walls_from_grid(env, grid, wall_height=wall_height)

        if show_path:
            env.add_wall((start_x, start_y, 0.5), (0.5, 0.5, 1.0), color="blue")
            env.add_wall((end_x, end_y, 0.5), (0.5, 0.5, 1.0), color="red")

        end_pos = (end_x, end_y)
        start_pos = (start_x, start_y)
        return grid, end_pos, start_pos


