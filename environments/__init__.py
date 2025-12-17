"""
Environments package for building and managing simulation environments.

This package provides utilities for creating and managing 3D environments
for robot simulations, including maze environments and scene builders.
"""

from .maze import (
    Wall3D,
    Env,
    to_world_pos,
    coords_to_map,
    find_bulb_valid_pos,
    add_floor,
    place_walls_from_grid,
    find_bulb_pos,
    build_maze_scene,
)

__all__ = [
    "Wall3D",
    "Env",
    "to_world_pos",
    "coords_to_map",
    "find_bulb_valid_pos",
    "add_floor",
    "place_walls_from_grid",
    "find_bulb_pos",
    "build_maze_scene",
]

