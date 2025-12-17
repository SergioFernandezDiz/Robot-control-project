"""
Frankie Pick-and-Place with Obstacle Avoidance
Uses the integrated controller navigation system (same as environment.py).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, List
import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import swift
import matplotlib as mtb
import matplotlib.pyplot as plt
import msvcrt
import os

_HAS_MSVCRT = True

mtb.rcParams.update(mtb.rcParamsDefault)

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import controller and utilities
from frankie import FrankieController, FrankieControllerParams, TaskState, FrankieAgent
from core.algebra import Vector3
from plotting.basic import plot_episode_metrics
from generators.obstacle_generator import Obstacle, get_obstacles, obstacles_to_world_grid

# Visual parameters
COLOR_RED = (0.8, 0.2, 0.3, 1.0)
COLOR_BLUE = (0.2, 0.5, 0.8, 1.0)
COLOR_GREEN = (0.2, 0.8, 0.3, 1.0)
COLOR_YELLOW = (0.9, 0.8, 0.2, 1.0)
COLOR_WHITE = (1.0, 1.0, 1.0, 0.6)
COLOR_GRAY = (0.5, 0.5, 0.5, 1.0)
COLOR_BRICK = (0.7, 0.3, 0.2, 1.0)
COLOR_OBSTACLE = (0.6, 0.3, 0.5, 0.8)  # Purple obstacles

BULB_RADIUS = 0.04
FIXTURE_RADIUS = 0.02
FLOOR_HEIGHT = 0.02
WORLD_CELL_SIZE = 0.25  # Grid resolution for obstacle avoidance
WAYPOINT_RADIUS = 0.03  # Small waypoint markers

# FrankieAgent is now imported from frankie package (line 30)


def add_world_bounds(env: swift.Swift, bounds: Tuple[float, float, float, float]) -> sg.Cuboid:
    min_x, max_x, min_y, max_y = bounds
    width, depth = max_x - min_x, max_y - min_y
    floor = sg.Cuboid(scale=[width, depth, FLOOR_HEIGHT], color=COLOR_WHITE)
    floor.T = sm.SE3((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, 0.01)
    env.add(floor)
    return floor


def add_wall(env: swift.Swift, position: Tuple[float, float, float], size: Tuple[float, float, float]) -> sg.Cuboid:
    wall = sg.Cuboid(scale=size, color=COLOR_BRICK)
    wall.T = sm.SE3(*position)
    env.add(wall)
    return wall


def run_episode(
    env: swift.Swift,
    frankie_agent: FrankieAgent,
    initial_joint_config: np.ndarray,
    initial_base_pose: Tuple[float, float, float],
    episode_idx: int,
    first_run: bool = False
):
    """Run one pick-and-place episode with obstacles using integrated controller navigation."""

    dt = 0.05
    simulation_time = 60.0  # seconds per episode

    world_bounds = (-4.0, 4.0, -4.0, 4.0)
    min_x, max_x, min_y, max_y = world_bounds

    # Scene setup
    add_world_bounds(env, world_bounds)
    wall_position = (3.5, 0.0, 0.5)
    wall_size = (0.1, 2.0, 1.2)
    wall = add_wall(env, wall_position, wall_size)
    
    screw_x = wall_position[0] - wall_size[0] / 2.0 - 0.02
    fixture_position = (screw_x, 0.0, 0.8)
    fixture_visual = sg.Cylinder(
        radius=FIXTURE_RADIUS * 1.2,
        length=0.04,
        pose=sm.SE3(*fixture_position) * sm.SE3.Ry(np.pi/2),
        color=COLOR_GRAY,
    )
    env.add(fixture_visual)

    # Generate obstacles using obstacle_generator
    robot_base_pos = (0.0, 0.0)
    rng = np.random.default_rng()
    obstacles = get_obstacles(
        world_bounds=world_bounds,
        num_obstacles=12,
        robot_base_pos=robot_base_pos,
        wall_position=(wall_position[0], wall_position[1]),
        rng=rng,
    )
    
    # Create visual representations of obstacles
    obstacle_visuals = []
    for obs in obstacles:
        obs_visual = sg.Cylinder(
            radius=obs.radius,
            length=obs.height,
            pose=sm.SE3(obs.position[0], obs.position[1], obs.height / 2.0),
            color=COLOR_OBSTACLE
        )
        env.add(obs_visual)
        obstacle_visuals.append(obs_visual)
    
    print(f"Spawned {len(obstacles)} obstacles.")

    # Convert obstacles to world grid for controller
    world_grid = obstacles_to_world_grid(obstacles, world_bounds, WORLD_CELL_SIZE)
    print(f"Created world grid: {world_grid.shape[0]} rows x {world_grid.shape[1]} cols")

    # Random bulb spawn (check obstacle collision)
    min_distance_from_base = 1.8
    max_attempts = 100
    tower_height = 0.40
    bulb_z = tower_height + 0.05

    def spawn_random_bulb():
        bulb_position_local = None
        for attempt in range(max_attempts):
            bulb_x = float(rng.uniform(min_x + 0.5, max_x - 1.0))
            bulb_y = float(rng.uniform(min_y + 0.5, max_y - 0.5))

            dist_from_base = np.hypot(bulb_x - robot_base_pos[0], bulb_y - robot_base_pos[1])
            dist_from_wall = abs(bulb_x - wall_position[0])
            lateral_offset = abs(bulb_y)
            
            collision = False
            for obs in obstacles:
                if obs.contains_point((bulb_x, bulb_y), safety_margin=0.4):
                    collision = True
                    break
            
            if dist_from_base >= min_distance_from_base and dist_from_wall >= 0.5 and lateral_offset >= 0.6 and not collision:
                bulb_position_local = (bulb_x, bulb_y, bulb_z)
                break

        tower_scale = [0.15, 0.15, tower_height]
        tower_obj = sg.Cuboid(
            scale=tower_scale,
            pose=sm.SE3(bulb_position_local[0], bulb_position_local[1], tower_height / 2.0),
            color=COLOR_GRAY,
        )
        env.add(tower_obj)

        bulb_obj = sg.Sphere(radius=BULB_RADIUS, pose=sm.SE3(*bulb_position_local), color=COLOR_YELLOW)
        env.add(bulb_obj)

        return bulb_position_local, tower_obj, bulb_obj

    bulb_position, tower, bulb_visual = spawn_random_bulb()
    manual_reset = False
    print(f"Initial bulb spawned at: {bulb_position}")

    # Controller setup (using integrated navigation system)
    controller_params = FrankieControllerParams(
        linear_velocity_gain=0.7,
        angular_velocity_gain=2.5,
        max_linear_velocity=0.6,
        max_angular_velocity=2.0,
        base_nav_tolerance=0.05,
        arm_gain=12.0,
        pinv_damping=1.0,
        q_dot_limit=1.5,
        pos_tolerance=2.5e-2,
        screw_rotations=2.0,
        grasp_height_offset=0.06,
        bulb_approach_distance=0.8,
        wall_approach_distance=1.0,
        wall_stop_distance=1.0,
        start_position=(0.0, 0.0),
        wall_thickness=wall_size[0],
        arm_safety_margin=0.02,
        maze_to_bulb_distance_threshold=1.0,
    )
    controller = FrankieController(controller_params)
    
    # Set world grid for obstacle avoidance
    controller.set_world(world_grid, WORLD_CELL_SIZE)
    controller.set_bulb_position(bulb_position)
    controller.set_wall_fixture(
        fixture_position,
        orientation=0.0,
        wall_x=wall_position[0],
        wall_thickness=wall_size[0],
    )
    controller.params.start_position = (0.0, 0.0)
    controller.start_task()
    
    # Debug: Check initial state
    print(f"Controller initialized:")
    print(f"  State: {controller.state.value}")
    print(f"  World grid set: {controller.world is not None}")
    print(f"  Bulb position: {controller.bulb_position}")
    print(f"  Wall fixture: {controller.wall_fixture_position}")
    print(f"  Maze target: {controller._maze_target}")

    # Save original home position (never changes, used to check if robot is at home)
    original_home_position = (0.0, 0.0)

    # Path visualization
    path_markers = []
    prev_maze_waypoints_len = 0
    prev_state = TaskState.IDLE
    visualized_paths = {
        "bulb": False,
        "wall": False,
        "home": False,
    }
    path_color_map = {
        "bulb": COLOR_GREEN,
        "wall": COLOR_BLUE,
        "home": COLOR_RED,
    }
    
    # Flag to prevent infinite loop when already at home
    already_at_home = False

    def visualize_waypoints(waypoints: List[Vector3], color: Tuple[float, float, float, float], path_name: str, remove_last_n: int = 0):
        """Visualize waypoints as markers and connecting lines.
        
        Args:
            waypoints: List of waypoints to visualize
            color: Color for markers and lines
            path_name: Name for logging
            remove_last_n: Number of waypoints to remove from the end (for bulb path, remove last 2)
        """
        if not waypoints:
            return
        
        # Remove last N waypoints if requested (for bulb path)
        waypoints_to_show = waypoints[:-remove_last_n] if remove_last_n > 0 and len(waypoints) > remove_last_n else waypoints
        
        if not waypoints_to_show:
            return
        
        for i, wp in enumerate(waypoints_to_show):
            # Add small sphere marker
            marker = sg.Sphere(radius=WAYPOINT_RADIUS, pose=sm.SE3(wp.x, wp.y, 0.05), color=color)
            env.add(marker)
            path_markers.append(marker)
            
            # Connect waypoints with lines
            if i > 0:
                prev_wp = waypoints_to_show[i-1]
                line_length = np.hypot(wp.x - prev_wp.x, wp.y - prev_wp.y)
                if line_length > 1e-6:  # Only draw if waypoints are different
                    line_midpoint = ((wp.x + prev_wp.x)/2, (wp.y + prev_wp.y)/2, 0.05)
                    line_angle = np.arctan2(wp.y - prev_wp.y, wp.x - prev_wp.x)
                    
                    line = sg.Cylinder(radius=0.01, length=line_length, 
                                     pose=sm.SE3(line_midpoint) * sm.SE3.Rz(line_angle) * sm.SE3.Ry(np.pi/2),
                                     color=color)
                    env.add(line)
                    path_markers.append(line)
        
        removed_str = f" (removed {remove_last_n} from end)" if remove_last_n > 0 else ""
        print(f"Visualized {len(waypoints_to_show)} waypoints for {path_name} path{removed_str}")

    # Logging
    time_log, state_log, base_pos_log, ee_pos_log = [], [], [], []
    dist_to_bulb_log, dist_to_fixture_log, base_vel_log, arm_vel_norm_log = [], [], [], []

    steps = int(simulation_time / dt)
    task_completed_once = False
    t_episode = 0.0

    print("Starting simulation with integrated obstacle avoidance...")
    print(f"Initial robot base state: {frankie_agent.base_state()}")

    while t_episode < simulation_time:
        base_state = frankie_agent.base_state()
        robot = frankie_agent._robot

        # Compute control - controller handles navigation automatically
        qdot, base_cmd, current_state = controller.compute_control(robot, base_state)
        
        # Debug: Print state on first iteration
        if t_episode < dt * 2:
            print(f"[t={t_episode:.3f}s] State: {current_state.value}, qdot norm: {np.linalg.norm(qdot):.4f}, base_cmd: {base_cmd}")
            print(f"  Maze waypoints: {len(controller._maze_waypoints)}, maze_index: {controller._maze_index}")
            print(f"  Maze target: {controller._maze_target}")

        # Visualize waypoints when they are planned
        current_maze_waypoints_len = len(controller._maze_waypoints)
        
        # Detect when waypoints are first planned or when state changes to NAVIGATE_IN_MAZE
        if (current_state == TaskState.NAVIGATE_IN_MAZE and 
            current_maze_waypoints_len > 0 and 
            (current_maze_waypoints_len != prev_maze_waypoints_len or prev_state != TaskState.NAVIGATE_IN_MAZE)):
            # Determine path color based on target
            if controller._maze_target == "bulb":
                # Path to bulb (green) - remove last 2 waypoints (robot stops before reaching bulb)
                if not visualized_paths["bulb"]:
                    visualize_waypoints(controller._maze_waypoints, path_color_map["bulb"], "bulb", remove_last_n=2)
                    visualized_paths["bulb"] = True
            elif controller._maze_target == "end":
                # Check if we're returning home (after screwing) or going to wall (after picking)
                if prev_state in [TaskState.SCREW_BULB, TaskState.RETURN_TO_START]:
                    # Home path (red)
                    if not visualized_paths["home"]:
                        visualize_waypoints(controller._maze_waypoints, path_color_map["home"], "home")
                        visualized_paths["home"] = True
                else:
                    # Going to wall after picking bulb (blue)
                    if not visualized_paths["wall"]:
                        visualize_waypoints(controller._maze_waypoints, path_color_map["wall"], "wall")
                        visualized_paths["wall"] = True
        
        # Check if we're at home position (use original home position, not the updated one)
        home_x, home_y = original_home_position
        home_vec = Vector3.from_xy(home_x, home_y, 0.0)
        base_vec = Vector3.from_xy(base_state[0], base_state[1], 0.0)
        dist_to_start = Vector3.distance(base_vec, home_vec)
        at_home = dist_to_start <= controller.params.base_nav_tolerance * 1.5
        
        # Handle transition from SCREW_BULB to RETURN_TO_START - compute and visualize home path
        if current_state == TaskState.RETURN_TO_START and prev_state == TaskState.SCREW_BULB:
            # Reset the flag when starting to return home
            already_at_home = False
            # Just finished screwing, compute home path immediately and set it in controller
            if controller.world is not None and not visualized_paths["home"]:
                # Use controller's method to plan path (this will set _maze_waypoints)
                current_vec = Vector3.from_xy(base_state[0], base_state[1], 0.0)
                home_vec = Vector3.from_xy(home_x, home_y, 0.0)
                
                # Set both end position and start position to home (for RETURN_TO_START to work correctly)
                controller.set_end_position((home_x, home_y, 0.0))
                controller.params.start_position = (home_x, home_y)  # Ensure start_position is set to original home
                controller._maze_target = "end"
                controller._plan_maze_path(current_vec, home_vec)
                
                # Visualize the path that controller will use
                if controller._maze_waypoints:
                    visualize_waypoints(controller._maze_waypoints, path_color_map["home"], "home")
                    visualized_paths["home"] = True
                    controller._maze_index = 0  # Reset index to start following waypoints
                    print(f"[t={t_episode:.2f}s] Computed and visualized home path with {len(controller._maze_waypoints)} waypoints - RETURN_TO_START will follow this path")
                else:
                    print(f"[t={t_episode:.2f}s] Warning: Could not compute home path")
        
        # Handle RETURN_TO_START - plan path if not already planned
        if current_state == TaskState.RETURN_TO_START and controller.world is not None:
            # Check if actually at home first
            if at_home:
                # Actually at home, go to RESET
                already_at_home = True
                controller.state = TaskState.RESET
                print(f"[t={t_episode:.2f}s] At home position (dist={dist_to_start:.3f}m), transitioning to RESET")
            elif not controller._maze_waypoints:
                # Path not yet planned, plan it now
                current_vec = Vector3.from_xy(base_state[0], base_state[1], 0.0)
                home_vec = Vector3.from_xy(home_x, home_y, 0.0)
                controller.set_end_position((home_x, home_y, 0.0))
                controller.params.start_position = (home_x, home_y)  # Ensure start_position is set to original home
                controller._maze_target = "end"
                controller._plan_maze_path(current_vec, home_vec)
                if controller._maze_waypoints:
                    visualize_waypoints(controller._maze_waypoints, path_color_map["home"], "home")
                    visualized_paths["home"] = True
                    controller._maze_index = 0
                    print(f"[t={t_episode:.2f}s] Planned home path with {len(controller._maze_waypoints)} waypoints - RETURN_TO_START will follow this path")
        
        # Handle STOP_IN_FRONT_OF_WALL when target is home - should go to RETURN_TO_START or RESET
        # Only if NOT grasping (returning home after screwing)
        if current_state == TaskState.STOP_IN_FRONT_OF_WALL and controller._maze_target == "end" and not controller._grasped:
            # If we're at home position, go directly to RESET (arm should already be in rest position)
            if at_home or already_at_home:
                controller.state = TaskState.RESET
                already_at_home = True  # Set flag to prevent loop
                print(f"[t={t_episode:.2f}s] At home position (dist={dist_to_start:.3f}m), transitioning to RESET")
            else:
                # Not at home yet, but reached last waypoint - continue to home position
                # Check if we're close enough to home
                if dist_to_start <= controller.params.base_nav_tolerance * 2.0:
                    # Close enough, go to RESET
                    controller.state = TaskState.RESET
                    already_at_home = True
                    print(f"[t={t_episode:.2f}s] Close to home (dist={dist_to_start:.3f}m), transitioning to RESET")
                else:
                    # Still far from home, go to RETURN_TO_START to continue navigation
                    # But only if we haven't been here before (avoid loop)
                    if prev_state != TaskState.RETURN_TO_START:
                        controller.state = TaskState.RETURN_TO_START
                        print(f"[t={t_episode:.2f}s] Finished maze waypoints but not at home yet (dist={dist_to_start:.3f}m), transitioning to RETURN_TO_START")
                    else:
                        # Already tried RETURN_TO_START, go directly to RESET even if not exactly at home
                        controller.state = TaskState.RESET
                        already_at_home = True
                        print(f"[t={t_episode:.2f}s] Already tried RETURN_TO_START, going directly to RESET (dist={dist_to_start:.3f}m)")
        
        # Handle APPROACH_WALL when target is home and not grasping - should go to RESET
        if current_state == TaskState.APPROACH_WALL and controller._maze_target == "end" and not controller._grasped:
            # We're at home, no bulb to place, just reset the arm
            controller.state = TaskState.RESET
            already_at_home = True
            print(f"[t={t_episode:.2f}s] At home with no bulb, transitioning to RESET")
        
        prev_maze_waypoints_len = current_maze_waypoints_len
        prev_state = current_state

        # Apply control
        frankie_agent.apply_velocity_cmd(qdot, base_cmd)

        # EE pose
        T_ee = robot.fkine(robot.q)
        ee_pos = np.asarray(T_ee.t, dtype=float).reshape(3)

        # Metrics
        dist_to_bulb = float(np.linalg.norm(ee_pos - np.array(bulb_position)))
        dist_to_fixture = float(np.linalg.norm(ee_pos - np.array(fixture_position)))
        base_vel = float(np.linalg.norm([base_cmd[0] if base_cmd else 0.0, base_cmd[1] if base_cmd else 0.0]))
        arm_vel_norm = float(np.linalg.norm(qdot[2:]))

        # Log
        time_log.append(t_episode)
        state_log.append(current_state.value)
        base_pos_log.append(base_state.copy())
        ee_pos_log.append(ee_pos.copy())
        dist_to_bulb_log.append(dist_to_bulb)
        dist_to_fixture_log.append(dist_to_fixture)
        base_vel_log.append(base_vel)
        arm_vel_norm_log.append(arm_vel_norm)

        # Move bulb visual when grasped - but NOT during return home or reset
        # The bulb should only follow the end-effector when actually being transported
        if controller._grasped and current_state not in [
            TaskState.RETURN_TO_START,
            TaskState.RESET,
            TaskState.IDLE,
        ]:
            bulb_visual.T = sm.SE3(*ee_pos)
        elif current_state in [
            TaskState.TRANSPORT_TO_WALL,
            TaskState.STOP_IN_FRONT_OF_WALL,
            TaskState.APPROACH_WALL,
            TaskState.SCREW_BULB,
        ]:
            # Also follow during these states if grasped (redundant check but safe)
            if controller._grasped:
                bulb_visual.T = sm.SE3(*ee_pos)

        # Log task completion (but don't spawn new bulb - new episode starts when simulation_time is over)
        if controller.is_task_complete() and at_home and not task_completed_once:
            task_completed_once = True
            print(f"[t={t_episode:.2f}s] Task completed and robot at home (dist={dist_to_start:.3f}m)! Waiting for simulation time to end...")
        elif controller.is_task_complete() and not at_home and not task_completed_once:
            # Debug: task completed but not at home yet - wait for robot to return home
            if t_episode % 1.0 < dt:  # Print once per second
                print(f"[t={t_episode:.2f}s] Task completed but not at home yet (dist={dist_to_start:.3f}m, state={current_state.value}) - waiting to return home...")

        # State transition logging
        if len(state_log) > 1 and state_log[-1] != state_log[-2]:
            print(f"[t={t_episode:.2f}s] State: {current_state.value}")

        # Manual reset
        if _HAS_MSVCRT and msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch.lower() == 't':
                manual_reset = True
                print("[manual] 'T' pressed — forcing episode restart.")
                break

        env.step(dt)
        
        t_episode += dt

    # Convert logs
    time_log = np.array(time_log)
    base_pos_log = np.array(base_pos_log)
    ee_pos_log = np.array(ee_pos_log)
    dist_to_bulb_log = np.array(dist_to_bulb_log)
    dist_to_fixture_log = np.array(dist_to_fixture_log)
    base_vel_log = np.array(base_vel_log)
    arm_vel_norm_log = np.array(arm_vel_norm_log)

    # Plot
    wall_stop_pos_tuple = None
    if controller._stop_position is not None:
        wall_stop_pos_tuple = (controller._stop_position.x, controller._stop_position.y)
    
    plot_episode_metrics(
        time_log,
        state_log,
        base_pos_log,
        ee_pos_log,
        dist_to_bulb_log,
        dist_to_fixture_log,
        base_vel_log,
        arm_vel_norm_log,
        filename=f"episode_{episode_idx}_obstacle_avoidance_metrics.html",
        start_pos=initial_base_pose,
        bulb_pos_xy=(bulb_position[0], bulb_position[1]),
        wall_stop_pos=wall_stop_pos_tuple,
        bulb_pos_3d=bulb_position,
        fixture_pos_3d=fixture_position,
        TaskState=TaskState,
        robot_name="Frankie",
        obstacles=obstacles,
        world_bounds=world_bounds
    )

    # Cleanup and reset for next episode
    print(f"Cleaning up episode {episode_idx}...")
    
    # Reset robot
    frankie_agent._robot.q[:] = initial_joint_config
    frankie_agent._robot.qd[:] = 0.0
    frankie_agent._base_pose = np.array(initial_base_pose)
    frankie_agent._update_base_transform()
    
    # Hide all visuals from this episode
    tower.T = sm.SE3(50, 50, 0.5)
    bulb_visual.T = sm.SE3(50, 50, 0.5)
    
    # Hide obstacles
    for obs_visual in obstacle_visuals:
        obs_visual.T = sm.SE3(50, 50, 0.5)
    
    # Hide path markers
    for marker in path_markers:
        marker.T = sm.SE3(50, 50, 0.5)
    
    env.step(0.0)
    print(f"Episode {episode_idx} cleanup complete.")


def main() -> None:
    """Continuously run episodes."""
    
    env = swift.Swift()
    env.launch(realtime=True)
    
    temp_robot = rtb.models.URDF.Frankie()
    n_joints = temp_robot.n
    start_config = np.zeros(n_joints)
    start_config[2:9] = [0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4]
    start_base_pose = (0.0, 0.0, 0.0)
    
    initial_joint_config = start_config.copy()
    initial_base_pose = start_base_pose

    frankie_agent = FrankieAgent("frankie", start_config, start_base_pose)
    frankie_agent.register(env)

    try:
        episode = 1
        while True:
            print(f"\n\n########## EPISODE {episode} WITH OBSTACLES ##########")
            run_episode(env, frankie_agent, initial_joint_config, initial_base_pose, episode_idx=episode, first_run=(episode == 1))
            episode += 1
            print("Episode finished. Starting next episode (press Ctrl+C to stop)...")
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        print("Closing environment.")
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
