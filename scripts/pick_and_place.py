"""
Frankie Pick-and-Place Simulation
Simulates Frankie robot executing a pick-and-place task with a lightbulb.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple
import time
import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import swift
import matplotlib as mtb
import matplotlib.pyplot as plt
import msvcrt
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from plotting.basic import plot_episode_metrics

_HAS_MSVCRT = True

mtb.rcParams.update(mtb.rcParamsDefault)

# Import controller
from frankie import FrankieController, FrankieControllerParams, TaskState
from frankie import FrankieAgent

# Visual parameters
COLOR_RED = (0.8, 0.2, 0.3, 1.0)
COLOR_BLUE = (0.2, 0.5, 0.8, 1.0)
COLOR_GREEN = (0.2, 0.8, 0.3, 1.0)
COLOR_YELLOW = (0.9, 0.8, 0.2, 1.0)
COLOR_WHITE = (1.0, 1.0, 1.0, 0.6)
COLOR_GRAY = (0.5, 0.5, 0.5, 1.0)
COLOR_BRICK = (0.7, 0.3, 0.2, 1.0)  # Brick/terracotta color

BULB_RADIUS = 0.03
FIXTURE_RADIUS = 0.02
FLOOR_HEIGHT = 0.02
episode = 0


def add_world_bounds(env: swift.Swift, bounds: Tuple[float, float, float, float]) -> sg.Cuboid:
    """Draw a translucent floor tile highlighting the workspace."""
    min_x, max_x, min_y, max_y = bounds
    width, depth = max_x - min_x, max_y - min_y
    floor = sg.Cuboid(scale=[width, depth, FLOOR_HEIGHT], color=COLOR_WHITE)
    floor.T = sm.SE3((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, 0.01)
    env.add(floor)
    return floor


def add_wall(env: swift.Swift, position: Tuple[float, float, float], size: Tuple[float, float, float]) -> sg.Cuboid:
    """Add a wall to the environment."""
    wall = sg.Cuboid(scale=size, color=COLOR_BRICK)
    wall.T = sm.SE3(*position)
    env.add(wall)
    return wall



def run_episode(env: swift.Swift, frankie_agent: FrankieAgent, initial_joint_config: np.ndarray,
                initial_base_pose: Tuple[float, float, float], episode_idx: int, first_run: bool = False):    
    """Run one pick-and-place episode (continuous: respawns new random bulb after each completion)."""

    # Simulation parameters
    dt = 0.05
    simulation_time = 60.0  # seconds per episode

    # Workspace
    world_bounds = (-4.0, 4.0, -4.0, 4.0)
    min_x, max_x, min_y, max_y = world_bounds

    # Scene: floor + wall + fixture
    add_world_bounds(env, world_bounds)
    wall_position = (3.5, 0.0, 0.5)
    wall_size = (0.1, 2.0, 1.2)
    wall = add_wall(env, wall_position, wall_size)
    
    screw_x = wall_position[0] - wall_size[0] / 2.0 - 0.02
    fixture_position = (screw_x, 0.0, 0.8)
    # Use a cylinder as the target fixture (like a screw socket)
    fixture_visual = sg.Cylinder(
        radius=FIXTURE_RADIUS * 1.2,
        length=0.04,  # Short cylinder depth
        pose=sm.SE3(*fixture_position) * sm.SE3.Ry(np.pi/2),  # Rotate to point along X-axis
        color=COLOR_GRAY,
    )
    env.add(fixture_visual)

    # Random bulb spawn helper
    rng = np.random.default_rng()
    robot_base_pos = (0.0, 0.0)
    min_distance_from_base = 1.8
    max_attempts = 100
    tower_height = 0.40
    bulb_z = tower_height + 0.05

    def spawn_random_bulb():
        bulb_position_local = None
        for attempt in range(max_attempts):
            # bulb_x = float(rng.uniform(max(0.5, min_x + 0.5), max_x - 1.0))
            # bulb_y = float(rng.uniform(max(0.5, min_y + 0.5), max_y - 0.5))
            
            bulb_x = float(rng.uniform(min_x + 0.5, max_x - 1.0))
            bulb_y = float(rng.uniform(min_y + 0.5, max_y - 0.5))

            dist_from_base = np.hypot(bulb_x - robot_base_pos[0], bulb_y - robot_base_pos[1])
            dist_from_wall = abs(bulb_x - wall_position[0])
            lateral_offset = abs(bulb_y)
            if dist_from_base >= min_distance_from_base and dist_from_wall >= 0.5 and lateral_offset >= 0.6:
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

    # Spawn first bulb
    bulb_position, tower, bulb_visual = spawn_random_bulb()
    manual_reset = False
    print(f"Initial bulb spawned at: {bulb_position}")

    # Controller
    controller_params = FrankieControllerParams(
        linear_velocity_gain=0.7,  # higher = faster, less smooth
        angular_velocity_gain=2.5, # higher = turns quicker, can oscillate
        max_linear_velocity=0.6,   # absolute limit on forward speed of the base [m/s]
        max_angular_velocity=2.0,  # absolute limit on rotational speed of the base [rad/s]
        base_nav_tolerance=0.05,   # base must get very close to bulb/stop point
        arm_gain=12.0,              # proportional gain on Cartesian error for the arm (higher = faster EE motion)
        pinv_damping=1,         # damping used in the Jacobian pseudo-inverse (higher = more stable but less accurate)
        q_dot_limit=1.5,           # absolute joint-velocity limit for each arm joint [rad/s]
        pos_tolerance=2.5e-2,        # EE must be within ~10 cm of target to switch states
        screw_rotations=2.0,       # how many full 360° turns the EE performs while screwing the bulb
        grasp_height_offset=0.06,  # vertical offset above the bulb center used for grasping [m]
        approach_distance=0.8,    # base stops a bit further from the pillar to avoid collision [m]
        wall_stop_distance=1.0,   # base stops further from wall to give arm more room [m]
    )
    controller = FrankieController(controller_params)
    controller.set_bulb_position(bulb_position)
    controller.set_wall_fixture(
        fixture_position,
        orientation=0.0,
        wall_x=wall_position[0],
        wall_thickness=wall_size[0],
    )
    controller.params.start_position = (0.0, 0.0)
    controller.start_task()

    # Logging
    time_log, state_log, base_pos_log, ee_pos_log = [], [], [], []
    dist_to_bulb_log, dist_to_fixture_log, base_vel_log, arm_vel_norm_log = [], [], [], []

    steps = int(simulation_time / dt)
    task_completed_once = False
    cooldown_counter = 0.0
    t_episode = 0.0


    while t_episode < simulation_time:
        base_state = frankie_agent.base_state()
        robot = frankie_agent._robot

        # Compute control
        qdot, base_cmd, current_state = controller.compute_control(robot, base_state)

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

        # Move bulb visual while grasped/transport
        if current_state in [TaskState.TRANSPORT_TO_WALL, TaskState.APPROACH_WALL, TaskState.SCREW_BULB]:
            bulb_visual.T = sm.SE3(*ee_pos)

        # Detect completion and respawn if cooled down
        if controller.is_task_complete():
            task_completed_once = True
            print(f"[t={t_episode:.2f}s] Task completed!")

            # hide old visuals
            tower.T = sm.SE3(10, 10, 10)
            bulb_visual.T = sm.SE3(10, 10, 10)
            
            break  # esce subito per fare il plot

        # State transition logging
        if len(state_log) > 1 and state_log[-1] != state_log[-2]:
            print(f"[t={t_episode:.2f}s] State: {current_state.value}")

        if _HAS_MSVCRT and msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch.lower() == 't':
                manual_reset = True
                print("[manual] 'T' pressed: triggering manual reset.")
                break

        # Step environment
        env.step(dt)
        
        t_episode += dt

    time_log = np.array(time_log)
    base_pos_log = np.array(base_pos_log)
    ee_pos_log = np.array(ee_pos_log)
    dist_to_bulb_log = np.array(dist_to_bulb_log)
    dist_to_fixture_log = np.array(dist_to_fixture_log)
    base_vel_log = np.array(base_vel_log)
    arm_vel_norm_log = np.array(arm_vel_norm_log)


    plot_episode_metrics(
        time_log,
        state_log,
        base_pos_log,
        ee_pos_log,
        dist_to_bulb_log,
        dist_to_fixture_log,
        base_vel_log,
        arm_vel_norm_log,
        filename=f"episode_{episode_idx}_metrics.html",
        start_pos=initial_base_pose,
        bulb_pos_xy=(bulb_position[0], bulb_position[1]),
        wall_stop_pos=controller._stop_position,
        bulb_pos_3d=bulb_position,
        fixture_pos_3d=fixture_position,
        TaskState=TaskState,
        robot_name="Frankie"
    )


    # --- Timeout handling ---
  
    if (not controller.is_task_complete()) or manual_reset:
        print(f"[t={t_episode:.2f}s] Episode timeout OR manual reset. Resetting robot.")

        frankie_agent._robot.q[:] = initial_joint_config
        frankie_agent._robot.qd[:] = 0.0
        frankie_agent._base_pose = np.array(initial_base_pose)
        frankie_agent._update_base_transform()

        tower.T = sm.SE3(50,50,0.5)
        bulb_visual.T = sm.SE3(50,50,0.5)

        env.step(0.0)
        return

def main() -> None:
    """Continuously run episodes until the user stops the program (Ctrl+C)."""
    
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
            print(f"\n\n########## EPISODE {episode} ##########")
            run_episode(env, frankie_agent, initial_joint_config, initial_base_pose, episode_idx=episode, first_run=(episode==1))
            episode += 1
            print("Episode finished. Restarting (press Ctrl+C to stop)...")
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

