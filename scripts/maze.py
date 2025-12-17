from __future__ import annotations

import random
import numpy as np
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import swift

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from frankie import FrankieController, FrankieControllerParams, TaskState, FrankieAgent
from generators.maze_generator import get_maze
from core.algebra import Vector3
from core.astar import a_star_search
from plotting.maze import plot_episode_metrics


YELLOW_COLOR = "\033[93m"
RESET_COLOR = "\033[0m"

BLUE_BG_COLOR = "\033[104m"
RED_BG_COLOR = "\033[101m"
GREEN_BG_COLOR = "\033[102m"
YELLOW_BG_COLOR = "\033[103m"

MAZE_FLOOR_COLOR = (0.25, 0.25, 0.25, 1.0)
END_AREA_FLOOR_COLOR = (0.5, 0.5, 0.5, 1.0)
MAZE_WALL_COLOR = (0.2, 0.3, 0.6, 1.0)
END_POS_PILLAR_COLOR = (0.5, 0.3, 0.1, 1.0)

BULB_COLOR_YELLOW = (0.9, 0.8, 0.2, 1.0)
BULB_COLOR_BLUE = (0.2, 0.4, 0.9, 1.0)
BULB_COLOR_GREEN = (0.2, 0.9, 0.2, 1.0)

BULB_COLOR_MAP = {
    "yellow": BULB_COLOR_YELLOW,
    "blue": BULB_COLOR_BLUE,
    "green": BULB_COLOR_GREEN,
}

BULB_RADIUS = 0.03
FLOOR_HEIGHT = 0.02
WALL_HEIGHT = 0.5
WORLD_CELL_SIZE = 0.4
DT = 0.05
EPISODE_TIME = 600.0

bulb_color_name: str | None = None
placed_bulb_color_name: str | None = None


# FrankieAgent is now imported from frankie package


def cell_to_world(row: int, col: int, rows: int, cols: int, cell_size: float):
    width = cols * cell_size
    height = rows * cell_size
    origin_x = -width / 2.0
    origin_y = height / 2.0
    x = origin_x + (col + 0.5) * cell_size
    y = origin_y - (row + 0.5) * cell_size
    return x, y


def add_floors(env: swift.Swift, rows: int, cols: int, maze_rows: int, cell_size: float):
    width = cols * cell_size
    total_depth = rows * cell_size
    maze_depth = maze_rows * cell_size
    final_depth = total_depth - maze_depth

    top = total_depth / 2.0

    maze_center_y = top - maze_depth / 2.0
    floor_maze = sg.Cuboid(scale=[width, maze_depth, FLOOR_HEIGHT], color=MAZE_FLOOR_COLOR)
    floor_maze.T = sm.SE3(0.0, maze_center_y, 0.0)
    env.add(floor_maze)

    if final_depth > 1e-6:
        final_center_y = -maze_depth / 2.0
        floor_final = sg.Cuboid(scale=[width, final_depth, FLOOR_HEIGHT], color=END_AREA_FLOOR_COLOR)
        floor_final.T = sm.SE3(0.0, final_center_y, 0.0)
        env.add(floor_final)


def add_maze_walls(env: swift.Swift, world: np.ndarray, cell_size: float, maze_rows: int):
    rows, cols = world.shape
    for r in range(min(maze_rows, rows)):
        for c in range(cols):
            if world[r, c] == 1:
                x, y = cell_to_world(r, c, rows, cols, cell_size)
                wall = sg.Cuboid(scale=[cell_size, cell_size, WALL_HEIGHT], color=MAZE_WALL_COLOR)
                wall.T = sm.SE3(x, y, WALL_HEIGHT / 2.0)
                env.add(wall)


def pick_random_free_cell(world: np.ndarray, region_rows: tuple[int, int], min_dist: int = 0, avoid_cell=None):
    rows, cols = world.shape
    r0, r1 = region_rows
    candidates = []
    for r in range(max(0, r0), min(rows, r1)):
        for c in range(cols):
            if world[r, c] == 0:
                if avoid_cell is not None:
                    dr = r - avoid_cell[0]
                    dc = c - avoid_cell[1]
                    if abs(dr) + abs(dc) < min_dist:
                        continue
                candidates.append((r, c))
    if not candidates:
        return None
    return random.choice(candidates)


def find_room_center_cell(world: np.ndarray, room_start_row: int):
    rows, cols = world.shape
    center_col = cols // 2
    for r in range(rows - 2, room_start_row, -1):
        if world[r, center_col] == 0:
            return r, center_col
    for r in range(rows - 2, room_start_row, -1):
        for c in range(cols):
            if world[r, c] == 0:
                return r, c
    return None


def compute_path_waypoints(world: np.ndarray, src_cell, dest_cell, cell_size: float):
    rows, cols = world.shape
    grid = [[0 if world[i, j] == 0 else 1 for j in range(cols)] for i in range(rows)]
    path = a_star_search(grid, src_cell, dest_cell)
    if not path:
        return []
    waypoints = []
    for (r, c) in path:
        x, y = cell_to_world(r, c, rows, cols, cell_size)
        waypoints.append(Vector3(x, y, 0.0))
    return waypoints


def choose_bulb_sphere(bulb_spheres: dict[str, list[sg.Sphere]],
                       color_name: str,
                       fixture_sphere: sg.Sphere | None,
                       extra_forbidden=None) -> sg.Sphere:
    if extra_forbidden is None:
        extra_forbidden = []
    candidates = bulb_spheres[color_name]
    for s in candidates:
        if s is fixture_sphere:
            continue
        if s in extra_forbidden:
            continue
        return s
    return candidates[0]


def run_episode(env: swift.Swift, frankie_agent: FrankieAgent, second_agent: FrankieAgent, world: np.ndarray, episode_idx: int):
    global bulb_color_name, placed_bulb_color_name

    placed_bulb_count_R1 = 0 
    placed_bulb_count_R2 = 0
    

    print(f"[{YELLOW_COLOR}Environment{RESET_COLOR}] ===== EPISODE {episode_idx} =====")
    print(f"[{YELLOW_COLOR}Environment{RESET_COLOR}] Generating environment...")

    rows, cols = world.shape
    cell_size = WORLD_CELL_SIZE
    far_pos = (1000.0, 1000.0, 1000.0)

    maze_rows = min(rows, cols)
    room_start_row = maze_rows

    print(f"[{YELLOW_COLOR}Environment{RESET_COLOR}] Adding walls and floor")
    add_floors(env, rows, cols, maze_rows, cell_size)
    add_maze_walls(env, world, cell_size, maze_rows)

    width = cols * cell_size
    total_depth = rows * cell_size
    maze_depth = maze_rows * cell_size
    top = total_depth / 2.0
    boundary_y = top - maze_depth
    line_depth = 0.06
    line_z = FLOOR_HEIGHT * 0.6

    caution_total_width = width * 0.6
    segments = 6
    seg_width = caution_total_width / segments
    start_x = -caution_total_width / 2.0
    for i in range(segments):
        seg_color = (1.0, 1.0, 0.0, 1.0) if i % 2 == 0 else (0.05, 0.05, 0.05, 1.0)
        cx = start_x + (i + 0.5) * seg_width
        caution_block = sg.Cuboid(
            scale=[seg_width, line_depth, line_z],
            pose=sm.SE3(cx, boundary_y - line_depth / 2.0, line_z / 2.0),
            color=seg_color,
        )
        env.add(caution_block)

    total_depth = rows * cell_size
    width = cols * cell_size
    bottom_y = -total_depth / 2.0
    extra_depth = 1.0
    sorting_floor_center_y = bottom_y - extra_depth / 2.0
    sorting_floor = sg.Cuboid(scale=[width, extra_depth, FLOOR_HEIGHT], color=END_AREA_FLOOR_COLOR)
    sorting_floor.T = sm.SE3(0.0, sorting_floor_center_y, 0.0)
    env.add(sorting_floor)

    bulb_spheres: dict[str, list[sg.Sphere]] = {c: [] for c in BULB_COLOR_MAP.keys()}
    for color_name, rgba in BULB_COLOR_MAP.items():
        for _ in range(3):
            s = sg.Sphere(
                radius=BULB_RADIUS,
                pose=sm.SE3(*far_pos),
                color=rgba,
            )
            env.add(s)
            bulb_spheres[color_name].append(s)

    fixture_sphere_by_color: dict[str, sg.Sphere | None] = {c: None for c in BULB_COLOR_MAP.keys()}

    current_fixture_sphere: sg.Sphere | None = None
    current_active_sphere: sg.Sphere | None = None

    pending_bulbs: list[tuple[sg.Sphere, str]] = []
    current_second_sphere: sg.Sphere | None = None
    current_second_color: str | None = None
    second_busy = False

    start_cell = pick_random_free_cell(world, (1, maze_rows - 1))
    if start_cell is None:
        start_cell = (rows // 2, cols // 2)

    print(f"[{YELLOW_COLOR}Environment{RESET_COLOR}] Start position found")

    start_x, start_y = cell_to_world(start_cell[0], start_cell[1], rows, cols, cell_size)
    frankie_agent._base_pose[:] = np.array([start_x, start_y, 0.0])
    frankie_agent._update_base_transform()

    table_pillar_height = 0.45
    table_top_thickness = 0.05

    end_row = maze_rows + 1 if maze_rows + 1 < rows else maze_rows
    end_col = cols // 2
    end_x, end_y = cell_to_world(end_row, end_col, rows, cols, cell_size)

    print(f"[{YELLOW_COLOR}Environment{RESET_COLOR}] End position fixed")

    table_pillar = sg.Cylinder(
        radius=0.12,
        length=table_pillar_height,
        color=END_POS_PILLAR_COLOR,
    )
    table_pillar.T = sm.SE3(end_x, end_y, table_pillar_height / 2.0)
    env.add(table_pillar)

    table_top = sg.Cuboid(
        scale=[0.6, 0.6, table_top_thickness],
        pose=sm.SE3(end_x, end_y, table_pillar_height + table_top_thickness / 2.0),
        color=END_POS_PILLAR_COLOR,
    )
    env.add(table_top)

    fixture_z = table_pillar_height + table_top_thickness + BULB_RADIUS + 0.02
    fixture_position = (end_x, end_y, fixture_z)

    wall_height_second = 1.4
    wall_thickness_second = 0.3
    sorting_wall_center_y = sorting_floor_center_y - extra_depth / 2.0 + wall_thickness_second / 2.0
    sorting_wall = sg.Cuboid(
        scale=[width, wall_thickness_second, wall_height_second],
        pose=sm.SE3(0.0, sorting_wall_center_y, wall_height_second / 2.0),
        color=MAZE_FLOOR_COLOR,
    )
    env.add(sorting_wall)

    base_size = 0.18
    base_depth = 0.04
    base_height = base_size
    wall_front_y = sorting_wall_center_y + wall_thickness_second / 2.0
    base_center_y = wall_front_y + base_depth / 2.0
    peg_center_z = 0.85

    fixture_positions_second: dict[str, tuple[float, float, float]] = {}
    offsets_x = [-0.5, 0.0, 0.5]
    color_order = ["yellow", "blue", "green"]
    light_grey = (0.8, 0.8, 0.8, 1.0)
    for color_name, off_x in zip(color_order, offsets_x):
        px = off_x
        pz = peg_center_z

        base_block = sg.Cuboid(
            scale=[base_size, base_depth, base_height],
            pose=sm.SE3(px, base_center_y, pz),
            color=light_grey,
        )
        env.add(base_block)

        small_size = base_size * 0.5
        small_depth = 0.04
        small_height = small_size
        small_center_y = base_center_y + (base_depth + small_depth) / 2.0

        small_block = sg.Cuboid(
            scale=[small_size, small_depth, small_height],
            pose=sm.SE3(px, small_center_y, pz),
            color=light_grey,
        )
        env.add(small_block)

        color_height = 0.02
        color_block = sg.Cuboid(
            scale=[base_size, base_depth, color_height],
            pose=sm.SE3(px, base_center_y, pz + base_height / 2.0 + color_height / 2.0),
            color=BULB_COLOR_MAP[color_name],
        )
        env.add(color_block)

        peg_front_face_y = small_center_y + small_depth / 2.0
        fx = px
        fy = peg_front_face_y + 0.01
        fz = pz
        fixture_positions_second[color_name] = (fx, fy, fz)

    any_fixture = fixture_positions_second["yellow"]
    second_start_x = any_fixture[0]
    second_start_y = any_fixture[1] + 0.8
    second_agent._base_pose[:] = np.array([second_start_x, second_start_y, -np.pi / 2.0])
    second_agent._update_base_transform()

    R1_current_start_base = frankie_agent.base_state().copy()[:2]
    R2_current_start_base = second_agent.base_state().copy()[:2]

    bulb_cell = pick_random_free_cell(world, (1, maze_rows - 1), min_dist=10, avoid_cell=start_cell)
    if bulb_cell is None:
        bulb_cell = pick_random_free_cell(world, (1, maze_rows - 1))
        if bulb_cell is None:
            bulb_cell = (start_cell[0], min(start_cell[1] + 5, cols - 2))

    print(f"[{YELLOW_COLOR}Environment{RESET_COLOR}] Bulb position found")

    bulb_x, bulb_y = cell_to_world(bulb_cell[0], bulb_cell[1], rows, cols, cell_size)
    bulb_z = FLOOR_HEIGHT / 2.0 + BULB_RADIUS
    bulb_position = (bulb_x, bulb_y, bulb_z)

    bulb_color_name = random.choice(["yellow", "blue", "green"])
    current_active_sphere = choose_bulb_sphere(
        bulb_spheres,
        bulb_color_name,
        fixture_sphere_by_color[bulb_color_name],
    )
    current_active_sphere.T = sm.SE3(*bulb_position)

    print(f"[{YELLOW_COLOR}Environment{RESET_COLOR}] Creating controller...")
    controller_params = FrankieControllerParams(
        linear_velocity_gain=0.8,
        angular_velocity_gain=2.5,
        max_linear_velocity=0.7,
        max_angular_velocity=2.5,
        base_nav_tolerance=0.06,
        arm_gain=12.0,
        pinv_damping=0.5,
        q_dot_limit=1.5,
        pos_tolerance=2.5e-2,
        screw_rotations=2.0,
        grasp_height_offset=0.06,
        bulb_approach_distance=0.6,
        wall_approach_distance=1.0,
        wall_stop_distance=0.8,
        start_position=(start_x, start_y),
        wall_thickness=0.1,
        arm_safety_margin=0.02,
        maze_to_bulb_distance_threshold=1.0,
    )
    controller = FrankieController(controller_params, name="MazeGrasper", color_code="\033[92m", use_base_motion=True)

    second_controller_params = FrankieControllerParams(
        linear_velocity_gain=0.8,
        angular_velocity_gain=2.5,
        max_linear_velocity=0.7,
        max_angular_velocity=2.5,
        base_nav_tolerance=0.06,
        arm_gain=12.0,
        pinv_damping=0.5,
        q_dot_limit=1.5,
        pos_tolerance=2.5e-2,
        screw_rotations=2.0,
        grasp_height_offset=0.06,
        bulb_approach_distance=1.0,
        wall_approach_distance=1.0,
        wall_stop_distance=1.0,
        start_position=(second_start_x, second_start_y),
        wall_thickness=wall_thickness_second,
        arm_safety_margin=0.02,
        maze_to_bulb_distance_threshold=1.0,
    )
    second_controller = FrankieController(
        second_controller_params,
        name="BulbScrewer",
        color_code="\033[38;5;208m",
        use_base_motion=True,
    )

    world_for_controller = np.where((world == 1) | (world == 2), 1, 0)
    controller.set_world(world_for_controller, cell_size)
    controller.set_bulb_position(bulb_position)
    controller.set_end_position((end_x, end_y, 0.0))
    controller.set_wall_fixture(
        fixture_position,
        orientation=0.0,
        wall_x=None,
        wall_thickness=None,
    )

    controller.start_task(screw_bulb=False)

    # Log for Frankie metrics
    time_log_1, state_log_1, base_pos_log_1, ee_pos_log_1 = [], [], [], []
    dist_to_bulb_log_1, dist_to_fixture_log_1, base_vel_log_1, arm_vel_norm_log_1 = [], [], [], []
    # Log for second Frankie metrics
    time_log_2, state_log_2, base_pos_log_2, ee_pos_log_2 = [], [], [], []
    dist_to_bulb_log_2, dist_to_fixture_log_2, base_vel_log_2, arm_vel_norm_log_2 = [], [], [], []

    # Store original positions
    bulb_position_1 = bulb_position 
    fixture_position_1 = fixture_position
    current_bulb_position_2 = None
    current_fixture_target_2 = None

    t = 0.0
    print(f"[{YELLOW_COLOR}Environment{RESET_COLOR}] Starting simulation")
    while t < EPISODE_TIME:
        base_state = frankie_agent.base_state()
        robot = frankie_agent._robot
        qdot, base_cmd, current_state = controller.compute_control(robot, base_state)
        frankie_agent.apply_velocity_cmd(qdot, base_cmd, dt=DT)

        T_ee = robot.fkine(robot.q)
        ee_pos = np.asarray(T_ee.t, dtype=float).reshape(3)

        # Log 1
        time_log_1.append(t)
        state_log_1.append(current_state.value)
        base_pos_log_1.append(base_state.copy())
        ee_pos_log_1.append(ee_pos.copy())
        dist_to_bulb_log_1.append(float(np.linalg.norm(ee_pos - np.array(bulb_position_1))))
        dist_to_fixture_log_1.append(float(np.linalg.norm(ee_pos - np.array(fixture_position_1))))
        base_vel_1 = float(np.linalg.norm([base_cmd[0], base_cmd[1]])) if base_cmd is not None else 0.0
        base_vel_log_1.append(base_vel_1)
        arm_vel_norm_log_1.append(float(np.linalg.norm(qdot[2:])))


        base_state_second = second_agent.base_state()
        robot_second = second_agent._robot
        qdot_second, base_cmd_second, current_state_second = second_controller.compute_control(
            robot_second, base_state_second
        )
        second_agent.apply_velocity_cmd(qdot_second, base_cmd_second, dt=DT)

        T_ee_second = robot_second.fkine(robot_second.q)
        ee_pos_second = np.asarray(T_ee_second.t, dtype=float).reshape(3)

        # Log 2
        if second_busy and current_second_sphere is not None:
             current_bulb_position_2 = fixture_position 
             current_fixture_target_2 = fixture_positions_second[current_second_color] 
        else:
             current_bulb_position_2 = (np.nan, np.nan, np.nan)
             current_fixture_target_2 = (np.nan, np.nan, np.nan)

        time_log_2.append(t)
        state_log_2.append(current_state_second.value)
        base_pos_log_2.append(base_state_second.copy())
        ee_pos_log_2.append(ee_pos_second.copy())
        dist_to_bulb_2 = float(np.linalg.norm(ee_pos_second - np.array(current_bulb_position_2))) if not np.isnan(current_bulb_position_2[0]) else 0.0
        dist_to_fixture_2 = float(np.linalg.norm(ee_pos_second - np.array(current_fixture_target_2))) if not np.isnan(current_fixture_target_2[0]) else 0.0

        dist_to_bulb_log_2.append(dist_to_bulb_2)
        dist_to_fixture_log_2.append(dist_to_fixture_2)
        base_vel_2 = float(np.linalg.norm([base_cmd_second[0], base_cmd_second[1]])) if base_cmd_second is not None else 0.0
        base_vel_log_2.append(base_vel_2)
        arm_vel_norm_log_2.append(float(np.linalg.norm(qdot_second[2:])))

        if (
            current_active_sphere is not None
            and controller._grasped
            and current_state in [
                TaskState.GRASP_BULB,
                TaskState.TRANSPORT_TO_WALL,
                TaskState.NAVIGATE_IN_MAZE,
                TaskState.STOP_IN_FRONT_OF_WALL,
                TaskState.APPROACH_WALL,
            ]
        ):
            if np.all(np.isfinite(ee_pos)):
                current_active_sphere.T = sm.SE3(*ee_pos)

        T_ee_second = robot_second.fkine(robot_second.q)
        ee_pos_second = np.asarray(T_ee_second.t, dtype=float).reshape(3)

        if (
            current_second_sphere is not None
            and second_controller._grasped
            and current_state_second in [
                TaskState.GRASP_BULB,
                TaskState.TRANSPORT_TO_WALL,
                TaskState.STOP_IN_FRONT_OF_WALL,
                TaskState.APPROACH_WALL,
                TaskState.SCREW_BULB,
            ]
        ):
            if np.all(np.isfinite(ee_pos_second)):
                current_second_sphere.T = sm.SE3(*ee_pos_second)

        if second_busy and second_controller.is_task_complete():
            
            # Plot for second robot completed task
            placed_bulb_count_R2 += 1
            current_bulb_position_2_plot = fixture_position
            current_fixture_target_2_plot = fixture_positions_second[current_second_color]
            current_R2_stop_pos = (second_controller._stop_position.x, second_controller._stop_position.y)

            print(f"[{YELLOW_COLOR}Environment{RESET_COLOR}] R2 Task complete ({placed_bulb_count_R2}). Generating plot...")
        
           
            plot_episode_metrics(
                time_log_2,
                state_log_2,
                base_pos_log_2,
                ee_pos_log_2,
                dist_to_bulb_log_2,
                dist_to_fixture_log_2,
                base_vel_log_2,
                arm_vel_norm_log_2,
                filename=f"R2_task_{placed_bulb_count_R2 }_metrics.html",
                start_pos=R2_current_start_base,
                bulb_pos_xy=(current_bulb_position_2_plot[0], current_bulb_position_2_plot[1]), 
                wall_stop_pos=current_R2_stop_pos, 
                bulb_pos_3d=current_bulb_position_2_plot,
                fixture_pos_3d=current_fixture_target_2_plot, 
                TaskState=TaskState,
                robot_name="BulbScrewer (R2)" 
            )

            time_log_2, state_log_2, base_pos_log_2, ee_pos_log_2 = [], [], [], []
            dist_to_bulb_log_2, dist_to_fixture_log_2, base_vel_log_2, arm_vel_norm_log_2 = [], [], [], []
            R2_current_start_base = second_agent.base_state().copy()[:2]

            second_busy = False
            if current_second_sphere is not None and current_second_color is not None:
                final_pos = fixture_positions_second[current_second_color]
                current_second_sphere.T = sm.SE3(*final_pos)
                fixture_sphere_by_color[current_second_color] = current_second_sphere
            current_second_sphere = None
            current_second_color = None

        if (not second_busy) and pending_bulbs:
            sphere_to_pick, color_to_pick = pending_bulbs.pop(0)
            current_second_sphere = sphere_to_pick
            current_second_color = color_to_pick
            second_busy = True
            second_controller.set_bulb_position(fixture_position)
            fixture_target = fixture_positions_second[color_to_pick]
            second_controller.set_end_position(fixture_target)
            second_controller.set_wall_fixture(
                fixture_target,
                orientation=-np.pi / 2.0,
                wall_x=None,
                wall_thickness=second_controller.params.wall_thickness,
            )
            fx, fy, fz = fixture_target
            stop_y = fy + second_controller.params.wall_stop_distance
            second_controller._stop_position = Vector3.from_xy(fx, stop_y, fz)
            second_controller.reset_task()
            second_controller.start_task(screw_bulb=True)

        if controller.is_task_complete() and current_active_sphere is not None:

            placed_bulb_count_R1 += 1
            current_R1_stop_pos = (controller._stop_position.x, controller._stop_position.y)

            placed_bulb_color_name = bulb_color_name

            if placed_bulb_color_name == "green":
                bg = GREEN_BG_COLOR
            elif placed_bulb_color_name == "blue":
                bg = BLUE_BG_COLOR
            else:
                bg = YELLOW_BG_COLOR

            color_block = f"{bg} {RESET_COLOR}"
            print(
                f"[{YELLOW_COLOR}Environment{RESET_COLOR}] Placed bulb of color {color_block} "
                f"({placed_bulb_color_name})"
            )

            current_active_sphere.T = sm.SE3(*fixture_position)
            pending_bulbs.append((current_active_sphere, placed_bulb_color_name))
        
            # Plot for first robot completed task
            plot_episode_metrics(
                time_log_1,
                state_log_1,
                base_pos_log_1,
                ee_pos_log_1,
                dist_to_bulb_log_1,
                dist_to_fixture_log_1,
                base_vel_log_1,
                arm_vel_norm_log_1,
                filename=f"R1_task_{placed_bulb_count_R1 }_metrics.html",
                start_pos=R1_current_start_base,
                bulb_pos_xy=(bulb_position_1[0], bulb_position_1[1]), 
                wall_stop_pos=current_R1_stop_pos, 
                bulb_pos_3d=bulb_position_1,
                fixture_pos_3d=fixture_position,
                TaskState=TaskState,
                robot_name="MazeGrasper (R1)" 
            )
                
            time_log_1, state_log_1, base_pos_log_1, ee_pos_log_1 = [], [], [], []
            dist_to_bulb_log_1, dist_to_fixture_log_1, base_vel_log_1, arm_vel_norm_log_1 = [], [], [], []
            R1_current_start_base = frankie_agent.base_state().copy()[:2]

            current_base = frankie_agent.base_state()
            new_bulb_cell = pick_random_free_cell(world, (1, maze_rows - 1))
            print(f"[{YELLOW_COLOR}Environment{RESET_COLOR}] Found new random position for bulb")
            if new_bulb_cell is None:
                new_bulb_cell = (rows // 2, cols // 2)

            new_bulb_x, new_bulb_y = cell_to_world(
                new_bulb_cell[0], new_bulb_cell[1], rows, cols, cell_size
            )
            new_bulb_z = FLOOR_HEIGHT / 2.0 + BULB_RADIUS
            new_bulb_position = (new_bulb_x, new_bulb_y, new_bulb_z)

            bulb_color_name = random.choice(["yellow", "blue", "green"])

            extra_forbidden = []
            for s, c in pending_bulbs:
                if c == bulb_color_name:
                    extra_forbidden.append(s)
            if current_second_sphere is not None and current_second_color == bulb_color_name:
                extra_forbidden.append(current_second_sphere)
            if fixture_sphere_by_color[bulb_color_name] is not None:
                extra_forbidden.append(fixture_sphere_by_color[bulb_color_name])

            current_active_sphere = choose_bulb_sphere(
                bulb_spheres,
                bulb_color_name,
                fixture_sphere_by_color[bulb_color_name],
                extra_forbidden=extra_forbidden,
            )
            current_active_sphere.T = sm.SE3(*new_bulb_position)

            controller.set_bulb_position(new_bulb_position)
            controller.params.start_position = (float(current_base[0]), float(current_base[1]))

            controller.reset_task()
            controller.start_task(screw_bulb=False)
            bulb_position_1 = new_bulb_position

        env.step(DT)
        t += DT

    

def main():
    episode = 1
    env = None
    try:
        while True:
            world = get_maze(total_width=31, total_height=37)

            env = swift.Swift()
            env.launch(realtime=True)

            n_joints = 9
            start_config = np.zeros(n_joints)
            start_config[2:9] = [0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4]
            start_base_pose = (0.0, 0.0, 0.0)

            frankie_agent = FrankieAgent("MazeGrasper", start_config, start_base_pose, fix_geometry=True)
            frankie_agent.register(env)

            second_agent = FrankieAgent("BulbScrewer", start_config, start_base_pose, fix_geometry=True)
            second_agent.register(env)

            run_episode(env, frankie_agent, second_agent, world, episode)
            
            # Check if window was closed during episode
            if not env.keep_running:
                print(f"\n[{YELLOW_COLOR}Environment{RESET_COLOR}] Swift window closed. Exiting simulation.")
                break

            env.close()
            episode += 1
    except KeyboardInterrupt:
        print(f"\n[{YELLOW_COLOR}Environment{RESET_COLOR}] Simulation stopped by user.")
    finally:
        if env is not None:
            if env.keep_running:
                print(f"[{YELLOW_COLOR}Environment{RESET_COLOR}] Closing environment.")
                try:
                    env.close()
                except Exception:
                    pass
            else:
                print(f"[{YELLOW_COLOR}Environment{RESET_COLOR}] Environment already closed.")


if __name__ == "__main__":
    main()
