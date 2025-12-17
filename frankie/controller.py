from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np
import roboticstoolbox as rtb

from core.algebra import Vector3, damped_pseudoinverse, wrap_angle, base_state_to_vector3
from core.astar import a_star_search


GREEN_COLOR = "\033[92m"
RESET_COLOR = "\033[0m"


class TaskState(Enum):
    IDLE = "idle"
    NAVIGATE_IN_MAZE = "navigate_in_maze"
    NAVIGATE_TO_BULB = "navigate_to_bulb"
    APPROACH_BULB = "approach_bulb"
    GRASP_BULB = "grasp_bulb"
    TRANSPORT_TO_WALL = "transport_to_wall"
    STOP_IN_FRONT_OF_WALL = "stop_in_front_of_wall"
    APPROACH_WALL = "approach_wall"
    SCREW_BULB = "screw_bulb"
    RETURN_TO_START = "return_to_start"
    RESET = "reset"


@dataclass
class FrankieControllerParams:
    linear_velocity_gain: float = 0.5
    angular_velocity_gain: float = 1.0
    max_linear_velocity: float = 0.5
    max_angular_velocity: float = 1.5
    base_nav_tolerance: float = 0.15

    arm_gain: float = 12.0
    pinv_damping: float = 0.3
    q_dot_limit: float = 1.5
    pos_tolerance: float = 2.5e-2
    screw_rotations: float = 2.0

    grasp_height_offset: float = 0.01

    bulb_approach_distance: float = 0.6
    wall_approach_distance: float = 1.0

    approach_distance: float = 0.7

    wall_stop_distance: float = 0.6
    start_position: Tuple[float, float] = (0.0, 0.0)
    wall_thickness: float = 0.1
    arm_safety_margin: float = 0.02

    maze_to_bulb_distance_threshold: float = 1.0


class FrankieController:
    def __init__(self,params: FrankieControllerParams,name: str = "FrankieController",color_code: str = GREEN_COLOR,use_base_motion: bool = True,) -> None:

        self.params = params
        self.state: TaskState = TaskState.IDLE

        self.bulb_position: Optional[Vector3] = None
        self.wall_fixture_position: Optional[Vector3] = None
        self.wall_fixture_orientation: Optional[float] = None
        self.wall_x_position: Optional[float] = None
        self.wall_surface_x: Optional[float] = None

        self._stop_position: Optional[Vector3] = None
        self._maze_waypoints: List[Vector3] = []
        self._maze_index: int = 0

        self._screw_angle: float = 0.0
        self._screw_target_angle: float = 0.0
        self._grasped: bool = False
        self._screw_bulb: bool = True

        self.world: Optional[np.ndarray] = None
        self.world_size: float = 1.0
        self._maze_target: Optional[str] = None

        self.end_position: Optional[Vector3] = None

        self._label_name: str = name
        self._color_code: str = color_code
        self.use_base_motion: bool = use_base_motion

    def set_bulb_position(self, position: Tuple[float, float, float] | Iterable[float]) -> None:
        self.bulb_position = Vector3.from_iterable(position)

    def set_end_position(self, position: Tuple[float, float, float] | Iterable[float]) -> None:
        self.end_position = Vector3.from_iterable(position)

    def set_wall_fixture(self,position: Tuple[float, float, float] | Iterable[float],orientation: float = 0.0,wall_x: Optional[float] = None,wall_thickness: Optional[float] = None,) -> None:
        """
        Set the wall fixture position and orientation for bulb screwing task.
        
        Configures the target position and orientation where the robot arm must place/screw a bulb
        into a wall fixture. 
        
        Args:
            position: 3D position of the wall fixture 
            orientation: Rotation angle for the end-effector at the fixture 
            wall_x: X-coordinate of the wall surface. 
            wall_thickness: Thickness of the wall in meters. Used to compute wall_surface_x position.
        
        """
        
        self.wall_fixture_position = Vector3.from_iterable(position)
        self.wall_fixture_orientation = float(orientation)

        if wall_x is not None:
            self.wall_x_position = float(wall_x)
            thickness = wall_thickness if wall_thickness is not None else self.params.wall_thickness
            self.wall_surface_x = wall_x - (thickness / 2.0)

            stop_x = wall_x - self.params.wall_stop_distance
            stop_y = float(position[1])
            stop_z = float(position[2]) if len(tuple(position)) >= 3 else 0.0
            self._stop_position = Vector3.from_xy(stop_x, stop_y, stop_z)
        else:
            stop_x = float(position[0])
            stop_y = float(position[1]) - self.params.wall_stop_distance
            stop_z = float(position[2]) if len(tuple(position)) >= 3 else 0.0
            self._stop_position = Vector3.from_xy(stop_x, stop_y, stop_z)

    def set_maze_waypoints(self,waypoints: Iterable[Vector3] | Iterable[Tuple[float, float, float]] | Iterable[Iterable[float]],) -> None:
        result: List[Vector3] = []
        for w in waypoints:
            if isinstance(w, Vector3):
                result.append(w)
            else:
                result.append(Vector3.from_iterable(w))
        self._maze_waypoints = result
        self._maze_index = 0

    def set_world(self, world: np.ndarray, world_size: float) -> None:
        self.world = np.array(world, copy=True)
        self.world_size = float(world_size)

    def _print_state_change(self, new_state: TaskState, reason: str = "") -> None:
        if new_state == self.state:
            return

        if new_state == TaskState.NAVIGATE_IN_MAZE:
            if self._maze_target == "bulb":
                msg = "Navigating through maze to reach bulb"
            elif self._maze_target == "end":
                msg = "Navigating through maze to reach end position"
            else:
                msg = "Navigating through maze"
        elif new_state == TaskState.NAVIGATE_TO_BULB:
            msg = "Driving directly towards bulb"
        elif new_state == TaskState.APPROACH_BULB:
            msg = "Approaching bulb with arm"
        elif new_state == TaskState.GRASP_BULB:
            msg = "Grasping bulb"
        elif new_state == TaskState.TRANSPORT_TO_WALL:
            msg = "Driving directly towards end/wall position"
        elif new_state == TaskState.STOP_IN_FRONT_OF_WALL:
            msg = "Reached end/wall position with base"
        elif new_state == TaskState.APPROACH_WALL:
            msg = "Approaching end/wall position with arm"
        elif new_state == TaskState.SCREW_BULB:
            msg = "Screwing bulb into fixture"
        elif new_state == TaskState.RETURN_TO_START:
            msg = "Returning to start position"
        elif new_state == TaskState.RESET:
            msg = "Resetting arm to ready configuration"
        elif new_state == TaskState.IDLE:
            msg = "Task idle"
        else:
            msg = f"State changed to {new_state.value}"

        label = f"[{self._color_code}{self._label_name}{RESET_COLOR}]"
        state_tag = f"[{new_state.name}]"
        if reason:
            print(f"{label} {state_tag} {msg} ({reason})")
        else:
            print(f"{label} {state_tag} {msg}")

        self.state = new_state

    def start_task(self, screw_bulb: bool = True) -> None:
        self._screw_bulb = bool(screw_bulb)
        self._grasped = False
        self._screw_angle = 0.0
        self._screw_target_angle = 0.0
        self._maze_index = 0

        label = f"[{self._color_code}{self._label_name}{RESET_COLOR}]"
        print(f"{label} [START_TASK] start_task called (screw_bulb={self._screw_bulb})")

        if not self.use_base_motion:
            self._maze_target = None
            self._maze_waypoints = []
            if self.bulb_position is not None:
                self._print_state_change(TaskState.APPROACH_BULB, "start_task (no base motion)")
            else:
                self._print_state_change(TaskState.IDLE, "no bulb_position for stationary task")
            return

        if self.world is not None:
            self._maze_target = "bulb"
            self._maze_waypoints = []
            self._print_state_change(TaskState.NAVIGATE_IN_MAZE, "start_task with maze world")
        elif self._maze_waypoints:
            self._maze_target = "bulb"
            self._print_state_change(TaskState.NAVIGATE_IN_MAZE, "start_task with precomputed waypoints")
        else:
            self._maze_target = None
            self._print_state_change(TaskState.NAVIGATE_TO_BULB, "start_task without maze")

    def reset_task(self) -> None:
        self._print_state_change(TaskState.IDLE, "manual reset")
        self._grasped = False
        self._screw_angle = 0.0
        self._maze_index = 0
        self._maze_target = None

    def _compute_base_control(self, base_state: Sequence[float], target: Vector3) -> Tuple[float, float]:
        x = float(base_state[0])
        y = float(base_state[1])
        theta = float(base_state[2])

        gx = target.x
        gy = target.y

        dx = gx - x
        dy = gy - y
        rho = float(np.hypot(dx, dy))
        theta_star = float(np.arctan2(dy, dx))
        heading_err = wrap_angle(theta_star - theta)

        v = self.params.linear_velocity_gain * rho
        omega = self.params.angular_velocity_gain * heading_err

        v = float(np.clip(v, -self.params.max_linear_velocity, self.params.max_linear_velocity))
        omega = float(np.clip(omega, -self.params.max_angular_velocity, self.params.max_angular_velocity))

        return v, omega

    def _compute_maze_base_control(self, base_state: Sequence[float], target: Vector3) -> Tuple[float, float]:
        x = float(base_state[0])
        y = float(base_state[1])
        theta = float(base_state[2])

        gx = target.x
        gy = target.y

        dx = gx - x
        dy = gy - y
        rho = float(np.hypot(dx, dy))
        theta_star = float(np.arctan2(dy, dx))
        heading_err = wrap_angle(theta_star - theta)

        heading_threshold = 0.4
        if abs(heading_err) > heading_threshold:
            v = 0.0
        else:
            v = self.params.max_linear_velocity

        omega = self.params.angular_velocity_gain * heading_err

        v = float(np.clip(v, -self.params.max_linear_velocity, self.params.max_linear_velocity))
        omega = float(np.clip(omega, -self.params.max_angular_velocity, self.params.max_angular_velocity))

        if rho < self.params.base_nav_tolerance * 0.5:
            v = 0.0
            omega = 0.0

        return v, omega

    def _compute_arm_control(self,robot: rtb.ERobot,target_position: Vector3,base_state: Optional[Sequence[float]] = None,target_orientation: Optional[float] = None,) -> Tuple[np.ndarray, np.ndarray]:
        
        q = robot.q
        T_ee = robot.fkine(q)
        p_ee = np.asarray(T_ee.t, dtype=float).reshape(3)

        target_np = target_position.to_numpy()

        if base_state is not None:
            theta = float(base_state[2])
            R_base = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta), np.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            e_pos = R_base.T @ (target_np - p_ee)
        else:
            e_pos = target_np - p_ee

        if target_orientation is not None:
            current_rot = float(np.arctan2(T_ee.R[1, 0], T_ee.R[0, 0]))
            e_rot = wrap_angle(target_orientation - current_rot)
        else:
            e_rot = 0.0

        J = robot.jacob0(q)
        Jpos = J[:3, 2:]

        Jpinv = damped_pseudoinverse(Jpos, self.params.pinv_damping)

        qdot_arm = (self.params.arm_gain * (Jpinv @ e_pos.reshape(3, 1))).ravel()

        qdot = np.zeros(robot.n)
        qdot[2:] = qdot_arm

        if target_orientation is not None and np.linalg.norm(e_pos) < self.params.pos_tolerance:
            ee_idx = robot.n - 1
            qdot[ee_idx] += 0.5 * e_rot

        if abs(e_rot) > 0.01:
            ee_idx = robot.n - 1
            qdot[ee_idx] += 0.5 * e_rot

        return qdot, e_pos

    def _world_to_grid(self, pos: Vector3) -> Tuple[int, int]:
        """
        Convert a 3D world position to grid cell indices (row, col).
        """

        if self.world is None:
            raise ValueError("World grid not set")

        cell_size = self.world_size
        rows, cols = self.world.shape

        width = cols * cell_size
        height = rows * cell_size

        origin_x = -width / 2.0
        origin_y = height / 2.0

        col_f = (pos.x - origin_x) / cell_size - 0.5
        row_f = (origin_y - pos.y) / cell_size - 0.5

        col = int(round(col_f))
        row = int(round(row_f))

        row = int(np.clip(row, 0, rows - 1))
        col = int(np.clip(col, 0, cols - 1))
        return row, col

    def _grid_to_world(self, row: int, col: int, z: float = 0.0) -> Vector3:
        """
         Convert grid cell indices (row, col) to a 3D world position.

        Transforms discrete grid cell coordinates into continuous world coordinates.
        This is the inverse operation of _world_to_grid() and is used to convert

        """

        cell_size = self.world_size
        rows, cols = self.world.shape

        width = cols * cell_size
        height = rows * cell_size
        origin_x = -width / 2.0
        origin_y = height / 2.0

        x = origin_x + (float(col) + 0.5) * cell_size
        y = origin_y - (float(row) + 0.5) * cell_size
        return Vector3(x, y, z)

    def _find_nearest_free_cell(self, row: int, col: int) -> Tuple[int, int]:
        """
        Find the nearest free cell in the grid using BFS.
                
        This is used during path planning to ensure the destination or source cell
        for A* search is always walkable, 
        
        """
        if self.world is None:
            return row, col
        h, w = self.world.shape
        if self.world[row, col] == 0:
            return row, col

        from collections import deque

        visited = set()
        q = deque()
        q.append((row, col))
        visited.add((row, col))

        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        while q:
            r, c = q.popleft()
            for dr, dc in directions:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                    if self.world[nr, nc] == 0:
                        return nr, nc
                    visited.add((nr, nc))
                    q.append((nr, nc))

        return row, col

    def _plan_maze_path(self, base_vec: Vector3, target_pos: Vector3) -> None:
        if self.world is None:
            self._maze_waypoints = []
            self._maze_index = 0
            return

        try:
            src_row, src_col = self._world_to_grid(base_vec)
            dest_row, dest_col = self._world_to_grid(target_pos)
        except ValueError:
            self._maze_waypoints = []
            self._maze_index = 0
            return

        dest_row, dest_col = self._find_nearest_free_cell(dest_row, dest_col)

        h, w = self.world.shape
        grid = [[0 if self.world[i, j] == 0 else 1 for j in range(w)] for i in range(h)]
        path = a_star_search(grid, (src_row, src_col), (dest_row, dest_col))
        if not path:
            self._maze_waypoints = []
            self._maze_index = 0
            return

        waypoints: List[Vector3] = []
        for r, c in path:
            waypoints.append(self._grid_to_world(r, c, base_vec.z))

        self._maze_waypoints = waypoints
        self._maze_index = 0

    def _get_end_target_position(self) -> Optional[Vector3]:
        if self.end_position is not None:
            return self.end_position
        if self._stop_position is not None:
            return self._stop_position
        if self.wall_fixture_position is not None and self.wall_x_position is not None:
            stop_x = self.wall_x_position - self.params.wall_stop_distance
            stop_y = self.wall_fixture_position.y
            stop_z = self.wall_fixture_position.z
            self._stop_position = Vector3.from_xy(stop_x, stop_y, stop_z)
            return self._stop_position
        return self.wall_fixture_position

    def _base_distance(self, a: Vector3, b: Vector3) -> float:
        dx = a.x - b.x
        dy = a.y - b.y
        return float(np.hypot(dx, dy))

    def compute_control(self,robot: rtb.ERobot,base_state: Optional[Sequence[float]] = None,) -> Tuple[np.ndarray, Optional[Tuple[float, float]], TaskState]:
        
        if self.state == TaskState.IDLE:
            qdot_idle = np.zeros(robot.n)
            return qdot_idle, None, self.state

        if base_state is None:
            raise ValueError("base_state is required for control computation")

        q = robot.q
        base_vec = base_state_to_vector3(base_state)

        if self.state == TaskState.NAVIGATE_IN_MAZE and self.use_base_motion:
            target_pos: Optional[Vector3] = None
            if self._maze_target == "bulb":
                target_pos = self.bulb_position
            elif self._maze_target == "end":
                target_pos = self._get_end_target_position()

            if self._maze_target == "bulb" and self.bulb_position is not None:
                dist_to_bulb = self._base_distance(base_vec, self.bulb_position)
                if dist_to_bulb <= self.params.maze_to_bulb_distance_threshold:
                    self._print_state_change(
                        TaskState.NAVIGATE_TO_BULB,
                        "close enough to bulb, leaving maze path and switching to local approach",
                    )
                    qdot_zero = np.zeros(robot.n)
                    return qdot_zero, None, self.state

            if self._maze_target == "end":
                target_for_dist = self.wall_fixture_position or self.end_position
                if target_for_dist is not None:
                    dist_to_end = self._base_distance(base_vec, target_for_dist)
                    if dist_to_end <= self.params.wall_approach_distance:
                        self._print_state_change(
                            TaskState.STOP_IN_FRONT_OF_WALL,
                            "close enough to end, stopping base and switching to arm approach",
                        )
                        qdot_zero = np.zeros(robot.n)
                        return qdot_zero, None, self.state

            if self.state == TaskState.NAVIGATE_IN_MAZE:
                if (not self._maze_waypoints or self._maze_index >= len(self._maze_waypoints)) and target_pos is not None:
                    self._plan_maze_path(base_vec, target_pos)

                if not self._maze_waypoints or self._maze_index >= len(self._maze_waypoints):
                    if self._maze_target == "bulb":
                        self._print_state_change(
                            TaskState.NAVIGATE_TO_BULB,
                            "no more maze waypoints towards bulb",
                        )
                    elif self._maze_target == "end":
                        self._print_state_change(
                            TaskState.STOP_IN_FRONT_OF_WALL,
                            "no more maze waypoints towards end",
                        )
                        qdot_zero = np.zeros(robot.n)
                        return qdot_zero, None, self.state

                    qdot_zero = np.zeros(robot.n)
                    return qdot_zero, None, self.state

                current_wp = self._maze_waypoints[self._maze_index]
                dist_to_wp = self._base_distance(base_vec, current_wp)

                if dist_to_wp <= self.params.base_nav_tolerance:
                    self._maze_index += 1
                    if self._maze_index < len(self._maze_waypoints):
                        current_wp = self._maze_waypoints[self._maze_index]
                    else:
                        if self._maze_target == "bulb":
                            self._print_state_change(
                                TaskState.NAVIGATE_TO_BULB,
                                "reached last maze waypoint near bulb",
                            )
                        elif self._maze_target == "end":
                            self._print_state_change(
                                TaskState.STOP_IN_FRONT_OF_WALL,
                                "reached last maze waypoint near end",
                            )
                        qdot_zero = np.zeros(robot.n)
                        return qdot_zero, None, self.state

                v, omega = self._compute_maze_base_control(base_state, current_wp)

                qdot = np.zeros(robot.n)
                qdot[0] = omega
                qdot[1] = v

                if hasattr(robot, "qr"):
                    ready_config = robot.qr
                else:
                    ready_config = np.zeros(robot.n)

                e_q = ready_config - q
                arm_qdot = self.params.arm_gain * e_q
                arm_qdot = np.clip(arm_qdot, -self.params.q_dot_limit, self.params.q_dot_limit)
                qdot[2:] = arm_qdot[2:]

                return qdot, (v, omega), self.state

        if self.state == TaskState.NAVIGATE_TO_BULB and self.use_base_motion:
            if self.bulb_position is None:
                self._print_state_change(TaskState.IDLE, "bulb_position not set")
                qdot_zero = np.zeros(robot.n)
                return qdot_zero, None, self.state

            dist_to_bulb = self._base_distance(base_vec, self.bulb_position)

            if dist_to_bulb <= self.params.bulb_approach_distance:
                self._print_state_change(TaskState.APPROACH_BULB, "within approach distance of bulb")
            else:
                v, omega = self._compute_base_control(base_state, self.bulb_position)
                qdot = np.zeros(robot.n)
                qdot[0] = omega
                qdot[1] = v
                return qdot, (v, omega), self.state

        if self.state == TaskState.APPROACH_BULB:
            if self.bulb_position is None:
                self._print_state_change(TaskState.IDLE, "bulb_position lost during approach")
                qdot_zero = np.zeros(robot.n)
                return qdot_zero, None, self.state

            grasp_target = self.bulb_position

            qdot, e_pos = self._compute_arm_control(robot, grasp_target, base_state=base_state)
            qdot[0] = 0.0
            qdot[1] = 0.0

            if float(np.linalg.norm(e_pos)) <= self.params.pos_tolerance:
                self._grasped = True
                self._print_state_change(TaskState.GRASP_BULB, "end-effector reached bulb")

            return qdot, None, self.state

        if self.state == TaskState.GRASP_BULB:
            if self.bulb_position is None:
                self._print_state_change(TaskState.IDLE, "bulb_position lost during grasp")
                qdot_zero = np.zeros(robot.n)
                return qdot_zero, None, self.state

            safe_z = self.bulb_position.z + self.params.grasp_height_offset
            grasp_target = self.bulb_position.with_z(safe_z)

            qdot, e_pos = self._compute_arm_control(robot, grasp_target, base_state=base_state)
            qdot[0] = 0.0
            qdot[1] = 0.0

            if float(np.linalg.norm(e_pos)) <= self.params.pos_tolerance:
                if not self.use_base_motion:
                    self._print_state_change(
                        TaskState.APPROACH_WALL,
                        "lifted bulb, moving arm directly to wall (no base motion)",
                    )
                elif self.world is not None and (self.end_position is not None or self.wall_fixture_position is not None):
                    self._maze_target = "end"
                    self._maze_waypoints = []
                    self._maze_index = 0
                    self._print_state_change(
                        TaskState.NAVIGATE_IN_MAZE,
                        "lifted bulb, navigating maze to end (arm will go to rest while moving)",
                    )
                else:
                    self._print_state_change(
                        TaskState.TRANSPORT_TO_WALL,
                        "lifted bulb, driving directly to end/wall (no maze)",
                    )

            return qdot, None, self.state

        if self.state == TaskState.TRANSPORT_TO_WALL:
            if not self.use_base_motion:
                self._print_state_change(
                    TaskState.APPROACH_WALL,
                    "skipping base transport (no base motion)",
                )
                qdot_zero = np.zeros(robot.n)
                return qdot_zero, None, self.state

            if self._stop_position is None:
                if self.wall_fixture_position is None or self.wall_x_position is None:
                    self._print_state_change(TaskState.IDLE, "no wall fixture / stop position")
                    qdot_zero = np.zeros(robot.n)
                    return qdot_zero, None, self.state
                stop_x = self.wall_x_position - self.params.wall_stop_distance
                stop_y = self.wall_fixture_position.y
                stop_z = self.wall_fixture_position.z
                self._stop_position = Vector3.from_xy(stop_x, stop_y, stop_z)

            dist_to_stop = self._base_distance(base_vec, self._stop_position)

            if dist_to_stop <= self.params.base_nav_tolerance:
                self._print_state_change(
                    TaskState.STOP_IN_FRONT_OF_WALL,
                    "reached stop position in front of wall/end",
                )
                qdot_zero = np.zeros(robot.n)
                return qdot_zero, None, self.state

            v, omega = self._compute_base_control(base_state, self._stop_position)
            qdot = np.zeros(robot.n)
            qdot[0] = omega
            qdot[1] = v
            return qdot, (v, omega), self.state

        if self.state == TaskState.STOP_IN_FRONT_OF_WALL:
            qdot = np.zeros(robot.n)
            self._print_state_change(TaskState.APPROACH_WALL, "base stopped, switching to arm approach")
            return qdot, None, self.state

        if self.state == TaskState.APPROACH_WALL:
            target = self.wall_fixture_position or self.end_position
            if target is None:
                self._print_state_change(TaskState.IDLE, "no target (fixture/end) for arm approach")
                qdot_zero = np.zeros(robot.n)
                return qdot_zero, None, self.state

            # Check distance first to avoid unnecessary control computation
            T_ee = robot.fkine(robot.q)
            p_ee = np.asarray(T_ee.t, dtype=float).reshape(3)
            target_np = target.to_numpy()
            
            if base_state is not None:
                theta = float(base_state[2])
                R_base = np.array(
                    [
                        [np.cos(theta), -np.sin(theta), 0.0],
                        [np.sin(theta), np.cos(theta), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                e_pos_check = R_base.T @ (target_np - p_ee)
            else:
                e_pos_check = target_np - p_ee
            
            dist_to_target = float(np.linalg.norm(e_pos_check))
            
            # If close enough and not screwing, stop immediately to prevent oscillation
            if dist_to_target <= self.params.pos_tolerance:
                if self._screw_bulb and self.wall_fixture_position is not None:
                    self._screw_target_angle = self.params.screw_rotations * 2.0 * np.pi
                    self._print_state_change(TaskState.SCREW_BULB, "at fixture, start screwing")
                    # Still compute control for screwing transition
                    qdot, e_pos = self._compute_arm_control(
                        robot,
                        target,
                        base_state=base_state,
                        target_orientation=self.wall_fixture_orientation,
                    )
                    qdot[0] = 0.0
                    qdot[1] = 0.0
                    return qdot, None, self.state
                else:
                    # Stop arm control immediately to prevent shaking when placing bulb
                    self._grasped = False
                    self._print_state_change(TaskState.IDLE, "placed bulb at end, going idle")
                    qdot_zero = np.zeros(robot.n)
                    return qdot_zero, None, self.state

            # Only use orientation control if we need to screw the bulb
            orientation_to_use = self.wall_fixture_orientation if self._screw_bulb else None
            
            qdot, e_pos = self._compute_arm_control(
                robot,
                target,
                base_state=base_state,
                target_orientation=orientation_to_use,
            )
            qdot[0] = 0.0
            qdot[1] = 0.0

            return qdot, None, self.state

        if self.state == TaskState.SCREW_BULB:
            if self.wall_fixture_position is None:
                self._print_state_change(TaskState.IDLE, "fixture lost during screwing")
                qdot_zero = np.zeros(robot.n)
                return qdot_zero, None, self.state

            qdot = np.zeros(robot.n)

            ee_idx = robot.n - 1
            if self._screw_angle < self._screw_target_angle:
                screw_rate = 1.5
                dt = 0.05
                self._screw_angle += screw_rate * dt
                qdot[ee_idx] = screw_rate
            else:
                self._grasped = False
                if self.use_base_motion:
                    self._print_state_change(TaskState.RETURN_TO_START, "finished screwing, returning to start")
                else:
                    self._print_state_change(TaskState.RESET, "finished screwing, resetting arm (no base motion)")

            qdot[0] = 0.0
            qdot[1] = 0.0

            return qdot, None, self.state

        if self.state == TaskState.RETURN_TO_START:
            if not self.use_base_motion:
                self._print_state_change(TaskState.RESET, "skipping base return (no base motion)")
                qdot_zero = np.zeros(robot.n)
                return qdot_zero, None, self.state

            start_x, start_y = self.params.start_position
            start_vec = Vector3.from_xy(start_x, start_y, base_vec.z)

            dist_to_start = self._base_distance(base_vec, start_vec)

            if dist_to_start <= self.params.base_nav_tolerance:
                self._print_state_change(TaskState.RESET, "base back at start, resetting arm")
            else:
                # Check if we have A* waypoints to follow (from obstacle avoidance)
                if self._maze_waypoints and self._maze_index < len(self._maze_waypoints):
                    # Follow waypoints using same logic as NAVIGATE_IN_MAZE
                    current_wp = self._maze_waypoints[self._maze_index]
                    dist_to_wp = self._base_distance(base_vec, current_wp)

                    if dist_to_wp <= self.params.base_nav_tolerance:
                        self._maze_index += 1
                        if self._maze_index < len(self._maze_waypoints):
                            current_wp = self._maze_waypoints[self._maze_index]
                        else:
                            # Reached last waypoint, check if we're at start
                            if dist_to_start <= self.params.base_nav_tolerance:
                                self._print_state_change(TaskState.RESET, "reached start via waypoints, resetting arm")
                                qdot_zero = np.zeros(robot.n)
                                return qdot_zero, None, self.state
                            else:
                                # Last waypoint reached but not at start, continue directly
                                v, omega = self._compute_base_control(base_state, start_vec)
                                qdot = np.zeros(robot.n)
                                qdot[0] = omega
                                qdot[1] = v

                                if hasattr(robot, "qr"):
                                    ready_config = robot.qr
                                else:
                                    ready_config = np.zeros(robot.n)

                                e_q = ready_config - q
                                arm_qdot = self.params.arm_gain * e_q
                                arm_qdot = np.clip(arm_qdot, -self.params.q_dot_limit, self.params.q_dot_limit)
                                qdot[2:] = arm_qdot[2:]

                                return qdot, (v, omega), self.state

                    # Follow current waypoint
                    v, omega = self._compute_maze_base_control(base_state, current_wp)
                else:
                    # No waypoints available, navigate directly to start
                    v, omega = self._compute_base_control(base_state, start_vec)

                qdot = np.zeros(robot.n)
                qdot[0] = omega
                qdot[1] = v

                if hasattr(robot, "qr"):
                    ready_config = robot.qr
                else:
                    ready_config = np.zeros(robot.n)

                e_q = ready_config - q
                arm_qdot = self.params.arm_gain * e_q
                arm_qdot = np.clip(arm_qdot, -self.params.q_dot_limit, self.params.q_dot_limit)
                qdot[2:] = arm_qdot[2:]

                return qdot, (v, omega), self.state

        if self.state == TaskState.RESET:
            if hasattr(robot, "qr"):
                ready_config = robot.qr
            else:
                ready_config = np.zeros(robot.n)

            e_q = ready_config - q
            qdot = self.params.arm_gain * e_q
            qdot = np.clip(qdot, -self.params.q_dot_limit, self.params.q_dot_limit)

            qdot[0] = 0.0
            qdot[1] = 0.0

            if float(np.linalg.norm(e_q)) < 0.1:
                self._print_state_change(TaskState.IDLE, "arm reset complete")

            return qdot, None, self.state

        qdot_default = np.zeros(robot.n)
        return qdot_default, None, self.state

    def is_task_complete(self) -> bool:
        return self.state == TaskState.IDLE and self._grasped is False
