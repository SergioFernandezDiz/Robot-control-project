"""
FrankieAgent class for robot simulation.
Handles robot model and Swift visualization integration.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple
import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
import swift


class FrankieAgent:
    """Bundle Frankie robot model and Swift visualization."""

    def __init__(self, name: str, start_config: Sequence[float], start_base_pose: Tuple[float, float, float], 
                 fix_geometry: bool = False) -> None:
        """
        Initialize FrankieAgent.
        
        Args:
            name: Name identifier for the agent
            start_config: Initial joint configuration
            start_base_pose: Initial base pose [x, y, theta]
            fix_geometry: If True, fixes geometry attributes for Swift compatibility (used in maze simulation)
        """
        self.name = name
        # Use URDF Frankie model
        self._robot = rtb.models.URDF.Frankie()
        
        # Fix geometry attributes if needed (for maze simulation compatibility)
        if fix_geometry:
            for link in self._robot.links:
                if hasattr(link, "geometry") and link.geometry is None:
                    link.geometry = []
                if hasattr(link, "collision") and link.collision is None:
                    link.collision = []
                for attr in (
                    "geometryfilename",
                    "collisionfilename",
                    "_geometryfilename",
                    "_collisionfilename",
                ):
                    if hasattr(link, attr):
                        setattr(link, attr, [])
        
        self._robot.q = np.array(start_config, dtype=float)
        
        # Set base pose (mobile base position)
        self._base_pose = np.array(start_base_pose, dtype=float)  
        self._update_base_transform()

    def _update_base_transform(self) -> None:
        """Update robot base transform based on mobile base pose."""

        x, y, theta = self._base_pose

        # Set base transform from 2D pose to 3D matrix with trasnsiction and rotation 
        self._robot.base = sm.SE3(x, y, 0.0) * sm.SE3.Rz(theta)

    def register(self, env: swift.Swift) -> None:
        """Register the robot with the Swift environment."""

        env.add(self._robot)

    def q(self) -> np.ndarray:
        """Get current joint configuration."""

        return np.asarray(self._robot.q, dtype=float)

    def base_state(self) -> np.ndarray:
        """Get mobile base state [x, y, theta]."""
        
        return self._base_pose.copy()

    def apply_velocity_cmd(self, qdot: np.ndarray, base_cmd: Optional[Tuple[float, float]] = None, dt: float = 0.05) -> None:
        """
        Apply joint velocities and update mobile base if needed.
        
        Args:
            qdot: Joint velocity vector
            base_cmd: Optional (v, omega) command for base
            dt: Time step (default 0.05)
        """
        if base_cmd is not None:
            # Use unicycle dynamics for base pose tracking
            v, omega = base_cmd
            self._base_pose[0] += v * np.cos(self._base_pose[2]) * dt
            self._base_pose[1] += v * np.sin(self._base_pose[2]) * dt
            self._base_pose[2] += omega * dt
            # Wrap angle
            self._base_pose[2] = (self._base_pose[2] + np.pi) % (2 * np.pi) - np.pi
            self._update_base_transform()
            # Don't apply base joint velocities - base is controlled via transform
            # Only apply arm joint velocities
            qdot_arm = qdot.copy()
            qdot_arm[0] = 0.0  # Don't move base rotation joint
            qdot_arm[1] = 0.0  # Don't move base translation joint
            self._robot.qd = qdot_arm
        else:
            # No base command - update base from joint velocities if they're non-zero
            omega = qdot[0] if len(qdot) > 0 else 0.0
            v = qdot[1] if len(qdot) > 1 else 0.0
            if abs(omega) > 1e-6 or abs(v) > 1e-6:
                # Convert base joint velocities to unicycle motion
                # Joint 1 translates along base x-axis, need to project to world
                self._base_pose[0] += v * np.cos(self._base_pose[2]) * dt
                self._base_pose[1] += v * np.sin(self._base_pose[2]) * dt
                self._base_pose[2] += omega * dt
                self._base_pose[2] = (self._base_pose[2] + np.pi) % (2 * np.pi) - np.pi
                self._update_base_transform()
            # Apply all joint velocities
            self._robot.qd = qdot

