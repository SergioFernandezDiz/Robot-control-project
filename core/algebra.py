from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math
import numpy as np


@dataclass(frozen=True)
class Vector3:
    x: float
    y: float
    z: float

    @staticmethod
    def from_iterable(values: Iterable[float]) -> Vector3:
        arr = np.asarray(list(values), dtype=float).reshape(3)
        return Vector3(float(arr[0]), float(arr[1]), float(arr[2]))

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def with_x(self, x: float) -> Vector3:
        return Vector3(x, self.y, self.z)

    def with_y(self, y: float) -> Vector3:
        return Vector3(self.x, y, self.z)

    def with_z(self, z: float) -> Vector3:
        return Vector3(self.x, self.y, z)

    def __add__(self, other: Vector3) -> Vector3:
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3) -> Vector3:
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vector3:
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vector3:
        return self.__mul__(scalar)

    def dot(self, other: Vector3) -> float:
        return float(self.x * other.x + self.y * other.y + self.z * other.z)

    def norm(self) -> float:
        return float(math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z))

    def normalized(self) -> Vector3:
        n = self.norm()
        if n == 0.0:
            return Vector3(0.0, 0.0, 0.0)
        inv = 1.0 / n
        return Vector3(self.x * inv, self.y * inv, self.z * inv)

    @staticmethod
    def distance(a: Vector3, b: Vector3) -> float:
        return (a - b).norm()

    @staticmethod
    def from_xy(x: float, y: float, z: float = 0.0) -> Vector3:
        return Vector3(float(x), float(y), float(z))

    @staticmethod
    def zero() -> Vector3:
        return Vector3(0.0, 0.0, 0.0)



#================= Helpers =================#

def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def damped_pseudoinverse(J: np.ndarray, pinv_damping: float) -> np.ndarray:
    m, _ = J.shape
    return J.T @ np.linalg.inv(J @ J.T + pinv_damping * np.eye(m))


def base_state_to_vector3(base_state: Sequence[float]) -> Vector3:
    x = float(base_state[0])
    y = float(base_state[1])
    return Vector3.from_xy(x, y, 0.0)

#============================================#
