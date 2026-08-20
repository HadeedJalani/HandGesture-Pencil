"""Small, dependency-free helpers shared across the app."""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

from config import SMOOTHING_ALPHA

Point = Tuple[int, int]


class CoordinateSmoother:
    """
    Exponential moving average (EMA) smoother for fingertip coordinates.

    Reduces jitter from per-frame landmark noise without the latency of a
    larger moving-average window.
    """

    def __init__(self, alpha: float = SMOOTHING_ALPHA) -> None:
        self._alpha = alpha
        self._x: Optional[float] = None
        self._y: Optional[float] = None

    def smooth(self, raw_x: int, raw_y: int) -> Point:
        if self._x is None or self._y is None:
            self._x, self._y = float(raw_x), float(raw_y)
        else:
            self._x = self._alpha * raw_x + (1 - self._alpha) * self._x
            self._y = self._alpha * raw_y + (1 - self._alpha) * self._y

        return int(self._x), int(self._y)

    def reset(self) -> None:
        self._x = None
        self._y = None


class FPSCounter:
    """Rolling FPS estimate using an EMA over frame deltas."""

    def __init__(self, smoothing: float = 0.9) -> None:
        self._smoothing = smoothing
        self._last_time = time.perf_counter()
        self._fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        delta = now - self._last_time
        self._last_time = now

        if delta > 0:
            instantaneous = 1.0 / delta
            self._fps = (
                self._fps * self._smoothing + instantaneous * (1 - self._smoothing)
                if self._fps
                else instantaneous
            )

        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


def point_distance(point1: Point, point2: Point) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])
