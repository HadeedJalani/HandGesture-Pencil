"""
Owns the drawing canvas: strokes, erasing, brush size, and compositing the
canvas back over the live camera feed.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    BRUSH_THICKNESS,
    ERASER_THICKNESS,
    MAX_BRUSH_THICKNESS,
    MAX_JUMP_DISTANCE,
    MIN_BRUSH_THICKNESS,
)
from utils import point_distance

Point = Tuple[int, int]
Color = Tuple[int, int, int]


class Painter:
    """Maintains an off-screen canvas and merges it onto camera frames."""

    def __init__(
        self,
        brush_thickness: int = BRUSH_THICKNESS,
        eraser_thickness: int = ERASER_THICKNESS,
    ) -> None:
        self.canvas: Optional[np.ndarray] = None
        self.previous_point: Optional[Point] = None
        self.brush_thickness = brush_thickness
        self.eraser_thickness = eraser_thickness

    def create_canvas(self, frame: np.ndarray) -> None:
        """Lazily allocate a canvas matching the frame size (once)."""
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)

    def draw(self, current_point: Point, color: Color) -> None:
        if self.previous_point is None:
            self.previous_point = current_point
            return

        distance = point_distance(self.previous_point, current_point)

        # Reject sudden teleports caused by tracking glitches instead of
        # drawing a stray line across the canvas.
        if distance <= MAX_JUMP_DISTANCE:
            cv2.line(
                self.canvas,
                self.previous_point,
                current_point,
                color,
                self.brush_thickness,
                lineType=cv2.LINE_AA,
            )

        self.previous_point = current_point

    def erase(self, point: Point) -> None:
        cv2.circle(
            self.canvas,
            point,
            self.eraser_thickness // 2,
            (0, 0, 0),
            -1,
        )

    def reset_previous(self) -> None:
        self.previous_point = None

    def clear(self) -> None:
        if self.canvas is not None:
            self.canvas[:] = 0
        self.previous_point = None

    def grow_brush(self, delta: int) -> None:
        self.brush_thickness = int(
            np.clip(self.brush_thickness + delta, MIN_BRUSH_THICKNESS, MAX_BRUSH_THICKNESS)
        )

    def create_mask(self) -> np.ndarray:
        gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, drawing_mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
        return drawing_mask

    def combine(self, frame: np.ndarray) -> np.ndarray:
        """Composite the canvas over the frame, canvas pixels take priority."""
        drawing_mask = self.create_mask()
        inverse_mask = cv2.bitwise_not(drawing_mask)

        frame_background = cv2.bitwise_and(frame, frame, mask=inverse_mask)
        canvas_foreground = cv2.bitwise_and(self.canvas, self.canvas, mask=drawing_mask)

        return cv2.add(frame_background, canvas_foreground)
