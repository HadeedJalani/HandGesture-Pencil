"""
All on-screen UI: toolbar, cursors, HUD text.

The toolbar's static artwork (background, swatches, icons, labels) is
rendered once into an off-screen buffer and blitted every frame instead of
being redrawn button-by-button, which meaningfully cuts per-frame OpenCV
draw calls. Only the small dynamic bits (selection highlight, hover state)
are drawn fresh each frame.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    CLEAR_BUTTON_BOX,
    COLOR_BUTTONS,
    ERASER_BUTTON_BOX,
    ERASER_THICKNESS,
    TOOLBAR_HEIGHT,
    UI_ACCENT,
    UI_BG,
    UI_BG_LIGHT,
    UI_MUTED,
    UI_TEXT,
    WHITE,
)
from gesture import Gesture

Point = Tuple[int, int]
Color = Tuple[int, int, int]

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _rounded_rect(
    img: np.ndarray, pt1: Point, pt2: Point, color: Color, radius: int = 10, thickness: int = -1
) -> None:
    """Draw a filled or outlined rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                       (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
            cv2.circle(img, (cx, cy), radius, color, -1)
    else:
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)


class ToolbarRenderer:
    """Pre-renders the toolbar chrome once, then overlays cheap dynamic bits."""

    def __init__(self, frame_width: int) -> None:
        self._width = frame_width
        self._base = self._build_base()

    def _build_base(self) -> np.ndarray:
        base = np.full((TOOLBAR_HEIGHT, self._width, 3), UI_BG, dtype=np.uint8)

        # Subtle bottom border
        cv2.line(base, (0, TOOLBAR_HEIGHT - 1), (self._width, TOOLBAR_HEIGHT - 1), UI_BG_LIGHT, 2)

        # Color swatches
        for button in COLOR_BUTTONS:
            x1, y1, x2, y2 = button.box
            _rounded_rect(base, (x1, y1), (x2, y2), button.color, radius=12)

        # Eraser button
        ex1, ey1, ex2, ey2 = ERASER_BUTTON_BOX
        _rounded_rect(base, (ex1, ey1), (ex2, ey2), UI_BG_LIGHT, radius=12)
        cv2.circle(base, ((ex1 + ex2) // 2, (ey1 + ey2) // 2), 14, WHITE, 2)
        cv2.putText(base, "ERS", (ex1 + 10, ey2 + 18), _FONT, 0.4, UI_MUTED, 1, cv2.LINE_AA)

        # Clear button
        cx1, cy1, cx2, cy2 = CLEAR_BUTTON_BOX
        _rounded_rect(base, (cx1, cy1), (cx2, cy2), UI_BG_LIGHT, radius=12)
        cv2.putText(base, "CLEAR", (cx1 + 14, (cy1 + cy2) // 2 + 6), _FONT, 0.55, UI_TEXT, 2, cv2.LINE_AA)

        # App title, right aligned
        title = "AI AIR PENCIL"
        (tw, _), _ = cv2.getTextSize(title, _FONT, 0.7, 2)
        cv2.putText(
            base, title, (self._width - tw - 24, TOOLBAR_HEIGHT // 2 + 8),
            _FONT, 0.7, UI_ACCENT, 2, cv2.LINE_AA,
        )

        return base

    def render(self, frame: np.ndarray, current_color: Color, active_gesture: Gesture) -> None:
        """Blit the pre-rendered toolbar and overlay dynamic selection state."""
        frame[0:TOOLBAR_HEIGHT, 0 : self._width] = self._base

        for button in COLOR_BUTTONS:
            if button.color == current_color:
                x1, y1, x2, y2 = button.box
                _rounded_rect(
                    frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), WHITE, radius=14, thickness=2
                )

        if active_gesture == Gesture.ERASE:
            ex1, ey1, ex2, ey2 = ERASER_BUTTON_BOX
            _rounded_rect(frame, (ex1 - 4, ey1 - 4), (ex2 + 4, ey2 + 4), UI_ACCENT, radius=14, thickness=2)


def select_color(point: Point, current_color: Color) -> Color:
    """Return a new color if `point` lands inside a swatch, else unchanged."""
    x, y = point
    for button in COLOR_BUTTONS:
        x1, y1, x2, y2 = button.box
        if x1 <= x <= x2 and y1 <= y <= y2:
            return button.color
    return current_color


def is_over_clear_button(point: Point) -> bool:
    x, y = point
    x1, y1, x2, y2 = CLEAR_BUTTON_BOX
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_hud(frame: np.ndarray, gesture: Gesture, brush_thickness: int, fps: float) -> None:
    """Bottom-left status readout: mode, brush size, FPS."""
    h = frame.shape[0]
    lines = [
        f"Mode: {gesture}",
        f"Brush: {brush_thickness}px",
        f"FPS: {fps:.0f}",
    ]
    y = h - 16
    for line in reversed(lines):
        cv2.putText(frame, line, (16, y), _FONT, 0.55, UI_TEXT, 1, cv2.LINE_AA)
        y -= 24


def draw_cursor(frame: np.ndarray, point: Point, color: Color) -> None:
    cv2.circle(frame, point, 9, color, -1, cv2.LINE_AA)
    cv2.circle(frame, point, 9, WHITE, 1, cv2.LINE_AA)


def draw_eraser_cursor(frame: np.ndarray, point: Point) -> None:
    cv2.circle(frame, point, ERASER_THICKNESS // 2, WHITE, 2, cv2.LINE_AA)


def draw_help_overlay(frame: np.ndarray, visible: bool) -> None:
    """Optional keyboard shortcut cheat-sheet, toggled with 'h'."""
    if not visible:
        cv2.putText(
            frame, "Press H for help", (frame.shape[1] - 210, frame.shape[0] - 16),
            _FONT, 0.5, UI_MUTED, 1, cv2.LINE_AA,
        )
        return

    lines = [
        "Index finger      -> Draw",
        "Index + Middle    -> Select color / tool",
        "Open palm         -> Erase",
        "Fist              -> Idle",
        "[ / ]             -> Shrink / grow brush",
        "C                 -> Clear canvas",
        "S                 -> Toggle hand skeleton",
        "H                 -> Toggle this help",
        "Q / Esc           -> Quit",
    ]
    box_w, box_h = 430, 24 * len(lines) + 24
    x1, y1 = frame.shape[1] - box_w - 16, frame.shape[0] - box_h - 16
    overlay = frame.copy()
    _rounded_rect(overlay, (x1, y1), (x1 + box_w, y1 + box_h), UI_BG, radius=12)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, dst=frame)

    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (x1 + 18, y1 + 30 + i * 24), _FONT, 0.5, UI_TEXT, 1, cv2.LINE_AA
        )
