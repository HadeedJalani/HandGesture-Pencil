"""
All on-screen UI: toolbar, cursors, HUD text.

The toolbar layout (button positions) is computed at runtime from the
actual captured frame width -- never assumed -- because webcams frequently
ignore a requested resolution and fall back to whatever they support
(640x480 is common even when 1280x720 was requested). Building the layout
from a fixed assumed width would push buttons like Clear off-screen and
make them unreachable on narrower cameras.

The toolbar's static artwork (background, swatches, icons, labels) is
rendered once per resolution into an off-screen buffer and blitted every
frame instead of being redrawn button-by-button, which meaningfully cuts
per-frame OpenCV draw calls. Only the small dynamic bits (selection
highlight, hover state) are drawn fresh each frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from config import (
    BUTTON_GAP,
    BUTTON_MARGIN,
    BUTTON_SIZE,
    ERASER_THICKNESS,
    PALETTE,
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
Box = Tuple[int, int, int, int]  # x1, y1, x2, y2

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Minimum sizes we'll shrink down to before we'd rather clip than vanish.
_MIN_BUTTON_SIZE = 34
_MIN_GAP = 8
_MIN_MARGIN = 10

# Rough space reserved on the right for the app title so buttons never
# collide with it on narrow frames.
_TITLE_RESERVED_PX = 170


def _rounded_rect(
    img: np.ndarray, pt1: Point, pt2: Point, color: Color, radius: int = 10, thickness: int = -1
) -> None:
    """Draw a filled or outlined rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

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


@dataclass(frozen=True)
class _LaidOutColorButton:
    name: str
    color: Color
    box: Box


class ToolbarRenderer:
    """
    Builds a toolbar layout sized to the real frame width, pre-renders its
    static artwork once, then blits + overlays dynamic state each frame.
    """

    def __init__(self, frame_width: int) -> None:
        self._width = frame_width
        self._color_buttons: List[_LaidOutColorButton] = []
        self._eraser_box: Box = (0, 0, 0, 0)
        self._clear_box: Box = (0, 0, 0, 0)
        self._layout(frame_width)
        self._base = self._build_base()

    def _layout(self, frame_width: int) -> None:
        """Compute button boxes that always fit within `frame_width`."""
        n_colors = len(PALETTE)

        def total_width(button_size: int, gap: int, margin: int) -> int:
            colors_w = n_colors * button_size + (n_colors - 1) * gap
            eraser_w = gap * 2 + button_size
            clear_w = gap * 2 + button_size * 2
            return margin + colors_w + eraser_w + clear_w + margin

        button_size, gap, margin = BUTTON_SIZE, BUTTON_GAP, BUTTON_MARGIN
        available = max(frame_width - _TITLE_RESERVED_PX, 1)

        # Shrink proportionally until everything fits, down to a sane floor.
        needed = total_width(button_size, gap, margin)
        if needed > available:
            scale = max(available / needed, 0.4)
            button_size = max(_MIN_BUTTON_SIZE, int(button_size * scale))
            gap = max(_MIN_GAP, int(gap * scale))
            margin = max(_MIN_MARGIN, int(margin * scale))

        y1 = (TOOLBAR_HEIGHT - button_size) // 2
        y2 = y1 + button_size

        x = margin
        buttons = []
        for name, color in PALETTE:
            buttons.append(_LaidOutColorButton(name, color, (x, y1, x + button_size, y2)))
            x += button_size + gap
        self._color_buttons = buttons

        ex1 = x + gap
        self._eraser_box = (ex1, y1, ex1 + button_size, y2)

        cx1 = self._eraser_box[2] + gap
        self._clear_box = (cx1, y1, min(cx1 + button_size * 2, frame_width - margin), y2)

        self._button_size = button_size

    def _build_base(self) -> np.ndarray:
        base = np.full((TOOLBAR_HEIGHT, self._width, 3), UI_BG, dtype=np.uint8)

        cv2.line(base, (0, TOOLBAR_HEIGHT - 1), (self._width, TOOLBAR_HEIGHT - 1), UI_BG_LIGHT, 2)

        for button in self._color_buttons:
            x1, y1, x2, y2 = button.box
            _rounded_rect(base, (x1, y1), (x2, y2), button.color, radius=12)

        ex1, ey1, ex2, ey2 = self._eraser_box
        _rounded_rect(base, (ex1, ey1), (ex2, ey2), UI_BG_LIGHT, radius=12)
        cv2.circle(base, ((ex1 + ex2) // 2, (ey1 + ey2) // 2), max(6, (ex2 - ex1) // 4), WHITE, 2)

        cx1, cy1, cx2, cy2 = self._clear_box
        _rounded_rect(base, (cx1, cy1), (cx2, cy2), UI_BG_LIGHT, radius=12)
        font_scale = 0.55 if (cx2 - cx1) > 90 else 0.4
        cv2.putText(
            base, "CLR" if (cx2 - cx1) <= 90 else "CLEAR",
            (cx1 + 10, (cy1 + cy2) // 2 + 6), _FONT, font_scale, UI_TEXT, 2, cv2.LINE_AA,
        )

        title = "HANDGESTURE AIR PENCIL"
        (tw, _), _ = cv2.getTextSize(title, _FONT, 0.7, 2)
        title_x = max(self._width - tw - 24, cx2 + 16)
        if title_x + tw <= self._width:
            cv2.putText(
                base, title, (title_x, TOOLBAR_HEIGHT // 2 + 8),
                _FONT, 0.7, UI_ACCENT, 2, cv2.LINE_AA,
            )

        return base

    def render(self, frame: np.ndarray, current_color: Color, active_gesture: Gesture) -> None:
        """Blit the pre-rendered toolbar and overlay dynamic selection state."""
        frame[0:TOOLBAR_HEIGHT, 0 : self._width] = self._base

        for button in self._color_buttons:
            if button.color == current_color:
                x1, y1, x2, y2 = button.box
                _rounded_rect(
                    frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), WHITE, radius=14, thickness=2
                )

        if active_gesture == Gesture.ERASE:
            ex1, ey1, ex2, ey2 = self._eraser_box
            _rounded_rect(frame, (ex1 - 4, ey1 - 4), (ex2 + 4, ey2 + 4), UI_ACCENT, radius=14, thickness=2)

    # --- Hit-testing against the layout actually on screen --------------- #

    def select_color(self, point: Point, current_color: Color) -> Color:
        """Return a new color if `point` lands inside a swatch, else unchanged."""
        x, y = point
        for button in self._color_buttons:
            x1, y1, x2, y2 = button.box
            if x1 <= x <= x2 and y1 <= y <= y2:
                return button.color
        return current_color

    def is_over_clear_button(self, point: Point) -> bool:
        x, y = point
        x1, y1, x2, y2 = self._clear_box
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
            frame, "Press H for help", (max(frame.shape[1] - 210, 10), frame.shape[0] - 16),
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
        "Made by           -> Hadeed Jalani"
    ]
    box_w = min(430, frame.shape[1] - 32)
    box_h = 24 * len(lines) + 24
    x1, y1 = frame.shape[1] - box_w - 16, frame.shape[0] - box_h - 16
    overlay = frame.copy()
    _rounded_rect(overlay, (x1, y1), (x1 + box_w, y1 + box_h), UI_BG, radius=12)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, dst=frame)

    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (x1 + 18, y1 + 30 + i * 24), _FONT, 0.5, UI_TEXT, 1, cv2.LINE_AA
        )
