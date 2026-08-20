"""
Central configuration for the HandGesture Air Pencil application.

Every tunable constant lives here so behaviour can be adjusted without
touching application logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Color = Tuple[int, int, int]  # OpenCV uses BGR, not RGB

# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720

# --------------------------------------------------------------------------- #
# MediaPipe Hands
# --------------------------------------------------------------------------- #
MAX_HANDS: int = 1
MIN_DETECTION_CONFIDENCE: float = 0.7
MIN_TRACKING_CONFIDENCE: float = 0.7
DRAW_HAND_SKELETON: bool = True

# --------------------------------------------------------------------------- #
# Palette (BGR)
# --------------------------------------------------------------------------- #
GREEN: Color = (0, 200, 83)
BLUE: Color = (255, 121, 24)
RED: Color = (60, 20, 220)
YELLOW: Color = (0, 210, 255)
PURPLE: Color = (200, 60, 160)

WHITE: Color = (255, 255, 255)
BLACK: Color = (0, 0, 0)

# UI accent colors
UI_BG: Color = (32, 32, 34)
UI_BG_LIGHT: Color = (52, 52, 56)
UI_ACCENT: Color = (255, 191, 0)
UI_TEXT: Color = (235, 235, 235)
UI_MUTED: Color = (150, 150, 155)

CURRENT_COLOR: Color = GREEN

# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
BRUSH_THICKNESS: int = 6
ERASER_THICKNESS: int = 70
MIN_BRUSH_THICKNESS: int = 2
MAX_BRUSH_THICKNESS: int = 40

# --------------------------------------------------------------------------- #
# Signal Processing
# --------------------------------------------------------------------------- #
SMOOTHING_ALPHA: float = 0.4        # Lower = smoother but laggier cursor
GESTURE_FRAMES_REQUIRED: int = 4    # Frames a gesture must persist to register
MAX_JUMP_DISTANCE: int = 80         # Rejects tracking glitches / teleports

# --------------------------------------------------------------------------- #
# Toolbar / Layout
# --------------------------------------------------------------------------- #
TOOLBAR_HEIGHT: int = 90
BUTTON_SIZE: int = 60
BUTTON_GAP: int = 16
BUTTON_MARGIN: int = 20


PALETTE: Tuple[Tuple[str, Color], ...] = (
    ("Green", GREEN),
    ("Blue", BLUE),
    ("Red", RED),
    ("Yellow", YELLOW),
    ("Purple", PURPLE),
)


@dataclass(frozen=True)
class ColorButton:
    name: str
    color: Color
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2


# NOTE: Button positions are NOT computed here. Webcams frequently ignore a
# requested capture resolution and fall back to whatever they actually
# support (640x480 is common), so any layout baked in at import time using
# a fixed assumed width can end up partly or fully off-screen. Instead,
# `ui.ToolbarRenderer` builds the button layout at runtime from the frame's
# real width, shrinking button size/spacing if needed so every button
# (including Clear) always stays on screen and reachable.

# --------------------------------------------------------------------------- #
# Gestures
# --------------------------------------------------------------------------- #
FINGER_PATTERN_DRAW: Tuple[int, ...] = (0, 1, 0, 0, 0)    # Index only
FINGER_PATTERN_SELECT: Tuple[int, ...] = (0, 1, 1, 0, 0)  # Index + Middle
FINGER_PATTERN_ERASE: Tuple[int, ...] = (1, 1, 1, 1, 1)   # Open palm
FINGER_PATTERN_IDLE: Tuple[int, ...] = (0, 0, 0, 0, 0)    # Fist
