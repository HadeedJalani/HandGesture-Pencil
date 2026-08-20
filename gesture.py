"""
Finger-state extraction and gesture classification/stabilization.

A "gesture" is derived from which of the 5 fingers are extended. Because
raw per-frame detection is noisy, `GestureStabilizer` requires a gesture to
persist for `GESTURE_FRAMES_REQUIRED` consecutive frames before it is
considered active, which eliminates single-frame flicker.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Sequence, Tuple

from config import (
    FINGER_PATTERN_DRAW,
    FINGER_PATTERN_ERASE,
    FINGER_PATTERN_IDLE,
    FINGER_PATTERN_SELECT,
    GESTURE_FRAMES_REQUIRED,
)

Point = Tuple[int, int]

# Fingertip landmark indices in MediaPipe's 21-point hand model:
# [thumb, index, middle, ring, pinky]
_TIP_IDS: Tuple[int, int, int, int, int] = (4, 8, 12, 16, 20)


class Gesture(str, Enum):
    """Recognized drawing-mode gestures."""

    DRAW = "DRAW"
    SELECT = "SELECT"
    ERASE = "ERASE"
    IDLE = "IDLE"

    def __str__(self) -> str:  # nicer default string form for overlays
        return self.value


_PATTERN_TO_GESTURE = {
    FINGER_PATTERN_DRAW: Gesture.DRAW,
    FINGER_PATTERN_SELECT: Gesture.SELECT,
    FINGER_PATTERN_ERASE: Gesture.ERASE,
    FINGER_PATTERN_IDLE: Gesture.IDLE,
}


def get_finger_states(landmarks: Sequence[Point], hand_type: str) -> List[int]:
    """
    Return a 5-element list of 0/1 flags, one per finger (thumb -> pinky),
    indicating whether that finger is extended.
    """
    fingers: List[int] = []

    # Thumb: compare x-position against the joint below it, mirrored by hand.
    thumb_tip_x = landmarks[_TIP_IDS[0]][0]
    thumb_joint_x = landmarks[_TIP_IDS[0] - 1][0]
    if hand_type == "Right":
        fingers.append(1 if thumb_tip_x < thumb_joint_x else 0)
    else:
        fingers.append(1 if thumb_tip_x > thumb_joint_x else 0)

    # Remaining four fingers: tip above its own lower knuckle (in image
    # coordinates, "above" means a smaller y value) means extended.
    for tip in _TIP_IDS[1:]:
        fingers.append(1 if landmarks[tip][1] < landmarks[tip - 2][1] else 0)

    return fingers


def detect_gesture(fingers: Sequence[int]) -> Gesture:
    """Map a finger-state pattern to a Gesture, defaulting to IDLE."""
    return _PATTERN_TO_GESTURE.get(tuple(fingers), Gesture.IDLE)


class GestureStabilizer:
    """
    Debounces gesture flicker by requiring a candidate gesture to repeat
    for several consecutive frames before it becomes the "stable" gesture.
    """

    def __init__(self, frames_required: int = GESTURE_FRAMES_REQUIRED) -> None:
        self._frames_required = frames_required
        self._candidate: Gesture = Gesture.IDLE
        self._streak: int = 0
        self._stable: Gesture = Gesture.IDLE

    def update(self, gesture: Gesture) -> Gesture:
        if gesture == self._candidate:
            self._streak += 1
        else:
            self._candidate = gesture
            self._streak = 1

        if self._streak >= self._frames_required:
            self._stable = gesture

        return self._stable

    @property
    def stable_gesture(self) -> Gesture:
        return self._stable
