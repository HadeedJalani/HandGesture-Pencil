"""
Thin wrapper around MediaPipe Hands.

Encapsulates model setup, per-frame inference, and landmark/handedness
extraction so the rest of the app never touches the MediaPipe API directly.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from config import (
    DRAW_HAND_SKELETON,
    MAX_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)

Point = Tuple[int, int]


class HandTracker:
    """Detects a single hand per frame and returns pixel-space landmarks."""

    def __init__(
        self,
        max_hands: int = MAX_HANDS,
        min_detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
        draw_skeleton: bool = DRAW_HAND_SKELETON,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._mp_draw = mp.solutions.drawing_utils
        self._draw_skeleton = draw_skeleton

        self._hands = self._mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame: np.ndarray) -> Tuple[Optional[List[Point]], Optional[str]]:
        """
        Run detection on a BGR frame (mutated in-place with an optional
        skeleton overlay) and return (landmarks, hand_type), or (None, None)
        if no hand is present.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # MediaPipe expects a read-only buffer for best performance.
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            return None, None

        hand = results.multi_hand_landmarks[0]

        if self._draw_skeleton:
            self._mp_draw.draw_landmarks(
                frame, hand, self._mp_hands.HAND_CONNECTIONS
            )

        h, w = frame.shape[:2]
        landmarks: List[Point] = [
            (int(lm.x * w), int(lm.y * h)) for lm in hand.landmark
        ]

        hand_type = results.multi_handedness[0].classification[0].label

        return landmarks, hand_type

    def toggle_skeleton(self) -> bool:
        """Flip skeleton overlay on/off; returns the new state."""
        self._draw_skeleton = not self._draw_skeleton
        return self._draw_skeleton

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
