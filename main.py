"""
HandGesture Air Pencil — draw in the air, tracked by your webcam.

Entry point. Wires together hand tracking, gesture recognition, the
canvas/painter, and the UI into a single render loop.

Run:
    python main.py
"""

from __future__ import annotations

import argparse
import logging
import sys

import cv2

from config import CAMERA_INDEX, CURRENT_COLOR, FRAME_HEIGHT, FRAME_WIDTH, TOOLBAR_HEIGHT
from gesture import Gesture, GestureStabilizer, detect_gesture, get_finger_states
from hand_tracker import HandTracker
from painter import Painter
from ui import (
    ToolbarRenderer,
    draw_cursor,
    draw_eraser_cursor,
    draw_help_overlay,
    draw_hud,
)
from utils import CoordinateSmoother, FPSCounter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("HandGesture_air_pencil")

WINDOW_NAME = "HandGesture Air Pencil"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw in the air using hand gestures.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Webcam device index.")
    parser.add_argument("--width", type=int, default=FRAME_WIDTH, help="Capture width.")
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT, help="Capture height.")
    parser.add_argument(
        "--no-skeleton", action="store_true", help="Start with hand skeleton overlay disabled."
    )
    return parser.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logger.error("Could not open webcam at index %d.", index)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def run() -> None:
    args = parse_args()
    cap = open_camera(args.camera, args.width, args.height)

    toolbar: ToolbarRenderer | None = None
    hand_tracker = HandTracker(draw_skeleton=not args.no_skeleton)
    gesture_stabilizer = GestureStabilizer()
    smoother = CoordinateSmoother()
    painter = Painter()
    fps_counter = FPSCounter()

    current_color = CURRENT_COLOR
    show_help = False

    logger.info("HandGesture Air Pencil started. Press H for controls, Q or Esc to quit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                logger.warning("Failed to read a frame from the camera; stopping.")
                break

            frame = cv2.flip(frame, 1)  # Mirror for a natural "in front of a mirror" feel

            if toolbar is None:
                toolbar = ToolbarRenderer(frame.shape[1])

            painter.create_canvas(frame)
            landmarks, hand_type = hand_tracker.detect(frame)

            active_gesture = gesture_stabilizer.stable_gesture

            if landmarks is not None:
                fingers = get_finger_states(landmarks, hand_type)
                raw_gesture = detect_gesture(fingers)
                active_gesture = gesture_stabilizer.update(raw_gesture)

                raw_x, raw_y = landmarks[8]  # Index fingertip
                point = smoother.smooth(raw_x, raw_y)

                if active_gesture == Gesture.DRAW:
                    draw_cursor(frame, point, current_color)
                    if point[1] > TOOLBAR_HEIGHT:
                        painter.draw(point, current_color)
                    else:
                        painter.reset_previous()

                elif active_gesture == Gesture.SELECT:
                    painter.reset_previous()
                    if toolbar.is_over_clear_button(point):
                        painter.clear()
                    else:
                        current_color = toolbar.select_color(point, current_color)

                elif active_gesture == Gesture.ERASE:
                    painter.reset_previous()
                    draw_eraser_cursor(frame, point)
                    if point[1] > TOOLBAR_HEIGHT:
                        painter.erase(point)

                else:  # IDLE
                    painter.reset_previous()
            else:
                painter.reset_previous()
                smoother.reset()

            output = painter.combine(frame)
            toolbar.render(output, current_color, active_gesture)
            draw_hud(output, active_gesture, painter.brush_thickness, fps_counter.tick())
            draw_help_overlay(output, show_help)

            cv2.imshow(WINDOW_NAME, output)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # Q or Esc
                break
            elif key == ord("c"):
                painter.clear()
            elif key == ord("h"):
                show_help = not show_help
            elif key == ord("s"):
                hand_tracker.toggle_skeleton()
            elif key == ord("["):
                painter.grow_brush(-2)
            elif key == ord("]"):
                painter.grow_brush(2)

    finally:
        cap.release()
        hand_tracker.close()
        cv2.destroyAllWindows()
        logger.info("Session ended.")


if __name__ == "__main__":
    run()
