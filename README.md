<div align="center">

# ✍️ HandGesture Air Pencil

**Draw in thin air. Your webcam and one finger are the only pen you need.**

A real-time, gesture-controlled drawing application built with **OpenCV** and **MediaPipe Hands** track a bare hand, recognize drawing gestures, and paint directly onto the camera feed with zero physical hardware.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-00A6A6?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

HandGestureAir Pencil turns your webcam into a touchless whiteboard. MediaPipe locates 21 landmarks on your hand every frame; a small finger-state engine turns those landmarks into one of four gestures - **draw**, **select**, **erase**, **idle** - which are debounced over several frames to stay stable even when tracking gets noisy. Strokes are rendered onto a persistent canvas and composited live over the camera feed.

## Features

- 🎯 **Real-time hand tracking** - single-hand, 21-point landmark detection via MediaPipe
- ✌️ **Four core gestures** - index finger to draw, index+middle to select, open palm to erase, fist to idle
- 🪄 **Gesture stabilization** - a frame-count debounce filter eliminates single-frame flicker between gestures
- 📉 **Jitter-free strokes** - exponential moving-average smoothing on the fingertip cursor, plus jump rejection for tracking glitches
- 🎨 **In-app toolbar** - 5-color palette, dedicated eraser and clear buttons, all selectable by gesture
- ⌨️ **Keyboard shortcuts** - resize the brush, clear the canvas, toggle the skeleton overlay or the help panel without leaving draw mode
- 📊 **Live HUD** - current mode, brush size, and FPS rendered on screen
- ⚡ **Efficient rendering** - the toolbar's static artwork is pre-rendered once and blitted per frame instead of being redrawn button-by-button, cutting per-frame OpenCV draw calls significantly

## Gesture Reference

| Gesture | Hand Pose | Action |
|---|---|---|
| **Draw** | ☝️ Index finger only | Paints a stroke in the current color |
| **Select** | ✌️ Index + middle finger | Pick a color, or tap **Clear** on the toolbar |
| **Erase** | 🖐️ Open palm | Erases under the cursor with a soft circular eraser |
| **Idle** | ✊ Closed fist | No action lift the pen |

## Keyboard Shortcuts

| Key | Effect |
|---|---|
| `[` / `]` | Shrink / grow brush size |
| `C` | Clear the canvas |
| `S` | Toggle the hand-skeleton overlay |
| `H` | Toggle the on-screen help panel |
| `Q` / `Esc` | Quit |

## Installation

**Requirements:** Python 3.9–3.12 and a working webcam.

```bash
# 1. Clone the repository
git clone https://github.com/HadeedJalani/HandGesture-Pencil.git
cd HandGesture-Pencil

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Optional flags:

```bash
python main.py --camera 1          # use a different webcam
python main.py --width 1920 --height 1080
python main.py --no-skeleton       # start with the hand skeleton hidden
```

## Project Structure

```
HandGesture-Pencil/
├── main.py            # Application entry point / render loop
├── config.py           # All tunable constants (camera, colors, thresholds, layout)
├── hand_tracker.py     # MediaPipe Hands wrapper — landmark + handedness extraction
├── gesture.py           # Finger-state extraction, gesture classification & stabilization
├── painter.py           # Canvas state: strokes, erasing, brush size, compositing
├── ui.py                 # Toolbar rendering, cursors, HUD, help overlay
├── utils.py              # Coordinate smoothing (EMA), FPS counter, geometry helpers
├── requirements.txt
├── legacy/                # Early single-file prototypes, kept for reference
└── LICENSE
```

## How It Works

```
Webcam Frame
     │
     ▼
HandTracker.detect()  ──► 21 hand landmarks + handedness
     │
     ▼
get_finger_states()   ──► [thumb, index, middle, ring, pinky] as 0/1
     │
     ▼
detect_gesture()      ──► raw Gesture (DRAW / SELECT / ERASE / IDLE)
     │
     ▼
GestureStabilizer      ──► debounced, stable Gesture
     │
     ▼
CoordinateSmoother     ──► jitter-free fingertip position
     │
     ▼
Painter                 ──► stroke / erase applied to canvas
     │
     ▼
ToolbarRenderer + HUD  ──► composited output frame
     │
     ▼
   Display
```

## Roadmap

- [ ] Undo / redo stack
- [ ] Save canvas to PNG
- [ ] Two-hand support (e.g. pinch-to-zoom on the canvas)
- [ ] Shape tools (line, rectangle, circle) via gesture combos

## Contributing

Issues and pull requests are welcome. For larger changes, please open an issue first to discuss what you'd like to change.

# ✍️ Author
Hadeed Jalani

## License

Distributed under the [MIT License](LICENSE).
