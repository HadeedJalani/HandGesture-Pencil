import cv2
import mediapipe as mp
import numpy as np


# 1. Initialize Camera

cap = cv2.VideoCapture(0)


# 2. Initialize MediaPipe Hands

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# Fingertip landmark IDs
tip_ids = [4, 8, 12, 16, 20]


# 3. Previous Point

previous_point = None


# 4. Canvas

canvas = None


# 5. Main Loop

while True:

    # Read frame
    success, frame = cap.read()

    if not success:
        print("Failed to read camera.")
        break


    # Flip frame
    frame = cv2.flip(frame, 1)


    
    # Create canvas

    if canvas is None:

        canvas = np.zeros_like(frame)


    # Convert BGR -> RGB

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame using MediaPipe
    

    results = hands.process(rgb)


    # Check if hand detected
    

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]


        # Draw hand skeleton
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )


        # Extract landmarks

        h, w, _ = frame.shape

        landmarks = []

        for lm in hand.landmark:

            x = int(lm.x * w)
            y = int(lm.y * h)

            landmarks.append((x, y))


        # Determine which fingers are up

        fingers = []


        # Get Left / Right hand
        hand_type = results.multi_handedness[0].classification[0].label


        # Thumb
        if hand_type == "Right":

            if landmarks[4][0] < landmarks[3][0]:
                fingers.append(1)
            else:
                fingers.append(0)

        else:

            if landmarks[4][0] > landmarks[3][0]:
                fingers.append(1)
            else:
                fingers.append(0)


        # Index, Middle, Ring, Little
        for tip in tip_ids[1:]:

            if landmarks[tip][1] < landmarks[tip - 2][1]:
                fingers.append(1)

            else:
                fingers.append(0)


        # Print Current Finger State

        print("Finger State:", fingers)


        # Get Landmark 8
        # Index fingertip

        current_point = landmarks[8]

        x, y = current_point


        # Draw circle on index fingertip
        cv2.circle(
            frame,
            (x, y),
            8,
            (0, 255, 0),
            -1
        )


        # Display Fingertip Coordinates

        cv2.putText(
            frame,
            f"Fingertip: ({x}, {y})",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


        # Drawing Mode
        # Only index finger UP

        if fingers == [0, 1, 0, 0, 0]:

            cv2.putText(
                frame,
                "Drawing Mode",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


            # First drawing point
            if previous_point is None:

                previous_point = current_point


            # Draw previous -> current
            else:

                cv2.line(
                    canvas,
                    previous_point,
                    current_point,
                    (0, 255, 0),
                    5
                )

                previous_point = current_point


        # Not Drawing

        else:

            previous_point = None

    # No Hand Detected

    else:

        previous_point = None


    # Combine Webcam + Canvas

    output = cv2.add(frame, canvas)


    # Display Output

    cv2.imshow("Air Pencil", output)


    # Keyboard

    key = cv2.waitKey(1) & 0xFF

    # Clear Canvas
    if key == ord('c'):

        canvas[:] = 0
        previous_point = None

    # Quit
    if key == ord(' '):
        break


# Release Resources

cap.release()
hands.close()
cv2.destroyAllWindows()
