import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Fingertip landmark IDs
tip_ids = [4, 8, 12, 16, 20]

# Start Webcam
cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror the webcam
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(rgb)

    mode = "No Hand"

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        # Get Left / Right hand
        hand_type = results.multi_handedness[0].classification[0].label

        # Draw hand skeleton
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        landmarks = []

        h, w, c = frame.shape

        # Store landmark coordinates
        for lm in hand.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)

            landmarks.append((x, y))

        fingers = []


        # Thumb


        if hand_type == "Right":

            if landmarks[4][0] < landmarks[3][0]:
                fingers.append(1)
            else:
                fingers.append(0)

        else:  # Left

            if landmarks[4][0] > landmarks[3][0]:
                fingers.append(1)
            else:
                fingers.append(0)


        # Other Four Fingers


        for tip in tip_ids[1:]:

            if landmarks[tip][1] < landmarks[tip - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Print hand + fingers
        print(hand_type, fingers)


        # Gesture Recognition
    

        if fingers == [0, 1, 0, 0, 0]:

            mode = "Drawing Mode"

        elif fingers == [0, 1, 1, 0, 0]:

            mode = "Selection Mode"

        elif fingers == [1, 1, 1, 1, 1]:

            mode = "Clear Mode"

        else:

            mode = "Unknown Gesture"

        # Display Left / Right hand
        cv2.putText(
            frame,
            hand_type + " Hand",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    # Display Mode
    cv2.putText(
        frame,
        mode,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()

#------run this in terminal to create virtual environment and activate it(in terminal)
#  py -3.12 -m venv venv
#  .\venv\scripts\activate.ps1
