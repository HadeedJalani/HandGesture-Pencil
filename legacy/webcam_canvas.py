import cv2
import numpy as np

cap = cv2.VideoCapture(0)

canvas = np.zeros((480,640,3),dtype=np.uint8)

# Draw ONLY ONCE
cv2.line(canvas,(100,100),(300,300),(0,255,0),3)
cv2.circle(canvas,(450,200),50,(0,0,255),-1)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame,(640,480))

    output = cv2.add(frame,canvas)

    cv2.imshow("Canvas + Webcam",output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        print("Canvas Cleared")
        canvas = np.zeros((480,640,3),dtype=np.uint8)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
