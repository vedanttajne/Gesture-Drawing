import cv2
import mediapipe as mp
import numpy as np
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

canvas = None
prev_x, prev_y = 0, 0

draw_color = (255, 0, 0)
brush_size = 8
eraser_size = 45

colors = [
    ((255, 0, 0), "Blue", 20, 20, 120, 70),
    ((0, 255, 0), "Green", 140, 20, 240, 70),
    ((0, 0, 255), "Red", 260, 20, 360, 70),
    ((0, 255, 255), "Yellow", 380, 20, 500, 70),
    ((0, 0, 0), "Eraser", 520, 20, 650, 70),
]

def fingers_up(hand):
    fingers = []

    if hand.landmark[8].y < hand.landmark[6].y:
        fingers.append(1)
    else:
        fingers.append(0)

    if hand.landmark[12].y < hand.landmark[10].y:
        fingers.append(1)
    else:
        fingers.append(0)

    return fingers

while True:
    success, frame = cap.read()
    if not success:
        print("Camera not detected")
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    h, w, c = frame.shape

    # Draw color buttons
    for color, name, x1, y1, x2, y2 in colors:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, name, (x1 + 10, y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            fingers = fingers_up(hand_landmarks)

            # Selection mode: index + middle
            if fingers == [1, 1]:
                prev_x, prev_y = 0, 0
                cv2.circle(frame, (x, y), 12, (0, 255, 255), -1)

                # Select color if finger touches top buttons
                if y < 80:
                    for color, name, x1, y1, x2, y2 in colors:
                        if x1 < x < x2 and y1 < y < y2:
                            draw_color = color

            # Drawing mode: only index
            elif fingers == [1, 0]:
                size = eraser_size if draw_color == (0, 0, 0) else brush_size

                cv2.circle(frame, (x, y), size, draw_color, -1)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, size)
                prev_x, prev_y = x, y

            else:
                prev_x, prev_y = 0, 0

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.putText(frame, "Index: Draw | Index+Middle: Move/Select",
                (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, "C: Clear | S: Save | Q: Quit",
                (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Advanced Gesture Drawing", frame)

    key = cv2.waitKey(1)

    if key == ord('c'):
        canvas = np.zeros_like(frame)

    elif key == ord('s'):
        filename = f"drawing_{int(time.time())}.png"
        cv2.imwrite(filename, canvas)
        print("Saved:", filename)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()