# smart driver drowsiness detection (accident prevention)
# R.R.S 28/06/2026

import cv2
import mediapipe as mp
import numpy as np
import os
from playsound import playsound
import threading
import time

# -----------------------------
# Alarm Function
# -----------------------------
alarm_on = False

def alarm():
    global alarm_on
    while alarm_on:
        playsound(os.path.join(os.path.dirname(__file__),
    "alarm.wav"))

# -----------------------------
# MediaPipe Face Mesh
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# Eye Landmark Indexes
# -----------------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20

counter = 0

# -----------------------------
# Eye Aspect Ratio Function
# -----------------------------
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
# -----------------------------
# Main Loop
# -----------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            h, w, _ = frame.shape

            left_eye = []
            right_eye = []

            for idx in LEFT_EYE:
                point = face_landmarks.landmark[idx]
                left_eye.append([int(point.x * w), int(point.y * h)])

            for idx in RIGHT_EYE:
                point = face_landmarks.landmark[idx]
                right_eye.append([int(point.x * w), int(point.y * h)])

            left_eye = np.array(left_eye, dtype=np.float32)
            right_eye = np.array(right_eye, dtype=np.float32)

            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)

            ear = (leftEAR + rightEAR) / 2.0

            cv2.putText(frame,
                        f"EAR: {ear:.2f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2)
            if ear < EAR_THRESHOLD:
                counter += 1

                cv2.putText(
                    frame,
                    "DROWSINESS DETECTED!",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )

                if counter >= FRAME_THRESHOLD:
                    if not alarm_on:
                        alarm_on = True
                        threading.Thread(target=alarm, daemon=True).start()

                    cv2.putText(
                        frame,
                        "WAKE UP!",
                        (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )

            else:
                counter = 0
                if alarm_on:
                    alarm_on=False
                    time.sleep(0.2) 

            # Draw eye landmarks
            for p in left_eye:
                cv2.circle(frame, tuple(p.astype(int)), 2, (0, 255, 0), -1)

            for p in right_eye:
                cv2.circle(frame, tuple(p.astype(int)), 2, (0, 255, 0), -1)

    cv2.imshow("Driver Drowsiness Detection", frame)
    # Press 'q' to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()

# Stop alarm before exiting
alarm_on = False
time.sleep(0.5)