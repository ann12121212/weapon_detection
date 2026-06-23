
import cv2
import numpy as np
import time
from ultralytics import YOLO
import mediapipe as mp

# Load YOLOv8 model (e.g., weapon detection)
yolo_model = YOLO("best.pt")  # Replace with your model

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Webcam setup
cap = cv2.VideoCapture(0)
start_program_time = time.time()  # Timer for gaze detection

if not cap.isOpened():
    print("❌ Could not open video source")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = face_mesh.process(frame_rgb)
    frame_rgb.flags.writeable = True
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # --------- Gaze Estimation ---------
    current_time = time.time()
    elapsed_time = current_time - start_program_time

    img_h, img_w, _ = frame.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            if face_2d.shape[0] < 4 or face_3d.shape[0] < 4:
                continue

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            if elapsed_time > 5:
                if y < -10:
                    gaze_text = "Looking Left"
                elif y > 10:
                    gaze_text = "Looking Right"
                elif x < -10:
                    gaze_text = "Looking Down"
                elif x > 10:
                    gaze_text = "Looking Up"
                else:
                    gaze_text = "Gaze Alert!!"

                cv2.putText(frame, gaze_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Starting in {int(5 - elapsed_time)} sec...",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

            # Draw landmarks
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # --------- YOLOv8 Detection ---------
    yolo_results = yolo_model.predict(source=frame, verbose=False)
    annotated_frame = yolo_results[0].plot()

    # --------- Show both on single frame ---------
    cv2.imshow("YOLO + Gaze Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
