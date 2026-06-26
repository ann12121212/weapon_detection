import cv2
import numpy as np
import time
import threading
from ultralytics import YOLO
import mediapipe as mp
import os
from dotenv import load_dotenv

from flask import Flask, render_template, Response
from flask_socketio import SocketIO

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()                         

# ---------- Global context for the AI chatbot ----------
latest_detection_context = {
    "gaze_behavior": "no suspicious gaze",
    "detected_object": "none",
    "location": "Campus Entrance"      
}

def generate_safegaze_protocol(gaze_behavior, detected_object, location):
    try:
        # Using standard gemini-2.0-flash to avoid experimental free-tier lite quotas
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-flash",   
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2
        )

        prompt_template = PromptTemplate(
            input_variables=["gaze_behavior", "detected_object", "location"],
            template=(
                "SafeGaze-AI Alert: Proactive threat detection triggered at {location}. "
                "System analysis indicates {gaze_behavior} combined with the potential presence of a {detected_object}. "
                "Generate a 5-step immediate action plan for on-ground security personnel to proactively assess, "
                "monitor, and safely de-escalate this situation before it develops into an incident."
            )
        )

        formatted_prompt = prompt_template.format(
            gaze_behavior=gaze_behavior,
            detected_object=detected_object,
            location=location
        )

        response = llm.invoke(formatted_prompt)
        return response.content   
    except Exception as e:
        print(f"⚠️ API Error: {e}. Utilizing fallback local security protocol.")
        return (
            f"🚨 **[FAIL-SAFE SECURITY PROTOCOL] Threat Level Raised at {location}**\n\n"
            f"**Context:** Unverified {detected_object.upper()} tracking with status: '{gaze_behavior}'.\n\n"
            f"1. **Isolate & Track:** Monitor target via auxiliary cameras immediately.\n"
            f"2. **Dispatch Units:** Quietly deploy ground units to intercept adjacent pathways.\n"
            f"3. **Clear Corridor:** Subtly redirect nearby pedestrian traffic away from {location}.\n"
            f"4. **Standby Containment:** Prepare facility access control barriers.\n"
            f"5. **Escalation:** If verified visually by ground units, contact authorities instantly."
        )

# ---------- Flask & SocketIO ----------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# ---------- Shared frame (for video streaming) ----------
output_frame = None
frame_lock = threading.Lock()

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# ---------- Video Streaming Generator ----------
def gen_frames():
    global output_frame
    while True:
        frame_to_encode = None
        
        with frame_lock:
            if output_frame is not None and output_frame.size > 0:
                frame_to_encode = output_frame.copy()

        # If there's no frame yet, sleep OUTSIDE the lock so the detection thread can run
        if frame_to_encode is None:
            time.sleep(0.03)
            continue

        # Encode outside the lock to keep the server lightweight and responsive
        ret, buffer = cv2.imencode('.jpg', frame_to_encode)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('chat_message')
def handle_chat_message(data):
    user_message = data.get('message', '').lower()
    if 'protocol' in user_message or 'action' in user_message:
        ctx = latest_detection_context
        if ctx['detected_object'] != 'none':
            try:
                protocol = generate_safegaze_protocol(
                    gaze_behavior=ctx['gaze_behavior'],
                    detected_object=ctx['detected_object'],
                    location=ctx['location']
                )
                reply = protocol
            except Exception as e:
                reply = f"Error generating protocol: {str(e)}"
        else:
            reply = "No weapon currently detected. I can't generate a threat protocol yet."
    else:
        reply = "You can ask me for a 'security protocol' when a weapon is detected. Type 'protocol' to generate one."
    socketio.emit('chat_response', {'message': reply})

def send_detection_update(data):
    socketio.emit('detection_update', data)

# ---------- YOLO & MediaPipe Setup ----------
yolo_model = YOLO("best.pt")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# ---------- Background Detection Loop ----------
def detection_loop():
    global output_frame
    cap = cv2.VideoCapture(0)
    
    start_time = time.time()
    prev_frame_time = time.time() # ⏱️ Added for FPS tracking
    
    # TIMERS: Prevent flooding the UI with messages
    last_gaze_emit = 0
    last_weapon_emit = 0

    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True
        display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        current_time = time.time()
        elapsed_time = current_time - start_time
        gaze_text = ""

        # ---------- Gaze Estimation ----------
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                img_h, img_w, _ = display_frame.shape
                face_2d = []
                face_3d = []

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

                if face_2d.shape[0] >= 4 and face_3d.shape[0] >= 4:
                    focal_length = 1 * img_w
                    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                           [0, focal_length, img_w / 2],
                                           [0, 0, 1]])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    if success:
                        rmat, _ = cv2.Rodrigues(rot_vec)
                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                        x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360

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
                        else:
                            gaze_text = f"Starting in {int(5 - elapsed_time)} sec..."

                mp_drawing.draw_landmarks(
                    image=display_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

                if gaze_text:
                    cv2.putText(display_frame, gaze_text, (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # ---------- YOLOv8 Weapon Detection + Performance ----------
        yolo_results = yolo_model.predict(source=display_frame, conf=0.15, verbose=False)
        result = yolo_results[0]
        display_frame = result.plot()

        # YOLO timings
        preprocess = result.speed["preprocess"]
        inference = result.speed["inference"]
        postprocess = result.speed["postprocess"]

        # FPS calculation
        current_frame_time = time.time()
        
        # Prevent division by zero if processing is incredibly fast
        time_diff = current_frame_time - prev_frame_time
        fps = 1 / time_diff if time_diff > 0 else 0
        prev_frame_time = current_frame_time

        # Display performance on frame
        cv2.putText(display_frame,
                    f"FPS: {fps:.1f}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2)

        cv2.putText(display_frame,
                    f"Inference: {inference:.1f} ms",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2)

        cv2.putText(display_frame,
                    f"Pre: {preprocess:.1f}  Post: {postprocess:.1f} ms",
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2)

        # Log performance to terminal
        print(f"FPS: {fps:.1f} | "
              f"Pre: {preprocess:.1f} ms | "
              f"Inference: {inference:.1f} ms | "
              f"Post: {postprocess:.1f} ms")

        weapons_detected = False
        weapon_label = ""

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = yolo_model.names[class_id]
                confidence = box.conf.item()

                if confidence > 0.15:
                    print(f"👀 YOLO sees: {class_name} (Conf: {confidence:.2f})")

                if class_name.lower() in ['knife', 'gun', 'pistol', 'rifle', 'weapon', 'scissors']:
                    weapons_detected = True
                    weapon_label = class_name
                    break

        # Share frame with Flask safely
        if display_frame is not None and display_frame.size > 0:
            with frame_lock:
                output_frame = display_frame.copy()

        # Context Updates for LLM
        if weapons_detected:
            latest_detection_context['detected_object'] = weapon_label
        else:
            latest_detection_context['detected_object'] = "none"

        if gaze_text and elapsed_time > 5 and gaze_text.startswith("Looking"):
            latest_detection_context['gaze_behavior'] = gaze_text
        else:
            latest_detection_context['gaze_behavior'] = "no suspicious gaze"

        # ---------- Throttled Socket Emits ----------
        # Update weapon to UI once every 3 seconds
        if weapons_detected:
            if current_time - last_weapon_emit > 3.0:
                send_detection_update({
                    'type': 'weapon',
                    'message': f'{weapon_label.capitalize()} detected!'
                })
                last_weapon_emit = current_time

        # Update gaze to UI exactly every 10 seconds
        if gaze_text and elapsed_time > 5 and not gaze_text.startswith("Starting"):
            if current_time - last_gaze_emit > 10.0:
                send_detection_update({
                    'type': 'gaze',
                    'message': gaze_text
                })
                last_gaze_emit = current_time

        time.sleep(0.03)

    cap.release()

# ---------- Start Server ----------
if __name__ == '__main__':
    print("Starting Safegaze_AI server...")
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
