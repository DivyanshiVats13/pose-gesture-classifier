import asyncio
import cv2
import mediapipe as mp
import numpy as np
import joblib
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Gesture Classifier API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

bundle = joblib.load("model/gesture_model.pkl")
clf, le = bundle["model"], bundle["encoder"]

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

def extract_landmarks(results):
    pose = np.zeros(33 * 4)
    lh   = np.zeros(21 * 3)
    rh   = np.zeros(21 * 3)
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            pose[i*4:i*4+4] = [lm.x, lm.y, lm.z, lm.visibility]
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            lh[i*3:i*3+3] = [lm.x, lm.y, lm.z]
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            rh[i*3:i*3+3] = [lm.x, lm.y, lm.z]
    return np.concatenate([pose, lh, rh])

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        await ws.close()
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.01)
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                rgb.flags.writeable = True

                mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                           mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                                           mp_holistic.HAND_CONNECTIONS)

                features = extract_landmarks(results).reshape(1, -1)
                gesture  = le.inverse_transform(clf.predict(features))[0]
                conf     = float(clf.predict_proba(features).max())

                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                img_b64 = base64.b64encode(buf).decode()

                await ws.send_json({"gesture": gesture,
                                     "confidence": round(conf, 3),
                                     "frame": img_b64})
                await asyncio.sleep(0.04)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"WS error: {e}")
        finally:
            cap.release()

@app.get("/classes")
def get_classes():
    return {"classes": list(le.classes_)}

@app.get("/health")
def health():
    return {"status": "ok"}