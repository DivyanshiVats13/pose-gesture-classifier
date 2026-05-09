import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

GESTURES = ["thumbs_up", "thumbs_down", "open_palm", "fist", "point"]
SAMPLES_PER_CLASS = 300
OUTPUT_FILE = "gesture_data.csv"

def extract_landmarks(results):
    """Flatten pose + both-hand landmarks into a 1D feature vector."""
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

    return np.concatenate([pose, lh, rh])   # 132 + 63 + 63 = 258 features


def collect():
    cap = cv2.VideoCapture(0)
    header_written = os.path.exists(OUTPUT_FILE)

    with mp_holistic.Holistic(min_detection_confidence=0.7,
                               min_tracking_confidence=0.7) as holistic:
        for gesture in GESTURES:
            count = 0
            print(f"\n>>> Get ready for: {gesture}. Press SPACE to start.")

            while True:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"READY: {gesture} | Press SPACE",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.imshow("Collector", frame)
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break

            while count < SAMPLES_PER_CLASS:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks,
                                          mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                                          mp_holistic.HAND_CONNECTIONS)

                features = extract_landmarks(results)

                with open(OUTPUT_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    if not header_written:
                        writer.writerow([f"f{i}" for i in range(258)] + ["label"])
                        header_written = True
                    writer.writerow(list(features) + [gesture])

                count += 1
                cv2.putText(frame, f"{gesture}: {count}/{SAMPLES_PER_CLASS}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
                cv2.imshow("Collector", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Data saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    collect()