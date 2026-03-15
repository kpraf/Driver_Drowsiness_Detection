"""
DROWSY DRIVER DETECTION SYSTEM
"""

from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
from collections import deque
import imutils
import dlib
import cv2
import csv
import time
import datetime
import numpy as np


# AUDIO SETUP
mixer.init()
mixer.music.load("music.wav")


# CONSTANTS
FRAME_CHECK    = 20      # consecutive frames before alert
CALIB_SECONDS  = 4       # calibration window duration
EAR_GRAPH_LEN  = 100     # number of EAR values shown in graph
DEFAULT_THRESH = 0.25    # fallback if calibration fails


# EYE ASPECT RATIO
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# CSV LOGGER
def log_event(writer, event_type, ear=None):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    writer.writerow([ts, event_type, f"{ear:.4f}" if ear is not None else "N/A"])


# EAR GRAPH (separate window)
def build_ear_graph(ear_history, thresh, width=400, height=200):
    graph = np.zeros((height, width, 3), dtype=np.uint8)

    # Background grid lines
    for y in range(0, height, 40):
        cv2.line(graph, (0, y), (width, y), (40, 40, 40), 1)
    for x in range(0, width, 40):
        cv2.line(graph, (x, 0), (x, height), (40, 40, 40), 1)

    max_val = 0.50
    pad = 20

    # Threshold line
    ty = int(height - pad - ((thresh / max_val) * (height - 2 * pad)))
    ty = max(pad, min(ty, height - pad))
    cv2.line(graph, (0, ty), (width, ty), (0, 80, 255), 1)
    cv2.putText(graph, f"Threshold: {thresh:.2f}", (5, ty - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 120, 255), 1)

    # EAR line plot
    if len(ear_history) > 1:
        pts = []
        for i, val in enumerate(ear_history):
            px = int(i * (width - 1) / (EAR_GRAPH_LEN - 1))
            py = int(height - pad - (min(val, max_val) / max_val) * (height - 2 * pad))
            py = max(pad, min(py, height - pad))
            pts.append((px, py))
        for i in range(1, len(pts)):
            color = (50, 220, 50) if ear_history[i] >= thresh else (50, 80, 255)
            cv2.line(graph, pts[i - 1], pts[i], color, 2)

    # Labels
    cv2.putText(graph, "EAR Monitor", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
    cv2.putText(graph, f"Current EAR: {list(ear_history)[-1]:.3f}" if ear_history else "No data",
                (210, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 235, 59), 1)
    cv2.putText(graph, "0.00", (5, height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(graph, f"{max_val:.2f}", (5, pad + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    return graph


# DLIB SETUP
detect  = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)

# State
flag          = 0
alert_count   = 0
ear_history   = deque(maxlen=EAR_GRAPH_LEN)
thresh        = DEFAULT_THRESH
session_start = time.time()

# CSV
csv_file   = open("drowsiness_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "Event", "EAR"])


# PHASE 1 ── CALIBRATION
print("[INFO] Calibration phase started — keep your eyes open.")
calib_ears  = []
calib_start = time.time()

while True:
    elapsed   = time.time() - calib_start
    remaining = CALIB_SECONDS - elapsed
    if remaining <= 0:
        break

    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=450)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for subject in detect(gray, 0):
        shape    = face_utils.shape_to_np(predict(gray, subject))
        leftEye  = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        ear      = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        calib_ears.append(ear)

    # Progress bar
    bar_w    = 300
    progress = int((elapsed / CALIB_SECONDS) * bar_w)
    cv2.rectangle(frame, (75, 220), (75 + bar_w, 245), (60, 60, 60), -1)
    cv2.rectangle(frame, (75, 220), (75 + progress, 245), (0, 220, 180), -1)

    cv2.putText(frame, "CALIBRATION", (150, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 180), 2)
    cv2.putText(frame, "Keep your eyes OPEN and look forward",
                (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, f"Measuring baseline EAR...  {remaining:.1f}s",
                (90, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 180), 1)

    cv2.imshow("Drowsy Detector", frame)
    cv2.waitKey(1)

# Set threshold
if calib_ears:
    avg_ear = float(np.mean(calib_ears))
    thresh  = round(avg_ear * 0.75, 4)
    print(f"[INFO] Calibration done.  Avg EAR = {avg_ear:.3f}  |  Threshold = {thresh:.3f}")
else:
    print(f"[WARN] No face detected during calibration. Using default threshold = {thresh}")

log_event(csv_writer, "SESSION_START")


# PHASE 2 ── MAIN DETECTION LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=450)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_ear = 0.0

    for subject in detect(gray, 0):
        shape    = face_utils.shape_to_np(predict(gray, subject))
        leftEye  = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # ── EAR ──────────────────────────────────────────────
        ear         = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        current_ear = ear

        # ── Draw Eye Contours ─────────────────────────────────
        cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # ── HUD ───────────────────────────────────────────────
        cv2.putText(frame, f"EAR: {ear:.2f}",
                    (340, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 235, 59), 2)

        # ── Drowsiness Check ──────────────────────────────────
        if ear < thresh:
            flag += 1
            if flag >= FRAME_CHECK:
                cv2.putText(frame, "** DROWSY ALERT **", (60, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
                cv2.putText(frame, "** DROWSY ALERT **", (60, 420),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
                if not mixer.music.get_busy():
                    mixer.music.play()
                    alert_count += 1
                    log_event(csv_writer, "DROWSY_ALERT", ear)
        else:
            flag = 0

    # ── Alert counter ─────────────────────────────────────────
    cv2.putText(frame, f"Alerts: {alert_count}",
                (340, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    cv2.putText(frame, f"Thresh: {thresh:.2f}",
                (340, 435), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)

    # ── Separate EAR graph window ─────────────────────────────
    ear_history.append(current_ear)
    graph = build_ear_graph(ear_history, thresh)
    cv2.imshow("EAR Monitor", graph)

    # ── Main camera window ────────────────────────────────────
    cv2.imshow("Drowsy Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") \
       or cv2.getWindowProperty("Drowsy Detector", cv2.WND_PROP_VISIBLE) < 1 \
       or cv2.getWindowProperty("EAR Monitor",     cv2.WND_PROP_VISIBLE) < 1:
        break

# PHASE 3 ── SESSION SUMMARY
log_event(csv_writer, "SESSION_END")
csv_file.close()
cv2.destroyAllWindows()
cap.release()

session_secs = time.time() - session_start
mins         = int(session_secs // 60)
secs         = int(session_secs % 60)

summary = f"""
╔══════════════════════════════════════════╗
║            SESSION SUMMARY               ║
╠══════════════════════════════════════════╣
║  Duration        : {mins}m {secs:02d}s
║  Total Alerts    : {alert_count}
║  EAR Threshold   : {thresh:.4f} (calibrated)
║  Log saved to    : drowsiness_log.csv
╚══════════════════════════════════════════╝
"""
print(summary)