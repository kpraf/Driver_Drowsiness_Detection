# Drowsy Driver Detection System v2.1

A computer vision system that detects driver drowsiness using **eye aspect ratio (EAR)** and facial landmarks.
The program monitors eye closure through a webcam and triggers an alarm if the driver appears drowsy.

## Features

* Eye Aspect Ratio (EAR) based drowsiness detection
* Automatic **calibration phase**
* **Real-time webcam monitoring**
* **Separate EAR graph window**
* **Audio alarm when drowsiness is detected**
* **CSV logging of events**
* **Session summary report**

---

# System Requirements

* Python **3.11 (recommended)**
* Webcam
* Windows / Mac / Linux

---

# Project Folder Structure

Your folder should look like this:

```
Driver_Drowsiness_Detection
│
├── Drowsiness_Detection.py
├── music.wav
├── drowsiness_log.csv        (auto-generated after running)
│
├── models
│   └── shape_predictor_68_face_landmarks.dat
│
└── README.md
```

⚠ Important:
The file **shape_predictor_68_face_landmarks.dat** is required for facial landmark detection.

---

# Step 1 — Install Python

Download Python **3.11** from:

https://www.python.org/downloads/

During installation make sure to check:

```
Add Python to PATH
```

---

# Step 2 — Install Required Libraries

Open **Command Prompt or PowerShell** and install the dependencies:

```
py -3.11 -m pip install opencv-python dlib imutils scipy pygame numpy
```

If Python 3.11 is your default version, you can also run:

```
pip install opencv-python dlib imutils scipy pygame numpy
```

---

# The facial landmark model is already included in the project folder.

You do not need to download or install anything for this step. The required file is already located in the **models** directory.

Example:

```
models/shape_predictor_68_face_landmarks.dat
```

---

# Step 4 — Run the Program

Navigate to the project folder:

```
cd Driver_Drowsiness_Detection
```

Run the program:

```
py -3.11 Drowsiness_Detection.py
```

---

# How the Program Works

## Phase 1 — Calibration

The system first measures your normal **Eye Aspect Ratio (EAR)** while your eyes are open.

This takes about **4 seconds**.

The program then calculates a personalized threshold.

---

## Phase 2 — Detection

The webcam monitors your eyes continuously.

If your EAR drops below the threshold for **20 consecutive frames**, the system assumes drowsiness and:

* Displays **DROWSY ALERT**
* Plays an **alarm sound**
* Logs the event

---

## Phase 3 — Session Summary

After closing the program, a summary appears:

```
SESSION SUMMARY
Duration
Total Alerts
EAR Threshold
Log file location
```

---

# Controls

Press:

```
Q
```

to quit the program.

Closing either window will also end the session.

---

# Output

The system creates a log file:

```
drowsiness_log.csv
```

Example log:

```
Timestamp,Event,EAR
2026-03-15 10:12:01,SESSION_START,N/A
2026-03-15 10:12:18,DROWSY_ALERT,0.19
2026-03-15 10:13:00,SESSION_END,N/A
```

---

# Troubleshooting

## Webcam not detected

If the camera does not open, change this line in the code:

```
cap = cv2.VideoCapture(0)
```

Try:

```
cap = cv2.VideoCapture(1)
```

---

## No face detected

Make sure:

* The webcam is working
* Your face is visible
* Lighting is sufficient

---

## Alarm not playing

Check that the file exists:

```
music.wav
```

---

# Dependencies

* OpenCV
* dlib
* imutils
* scipy
* pygame
* numpy
