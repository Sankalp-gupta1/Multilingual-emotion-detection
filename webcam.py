# ✅ webcam access karegi
# ✅ face detect karegi
# ✅ facial emotion predict karegi
# ✅ frame pe emotion likhegi
# ✅ Streamlit ko live frame degi
# ==========================================
# WEBCAM + FACIAL EMOTION DETECTION
# ==========================================

import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# ==========================================
# LOAD MODEL
# ==========================================

model = load_model(
    r"D:\emotion-detection-whatsap\facial_emotion\face_model\face_emotion_model_v2.keras",
    compile=False
)

# ==========================================
# LABELS
# ==========================================

labels = [
    'Angry 😡',
    'Disgust 🤢',
    'Fear 😨',
    'Happy 😄',
    'Neutral 😐',
    'Sad 😢',
    'Surprise 😲'
]

# ==========================================
# FACE CASCADE
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================================
# START CAMERA
# ==========================================

def start_camera():

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        return None, "Cannot access webcam"

    return cap, "Camera started"

# ==========================================
# STOP CAMERA
# ==========================================

def stop_camera(cap):

    cap.release()

# ==========================================
# DETECT EMOTION
# ==========================================

def detect_emotion(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    emotions_found = []

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (224, 224))

        face = face.astype("float32") / 255.0

        face = np.expand_dims(face, axis=0)

        prediction = model.predict(
            face,
            verbose=0
        )

        pred_index = np.argmax(prediction)

        confidence = float(
            np.max(prediction) * 100
        )

        emotion = labels[pred_index]

        emotions_found.append({
            "emotion": emotion,
            "confidence": confidence
        })

        # RECTANGLE
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # TEXT
        cv2.putText(
            frame,
            f"{emotion} {confidence:.1f}%",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    # ======================================
    # SAVE FRAME FOR ALL CLIENTS
    # ======================================

    cv2.imwrite(
        "shared_camera.jpg",
        frame
    )

    # SAVE EMOTIONS
    with open(
        "shared_emotion.json",
        "w"
    ) as f:

        json.dump(
            emotions_found,
            f
        )

    return frame, emotions_found