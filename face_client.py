import cv2
import numpy as np
import asyncio
import websockets
import json
from tensorflow.keras.models import load_model

# =========================
# LOAD MODEL
# =========================
model = load_model(
    r"C:\Users\hp pc\emotion-detection-whatsapp\facial_emotion\face_model\face_emotion_model.keras"
)

# =========================
# LABELS
# =========================
labels = [
    'Angry 😡',
    'Disgust 🤢',
    'Fear 😨',
    'Happy 😄',
    'Neutral 😐',
    'Sad 😢',
    'Surprise 😲'
]

# =========================
# FACE DETECTOR
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# =========================
# SERVER URL
# =========================
SERVER_URL = "ws://127.0.0.1:6790"

# =========================
# MAIN FUNCTION
# =========================
async def run_camera():

    sender = input("Enter your name: ").strip()

    async with websockets.connect(SERVER_URL) as websocket:

        # send join request
        await websocket.send(json.dumps({
            "type": "join",
            "sender": sender
        }))

        print("Connected to face server")

        # webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Cannot access webcam")
            return

        print("Webcam started. Press Q to quit.")

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:

                # crop face
                face = frame[y:y+h, x:x+w]

                # resize
                face = cv2.resize(face, (224, 224))

                # normalize
                face = face.astype("float32") / 255.0

                # reshape
                face = np.expand_dims(face, axis=0)

                # predict
                prediction = model.predict(face, verbose=0)

                pred_index = np.argmax(prediction)

                emotion = labels[pred_index]

                confidence = float(np.max(prediction) * 100)

                # draw rectangle
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0,255,0),
                    2
                )

                # draw text
                cv2.putText(
                    frame,
                    f"{emotion} ({confidence:.1f}%)",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

                # send to server
                data = {
                    "type": "emotion",
                    "sender": sender,
                    "emotion": emotion,
                    "confidence": confidence
                }

                await websocket.send(json.dumps(data))

            cv2.imshow("Live Facial Emotion Detection", frame)

            # quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# =========================
# RUN
# =========================
asyncio.run(run_camera())