import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model(r"D:\emotion-detection-whatsap\facial_emotion\face_model\face_emotion_model_v2.keras")

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # convert BGR -> RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # resize to model input
        img = cv2.resize(img, (224, 224))

        # normalize
        img = img / 255.0

        # reshape for model
        img = np.reshape(img, (1, 224, 224, 3))

        # prediction
        pred = model.predict(img, verbose=0)
        emotion = emotions[np.argmax(pred)]

        cv2.putText(frame, emotion, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    except Exception as e:
        print(e)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()