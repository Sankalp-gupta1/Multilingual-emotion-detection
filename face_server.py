import asyncio
import websockets
import json
import mysql.connector

# =========================
# MYSQL CONNECTION
# =========================
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="rishikesh@astha",
    database="chatdb"
)

cursor = db.cursor()

# =========================
# ACTIVE CAMERA HOLDER
# =========================
active_camera_user = None

# connected clients
clients = set()

# =========================
# SAVE TO DATABASE
# =========================
def save_to_db(sender, emotion, confidence):

    try:

        cursor.execute(
            """
            INSERT INTO facial_emotions
            (sender, emotion, confidence)
            VALUES (%s, %s, %s)
            """,
            (sender, emotion, confidence)
        )

        db.commit()

    except Exception as e:
        print("DB Error:", e)

# =========================
# HANDLE CLIENT
# =========================
async def handle(websocket):

    global active_camera_user

    sender = "Unknown"

    try:

        # receive join request
        raw = await websocket.recv()

        data = json.loads(raw)

        sender = data.get("sender", "Unknown")

        # =========================
        # SINGLE CAMERA LOCK
        # =========================
        if active_camera_user is None:

            active_camera_user = sender

            await websocket.send(json.dumps({
                "status": "allowed",
                "message": f"{sender} got camera access"
            }))

            print(f"{sender} got webcam access")

        else:

            await websocket.send(json.dumps({
                "status": "denied",
                "message": f"Camera already in use by {active_camera_user}"
            }))

            print(f"{sender} denied webcam access")

            return

        clients.add(websocket)

        # =========================
        # RECEIVE EMOTIONS
        # =========================
        async for raw_msg in websocket:

            try:

                msg = json.loads(raw_msg)

                if msg["type"] == "emotion":

                    emotion = msg["emotion"]

                    confidence = msg["confidence"]

                    print(
                        f"{sender} | {emotion} | {confidence:.2f}%"
                    )

                    # save in SQL
                    save_to_db(
                        sender,
                        emotion,
                        confidence
                    )

            except Exception as e:

                print("Message Error:", e)

    except websockets.exceptions.ConnectionClosed:

        print(f"{sender} disconnected")

    finally:

        clients.discard(websocket)

        # release webcam lock
        if active_camera_user == sender:

            active_camera_user = None

            print(f"{sender} released webcam access")

# =========================
# MAIN SERVER
# =========================
async def main():

    server = await websockets.serve(
        handle,
        "0.0.0.0",
        6790
    )

    print("Face Emotion Server Started")
    print("ws://0.0.0.0:6790")

    await server.wait_closed()

# =========================
# RUN SERVER
# =========================
asyncio.run(main())