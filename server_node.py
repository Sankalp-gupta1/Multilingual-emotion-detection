import asyncio
import websockets
import json
import requests
import mysql.connector

# URL of running FastAPI model service
MODEL_SERVICE_URL = "http://127.0.0.1:8000/predict"

clients = set()

# MySQL connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="rishikesh@astha",
    database="chatdb"
)
cursor = db.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    id INT AUTO_INCREMENT PRIMARY KEY,
    sender VARCHAR(100),
    text TEXT,
    emotion VARCHAR(50),
    sarcasm VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
db.commit()

def save_to_db(msg_data):
    try:
        cursor.execute(
            "INSERT INTO messages (sender, text, emotion, sarcasm) VALUES (%s, %s, %s, %s)",
            (msg_data["sender"], msg_data["text"], msg_data["emotion"], msg_data["sarcasm"])
        )
        db.commit()
    except Exception as e:
        print("DB insert error:", e)

async def handle(websocket):
    sender = "Unknown"
    try:
        # First message: sender name
        raw_msg = await websocket.recv()
        try:
            incoming = json.loads(raw_msg)
            sender = incoming.get("sender", "Unknown")
        except Exception:
            sender = "Unknown"

        clients.add(websocket)
        print(f"Client connected: {sender}")

        # Broadcast join message
        join_msg = json.dumps({
            "sender": "Server",
            "text": f"{sender} joined the chat",
            "emotion": "Info",
            "sarcasm": "No"
        })
        for c in clients.copy():
            try:
                await c.send(join_msg)
            except:
                clients.discard(c)

        async for raw_msg in websocket:
            try:
                incoming = json.loads(raw_msg)
                msg = incoming.get("text", "").strip()
            except Exception:
                msg = raw_msg.strip()

            if not msg:
                continue  # skip empty messages

            # Call ML model service
            try:
                resp = requests.post(MODEL_SERVICE_URL, json={"text": msg}, timeout=5)
                data = resp.json()
            except Exception as e:
                print("Model service error:", e)
                data = {"emotion": "Unknown", "sarcasm": "Unknown"}

            enriched = {
                "sender": sender,
                "text": msg,
                "emotion": data.get("emotion", "Unknown"),
                "sarcasm": data.get("sarcasm", "Unknown")
            }

            # Print to server console
            print(f"{sender}: {msg} | Emotion: {enriched['emotion']} | Sarcasm: {enriched['sarcasm']}")

            # Save in DB
            save_to_db(enriched)

            # Broadcast to all clients
            payload = json.dumps(enriched, ensure_ascii=False)
            for c in clients.copy():
                try:
                    await c.send(payload)
                except:
                    clients.discard(c)

    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {sender}")
    finally:
        clients.discard(websocket)

async def main():
    server = await websockets.serve(
        handle, "0.0.0.0", 6789, ping_interval=20, ping_timeout=60
    )
    print("WebSocket server started at ws://0.0.0.0:6789")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())












# import asyncio
# import websockets
# import json
# import requests
# import mysql.connector

# # URL of your running FastAPI model service
# MODEL_SERVICE_URL = "http://127.0.0.1:8000/predict"

# clients = set()

# # MySQL connection
# db = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="rishikesh@astha",
#     database="chatdb"
# )
# cursor = db.cursor()

# # Create table if not exists
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS messages (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     sender VARCHAR(100),
#     text TEXT,
#     emotion VARCHAR(50),
#     sarcasm VARCHAR(50),
#     timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# )
# """)
# db.commit()

# def save_to_db(msg_data):
#     try:
#         cursor.execute(
#             "INSERT INTO messages (sender, text, emotion, sarcasm) VALUES (%s, %s, %s, %s)",
#             (msg_data["sender"], msg_data["text"], msg_data["emotion"], msg_data["sarcasm"])
#         )
#         db.commit()
#     except Exception as e:
#         print("DB insert error:", e)

# async def handle(websocket):
#     sender = "Unknown"
#     try:
#         # First message: sender name
#         raw_msg = await websocket.recv()
#         try:
#             incoming = json.loads(raw_msg)
#             sender = incoming.get("sender", "Unknown")
#         except Exception:
#             sender = "Unknown"

#         clients.add(websocket)
#         print(f"Client connected: {sender}")

#         # Broadcast join message
#         join_msg = json.dumps({
#             "sender": "Server",
#             "text": f"{sender} joined the chat",
#             "emotion": "Info",
#             "sarcasm": "No"
#         })
#         for c in clients.copy():
#             try:
#                 await c.send(join_msg)
#             except:
#                 clients.discard(c)

#         async for raw_msg in websocket:
#             try:
#                 incoming = json.loads(raw_msg)
#                 msg = incoming.get("text", "").strip()
#             except Exception:
#                 msg = raw_msg.strip()

#             if not msg:
#                 continue  # skip empty messages

#             # Call ML model service
#             try:
#                 resp = requests.post(MODEL_SERVICE_URL, json={"text": msg}, timeout=5)
#                 data = resp.json()
#             except Exception as e:
#                 print("Model service error:", e)
#                 data = {"emotion": "Unknown", "sarcasm": "Unknown"}

#             enriched = {
#                 "sender": sender,
#                 "text": msg,
#                 "emotion": data.get("emotion", "Unknown"),
#                 "sarcasm": data.get("sarcasm", "Unknown")
#             }

#             # Print to server console
#             print(f"{sender}: {msg} | Emotion: {enriched['emotion']} | Sarcasm: {enriched['sarcasm']}")

#             # Save in DB
#             save_to_db(enriched)

#             # Broadcast to all clients
#             payload = json.dumps(enriched, ensure_ascii=False)
#             for c in clients.copy():
#                 try:
#                     await c.send(payload)
#                 except:
#                     clients.discard(c)

#     except websockets.exceptions.ConnectionClosed:
#         print(f"Client disconnected: {sender}")
#     finally:
#         clients.discard(websocket)

# async def main():
#     server = await websockets.serve(
#         handle, "0.0.0.0", 6789, ping_interval=20, ping_timeout=60
#     )
#     print("WebSocket server started at ws://0.0.0.0:6789")
#     await server.wait_closed()

# if __name__ == "__main__":
#     asyncio.run(main())








 