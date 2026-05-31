import asyncio
import websockets
import json

async def chat_client():
    uri = "ws://127.0.0.1:6789"
    sender = input("Enter your name: ").strip() or "Anonymous"

    async with websockets.connect(uri) as websocket:
        # Send initial name to server
        await websocket.send(json.dumps({"sender": sender, "text": ""}))

        print(f"Connected to chat server as {sender}. Type messages. Ctrl+C to exit.\n")

        message_queue = asyncio.Queue()

        # Background task: receive messages and put in queue
        async def recv():
            while True:
                try:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    await message_queue.put(data)
                except websockets.exceptions.ConnectionClosed:
                    await message_queue.put({"sender":"Server","text":"Server closed connection","emotion":"Info","sarcasm":"No"})
                    break
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    await message_queue.put({"sender":"Error","text":str(e),"emotion":"Error","sarcasm":"No"})
                    break

        recv_task = asyncio.create_task(recv())

        # Background task: print messages from queue
        async def printer():
            while True:
                try:
                    data = await message_queue.get()
                    sender_name = data.get('sender', 'Unknown')
                    prompt = "You: " if sender_name == sender else f"{sender_name}: "
                    print(
                        f"\n➡ {sender_name}: {data['text']}\n"
                        f"   Emotion: {data['emotion']}\n"
                        f"   Sarcasm: {data['sarcasm']}\n{prompt}",
                        end="", flush=True
                    )
                except asyncio.CancelledError:
                    break

        printer_task = asyncio.create_task(printer())

        try:
            while True:
                text = await asyncio.to_thread(input, "You: ")
                text = text.strip()
                if not text:
                    continue
                await websocket.send(json.dumps({"sender": sender, "text": text}))
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            recv_task.cancel()
            printer_task.cancel()
            try:
                await websocket.close()
            except:
                pass

if __name__ == "__main__":
    asyncio.run(chat_client())



# import asyncio
# import websockets
# import json

# async def chat_client():
#     uri = "ws://127.0.0.1:6789"
#     sender = input("Enter your name: ").strip() or "Anonymous"

#     async with websockets.connect(uri) as websocket:
#         # Send initial name to server
#         await websocket.send(json.dumps({"sender": sender, "text": ""}))

#         print(f"Connected to chat server as {sender}. Type messages. Ctrl+C to exit.")

#         # Task to receive messages from server
#         async def recv():
#             while True:
#                 try:
#                     msg = await websocket.recv()
#                     data = json.loads(msg)
#                     print(
#                         f"\n➡ {data.get('sender','Unknown')}: {data['text']}\n"
#                         f"   Emotion: {data['emotion']}\n"
#                         f"   Sarcasm: {data['sarcasm']}\nYou: ",
#                         end="",
#                         flush=True
#                     )
#                 except websockets.exceptions.ConnectionClosed:
#                     print("\nServer closed the connection.")
#                     break
#                 except Exception as e:
#                     print("\nReceive error:", e)
#                     break

#         recv_task = asyncio.create_task(recv())

#         try:
#             while True:
#                 text = await asyncio.to_thread(input, "You: ")
#                 text = text.strip()
#                 if not text:
#                     continue
#                 payload = json.dumps({"sender": sender, "text": text})
#                 await websocket.send(payload)
#         except KeyboardInterrupt:
#             print("\nExiting...")
#         finally:
#             recv_task.cancel()
#             try:
#                 await websocket.close()
#             except:
#                 pass

# if __name__ == "__main__":
#     asyncio.run(chat_client())




 