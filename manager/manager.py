import os
import time
import functools
import asyncio
import json
import base64
import traceback
import cv2
import numpy as np
import asyncpg
import websockets
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# ------------------------ Load .env
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

# ------------------------ Config
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "60"))
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
MODEL_WS_URL = os.getenv("MODEL_WS_URL", "ws://127.0.0.1:8001/ws")

# ------------------------ FastAPI App
app = FastAPI(title="Manager Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

last_received_time = time.time()
frame_counter = 0
db_pool: asyncpg.pool.Pool = None
clients = []

# ------------------------ Frontend
frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')
app.mount("/frontend", StaticFiles(directory=frontend_path), name="frontend")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))

# ------------------------ Persistent WS to model server
model_ws = None

async def connect_model_ws():
    global model_ws
    while True:
        try:
            print(f"[MANAGER] 🔌 Connecting to model WS at {MODEL_WS_URL}")
            model_ws = await websockets.connect(MODEL_WS_URL, max_size=2**24)
            print("[MANAGER] ✅ Connected to model WS")
            return
        except Exception as e:
            print(f"[MANAGER] ⚠️ Model WS connection failed: {e}")
            await asyncio.sleep(3)

async def ensure_model_ws():
    global model_ws
    if not model_ws or model_ws.state != websockets.State.OPEN:
        await connect_model_ws()
        return

# ------------------------ Annotate frame
def annotate_frame(frame_bytes: bytes, result: dict) -> bytes:
    """
    CPU-bound function to annotate a frame.
    """
    nparr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if not result.get("persons"):
        _, buffer = cv2.imencode(".jpg", img)
        return buffer.tobytes()
    for pid, data in result["persons"].items():
        keypoints = data.get("keypoints", [])
        action = data.get("action", "")
        for x, y in keypoints:
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
        if keypoints:
            x, y = keypoints[0]
            cv2.putText(img, action, (int(x), int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    _, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()

# ------------------------ Save to DB
async def save_to_db(timestamp: float, result: dict):
    global db_pool
    if not result.get("persons"):
        return
    async with db_pool.acquire() as conn:
        for pid, data in result["persons"].items():
            action = data.get("action", "")
            keypoints = data.get("keypoints", [])
            await conn.execute(
                """
                INSERT INTO results (timestamp, person_id, action, keypoints)
                VALUES ($1,$2,$3,$4)
                """,
                timestamp, int(pid), action, json.dumps(keypoints)
            )

# ------------------------ Broadcast to frontend
async def broadcast(result: dict, frame_bytes: bytes = None):
    loop = asyncio.get_running_loop()
    annotated_bytes = None
    if frame_bytes:
        # Run blocking cv2 code in a thread pool
        annotated_bytes = await loop.run_in_executor(
            None, functools.partial(annotate_frame, frame_bytes, result)
        )

    payload = {"result": result}
    if annotated_bytes:
        payload["frame"] = base64.b64encode(annotated_bytes).decode("utf-8")

    for ws in clients:
        try:
            await ws.send_json(payload)
        except:
            pass

# ------------------------ Frontend WS
@app.websocket("/ws")
async def frontend_ws(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    print("[MANAGER] 🌐 Frontend connected")
    try:
        while True:
            await asyncio.sleep(1)
    except:
        pass
    finally:
        clients.remove(websocket)
        print("[MANAGER] ⚠️ Frontend disconnected")

# ------------------------ Sender → Manager WS
@app.websocket("/ws_stream")
async def ws_stream_endpoint(websocket: WebSocket):
    global model_ws, frame_counter, last_received_time
    await websocket.accept()
    print("[MANAGER] 🎥 Sender connected")

    try:
        await ensure_model_ws()

        while True:
            frame_bytes = await websocket.receive_bytes()
            last_received_time = time.time()
            frame_counter += 1
            start_time = time.time()

            try:
                # Send frame to model and get result
                await model_ws.send(frame_bytes)
                result_json = await model_ws.recv()
            except websockets.exceptions.ConnectionClosed:
                print("[MANAGER] ⚠️ Model WS connection lost. Reconnecting...")
                await connect_model_ws()
                # Retry sending the frame
                await model_ws.send(frame_bytes)
                result_json = await model_ws.recv()

            result = json.loads(result_json)

            # --- Run post-processing tasks concurrently ---
            # These tasks don't need to block the main loop from receiving the next frame.
            asyncio.create_task(save_to_db(time.time(), result))
            asyncio.create_task(broadcast(result, frame_bytes))

            print(f"[MANAGER] Frame {frame_counter} → {(time.time()-start_time)*1000:.2f} ms")

    except Exception as e:
        # This will catch errors from the sender's connection
        print(f"[MANAGER] ❌ Stream error: {e}")
        traceback.print_exc()
    finally:
        print("[MANAGER] ⚠️ Sender disconnected")

# ------------------------ /results endpoint
@app.get("/results")
async def get_results(limit: int = 50):
    global db_pool
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT timestamp, person_id, action, keypoints FROM results ORDER BY timestamp DESC LIMIT $1", limit
        )
        return JSONResponse(content=[dict(row) for row in rows])

# ------------------------ Session monitor
async def session_monitor():
    global last_received_time, frame_counter
    while True:
        await asyncio.sleep(5)
        if frame_counter > 0 and time.time() - last_received_time > SESSION_TIMEOUT:
            print(f"[MANAGER] ⌛ Session timeout after {SESSION_TIMEOUT}s of inactivity. Clearing state.")
            # Reset state
            frame_counter = 0
            last_received_time = time.time()

# ------------------------ Startup
@app.on_event("startup")
async def startup_event():
    global db_pool
    db_pool = await asyncpg.create_pool(
        host=DB_HOST, port=DB_PORT, user=DB_USER,
        password=DB_PASSWORD, database=DB_NAME,
        ssl=None, min_size=1, max_size=5
    )
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id SERIAL PRIMARY KEY,
                timestamp DOUBLE PRECISION,
                person_id INT,
                action TEXT,
                keypoints JSONB
            )
            """
        )
    print("[MANAGER] ✅ Ready and DB connected")
    global last_received_time
    last_received_time = time.time()
    asyncio.create_task(session_monitor())
    asyncio.create_task(connect_model_ws())
