# backend/app/main.py
"""
Voice Call Translation Server

Key design decisions:
- Max 2 users per room (enforced at join time)
- One pipeline task per user at a time — excess segments DROPPED, never queued
- faster-whisper replaces whisper.cpp (no subprocess/file race conditions)
- 2 ThreadPoolExecutor workers (matches your CPU thread count)
"""

import os
import json
import uvicorn
import tempfile
import asyncio
import wave
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from app.services.translation_pipeline import TranslationPipeline
from app.services.vad_processor import StreamingVAD
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
import threading

# room_id -> {user_id -> UserSession}
rooms: dict[str, dict] = {}
pipeline: TranslationPipeline = None

# 2 workers = matches your i5's physical cores exposed to Whisper
executor = ThreadPoolExecutor(max_workers=2)

MAX_USERS_PER_ROOM = 2


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("🚀 Initializing Translation Engine...")
    pipeline = TranslationPipeline()

    print("⏳ Pre-warming translation cache...")
    pairs = [("en", "fr"), ("fr", "en"), ("en", "es"), ("es", "en"), ("en", "de"), ("de", "en")]
    for src, tgt in pairs:
        try:
            pipeline.translator.translate("warmup", src, tgt)
            print(f"  ✅ {src}→{tgt}")
        except Exception as e:
            print(f"  ⚠️  {src}→{tgt}: {e}")

    print("✅ System Ready!")
    yield

    rooms.clear()
    executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    room_info = {rid: list(users.keys()) for rid, users in rooms.items()}
    return {"status": "ok", "rooms": room_info}


class UserSession:
    """
    Manages one user's audio pipeline within a room.
    Only ONE pipeline task runs at a time — stale audio is dropped.
    """

    def __init__(self, user_id: str, ws: WebSocket, lang: str):
        self.user_id = user_id
        self.ws = ws
        self.lang = lang
        self.vad = StreamingVAD(
            silence_threshold=1.2,
            min_speech_duration=0.4,
            energy_threshold=400.0,   # lower = more sensitive; tune via logs
            max_speech_duration=8.0,
        )
        self._busy = False
        self._lock = threading.Lock()
        self.connected = True

    async def handle_chunk(self, chunk: bytes, target: "UserSession"):
        """Called for every incoming PCM chunk from this user."""
        should_process, audio = self.vad.add_chunk(chunk)
        if not should_process:
            return

        # Drop if already processing — prevents cascading queue buildup
        with self._lock:
            if self._busy:
                duration_s = len(audio) / (16000 * 2)
                print(f"[{self.user_id}] ⚠️  Busy — dropping {duration_s:.1f}s segment")
                return
            self._busy = True

        asyncio.create_task(self._run_pipeline(audio, target))

    async def _run_pipeline(self, audio: bytes, target: "UserSession"):
        try:
            duration_s = len(audio) / (16000 * 2)
            print(f"[{self.user_id}] 🔄 Processing {duration_s:.1f}s → {target.user_id}")

            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    self._sync_pipeline,
                    audio,
                    target.lang,
                ),
                timeout=35.0,  # ASR(~5s) + MT(<1s) + TTS(~2s) + headroom
            )

            if result is None:
                return

            out_path, src_text, trans_text = result

            # Sentinels from ASR (silence, hallucination, error, etc.)
            if not src_text or src_text.startswith("["):
                print(f"[{self.user_id}] No usable speech: {src_text}")
                return

            if not target.connected:
                print(f"[{self.user_id}] Target disconnected — discarding result")
                if out_path and os.path.exists(out_path):
                    os.unlink(out_path)
                return

            # Send translated audio first (lower latency perception)
            if out_path and os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    audio_bytes = f.read()
                os.unlink(out_path)
                await target.ws.send_bytes(audio_bytes)

            # Then send captions
            await target.ws.send_json({
                "type": "caption",
                "text": trans_text,
                "original": src_text,
            })
            print(f'✅ {self.user_id}→{target.user_id}: "{src_text[:60]}" → "{trans_text[:60]}"')

        except asyncio.TimeoutError:
            print(f"❌ [{self.user_id}] Pipeline timeout (>35s)")

        except Exception as e:
            err = str(e)
            if any(k in err.lower() for k in ("websocket", "disconnect", "close", "send")):
                print(f"[{self.user_id}] Target disconnected mid-send")
            else:
                print(f"❌ [{self.user_id}] Pipeline error: {e}")

        finally:
            with self._lock:
                self._busy = False

    def _sync_pipeline(self, audio_data: bytes, target_lang: str):
        """
        Runs in ThreadPoolExecutor.
        Writes raw PCM to a proper WAV file, then runs ASR → MT → TTS.
        """
        wav_path = None
        try:
            # Write PCM bytes as a valid WAV file so Whisper can read it
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)       # mono
                wf.setsampwidth(2)       # 16-bit = 2 bytes
                wf.setframerate(16000)   # 16kHz — must match browser AudioContext
                wf.writeframes(audio_data)

            file_size = os.path.getsize(wav_path)
            print(f"[Pipeline] WAV written: {file_size} bytes, {len(audio_data)//32000:.1f}s")

            return pipeline.process_audio(wav_path, self.lang, target_lang)

        except Exception as e:
            print(f"[Pipeline] Sync error: {e}")
            return None

        finally:
            # Always clean up the WAV file (TTS output is cleaned in _run_pipeline)
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass


@app.websocket("/ws/call/{room_id}/{user_id}")
async def voice_bridge(websocket: WebSocket, room_id: str, user_id: str):
    await websocket.accept()
    session = None

    try:
        # ── Room capacity check ──────────────────────────────────────────────
        room = rooms.setdefault(room_id, {})
        if len(room) >= MAX_USERS_PER_ROOM:
            await websocket.send_json({
                "type": "error",
                "message": f"Room '{room_id}' is full (max {MAX_USERS_PER_ROOM} users).",
            })
            await websocket.close(code=4003)
            return

        # ── Handshake ────────────────────────────────────────────────────────
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
            config = json.loads(raw)
        except asyncio.TimeoutError:
            await websocket.send_json({"type": "error", "message": "Handshake timed out"})
            return
        except json.JSONDecodeError:
            await websocket.send_json({"type": "error", "message": "Invalid JSON in handshake"})
            return

        lang = config.get("native_lang", "en")
        session = UserSession(user_id, websocket, lang)
        room[user_id] = session

        print(f"📡 {user_id} joined room '{room_id}' [{lang}] ({len(room)}/{MAX_USERS_PER_ROOM})")
        await websocket.send_json({"type": "connected", "user_id": user_id, "room": room_id})

        # Notify peer if they're already in the room
        peer = _get_peer(room_id, user_id)
        if peer:
            try:
                await peer.ws.send_json({"type": "peer_joined", "peer_id": user_id})
            except Exception:
                pass

        # ── Main receive loop ────────────────────────────────────────────────
        while True:
            try:
                msg = await websocket.receive()
            except WebSocketDisconnect:
                break

            if "bytes" in msg:
                peer = _get_peer(room_id, user_id)
                if peer:
                    await session.handle_chunk(msg["bytes"], peer)
                # If no peer yet, silently discard — VAD would buffer needlessly

            elif "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except Exception:
                    pass

    except asyncio.TimeoutError:
        print(f"⏰ {user_id} handshake timed out")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        if "receive" not in str(e).lower():
            print(f"⚠️  {user_id}: {e}")
    finally:
        if session:
            session.connected = False

        # Clean up room entry
        room = rooms.get(room_id, {})
        room.pop(user_id, None)
        if not room:
            rooms.pop(room_id, None)

        # Notify peer of disconnection
        peer = _get_peer(room_id, user_id)
        if peer:
            try:
                await peer.ws.send_json({"type": "peer_left", "peer_id": user_id})
            except Exception:
                pass

        print(f"🔌 {user_id} left room '{room_id}'")


def _get_peer(room_id: str, exclude_user_id: str):
    """Return the other user in the room, or None."""
    room = rooms.get(room_id, {})
    for uid, session in room.items():
        if uid != exclude_user_id and session.connected:
            return session
    return None


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False,   # Don't use reload=True — it re-initializes the model
    )