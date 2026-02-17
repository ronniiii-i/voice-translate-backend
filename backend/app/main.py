# backend/app/main.py
"""
Fixed: 
1. Timeout increased to 25s (Whisper base needs ~12s)  
2. Queue per user — max 1 task at a time, stale audio is DROPPED not queued
3. WebSocket disconnect handled cleanly
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

call_registry = {}
pipeline = None

# Only 2 workers — your CPU showed n_threads=2/2 in Whisper output.
# More workers = more CPU contention = everything slower.
executor = ThreadPoolExecutor(max_workers=2)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("🚀 Initializing Translation Engine...")
    pipeline = TranslationPipeline()

    print("⏳ Pre-warming translation models...")
    for src, tgt in [("en", "fr"), ("fr", "en"), ("en", "es"), ("es", "en"), ("en", "de"), ("de", "en")]:
        try:
            pipeline.translator.translate("warmup", src, tgt)
            print(f"  ✅ {src}→{tgt}")
        except Exception as e:
            print(f"  ⚠️ {src}→{tgt} failed: {e}")

    print("✅ System Ready!")
    yield
    call_registry.clear()
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
    return {"status": "ok", "active_users": len(call_registry)}


class UserSession:
    """
    One session per user.

    KEY FIX: Only ONE pipeline task runs at a time per user.
    If a new speech segment arrives while already processing,
    the NEW segment is DROPPED — not queued.

    This prevents the cascade of 20+ stacked tasks that was
    crashing connections and flooding the system.
    """

    def __init__(self, user_id: str, ws: WebSocket, lang: str):
        self.user_id = user_id
        self.ws = ws
        self.lang = lang
        self.vad = StreamingVAD(silence_threshold=1.2, min_speech_duration=0.5)
        self._busy = False
        self._lock = threading.Lock()
        self.connected = True

    async def handle_chunk(self, chunk: bytes, target: "UserSession"):
        should_process, audio = self.vad.add_chunk(chunk)
        if not should_process:
            return

        # Drop if already processing — never queue multiple tasks!
        with self._lock:
            if self._busy:
                duration = len(audio) / (16000 * 2)
                print(f"[{self.user_id}] Busy, dropping {duration:.1f}s segment")
                return
            self._busy = True

        asyncio.create_task(self._run_pipeline(audio, target))

    async def _run_pipeline(self, audio: bytes, target: "UserSession"):
        try:
            duration = len(audio) / (16000 * 2)
            print(f"[{self.user_id}] Processing {duration:.1f}s for {target.user_id}")

            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(executor, self._sync_pipeline, audio, target.lang),
                timeout=30.0  # ASR(~5s) + MT(~0.5s) + TTS(~1s) + headroom
            )

            if result is None:
                return

            out_path, src_text, trans_text = result

            if not src_text or src_text.startswith("["):
                print(f"[{self.user_id}] No usable speech detected")
                return

            if not target.connected:
                print(f"[{self.user_id}] Target disconnected, discarding")
                if out_path and os.path.exists(out_path):
                    os.unlink(out_path)
                return

            if out_path and os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    audio_bytes = f.read()
                os.unlink(out_path)
                await target.ws.send_bytes(audio_bytes)
                await target.ws.send_json({
                    "type": "caption",
                    "text": trans_text,
                    "original": src_text
                })
                print(f'✅ {self.user_id}→{target.user_id}: "{src_text[:60]}"')

        except asyncio.TimeoutError:
            print(f"❌ [{self.user_id}] Pipeline timeout >30s")

        except Exception as e:
            msg = str(e)
            if "websocket.send" in msg or "disconnect" in msg.lower() or "close" in msg.lower():
                print(f"[{self.user_id}] Target disconnected mid-send")
            else:
                print(f"❌ [{self.user_id}] Pipeline error: {e}")

        finally:
            with self._lock:
                self._busy = False

    def _sync_pipeline(self, audio_data: bytes, target_lang: str):
        """Synchronous pipeline — runs in thread pool."""
        wav_path = None
        try:
            wav_path = tempfile.mktemp(suffix=".wav")
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data)

            return pipeline.process_audio(wav_path, self.lang, target_lang)

        except Exception as e:
            print(f"[Pipeline] Sync error: {e}")
            return None

        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except:
                    pass


@app.websocket("/ws/call/{room_id}/{user_id}")
async def voice_bridge(websocket: WebSocket, room_id: str, user_id: str):
    await websocket.accept()
    session = None

    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        config = json.loads(raw)
        lang = config.get("native_lang", "en")

        session = UserSession(user_id, websocket, lang)
        call_registry[user_id] = session

        print(f"📡 {user_id} joined room '{room_id}' [{lang}]")
        await websocket.send_json({"type": "connected", "user_id": user_id})

        while True:
            try:
                msg = await websocket.receive()
            except WebSocketDisconnect:
                break

            if "bytes" in msg:
                peer = next(
                    (s for uid, s in call_registry.items() if uid != user_id),
                    None
                )
                if peer:
                    await session.handle_chunk(msg["bytes"], peer)

            elif "text" in msg:
                try:
                    data = json.loads(msg["text"])
                    if data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except:
                    pass

    except asyncio.TimeoutError:
        print(f"⏰ {user_id} handshake timed out")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        if "receive" not in str(e).lower():
            print(f"⚠️ {user_id}: {e}")
    finally:
        if session:
            session.connected = False
        call_registry.pop(user_id, None)
        print(f"🔌 {user_id} left")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")