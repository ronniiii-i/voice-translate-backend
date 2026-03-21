from logging import config
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

executor = ThreadPoolExecutor(max_workers=4)

MAX_USERS_PER_ROOM = 2

HISTORY_MAX = 6


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("🚀 Initializing Translation Engine...")
    pipeline = TranslationPipeline()

    # Warm up ASR only — MT models load lazily per session
    print("⏳ Pre-warming ASR...")
    try:
        import tempfile, wave, numpy as np
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        silence = bytes(16000 * 2)  # 1s of silence
        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(silence)
        pipeline.asr.transcribe(tmp, language="en")
        os.unlink(tmp)
        print("  ✅ ASR warmed")
    except Exception as e:
        print(f"  ⚠️  ASR warmup skipped: {e}")

    print("✅ System Ready!")
    yield

    rooms.clear()
    executor.shutdown(wait=False)
    if pipeline:
        pipeline.tts.shutdown()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": "LinguaCall Backend",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "websocket": "wss://<host>/ws/call/{room_id}/{user_id}",
    }


@app.get("/health")
async def health():
    room_info = {rid: list(users.keys()) for rid, users in rooms.items()}
    return {"status": "ok", "rooms": room_info}


@app.get("/rooms/{room_id}")
async def check_room(room_id: str):
    exists = room_id in rooms
    count = len(rooms[room_id]) if exists else 0
    return {"exists": exists, "room_id": room_id, "occupants": count}


# ── Pre-load helper (runs in executor) ───────────────────────────────────────

def _preload_mt_pairs(lang_a: str, lang_b: str) -> None:
    try:
        pipeline.translator.ensure_pair_loaded(lang_a, lang_b)
        pipeline.translator.ensure_pair_loaded(lang_b, lang_a)
        print(f"[MT] Session models ready: {lang_a}↔{lang_b}")
    except Exception as e:
        print(f"[MT] Preload warning: {e}")


# ── UserSession ───────────────────────────────────────────────────────────────

class UserSession:
    def __init__(self, user_id: str, ws: WebSocket, lang: str):
        self.user_id = user_id
        self.ws = ws
        self.lang = lang
        self.display_name = user_id
        self.vad = StreamingVAD(
            silence_threshold=1.2,
            min_speech_duration=0.8,
            energy_threshold=800.0,
            max_speech_duration=15.0,
        )
        self._busy = False
        self._lock = threading.Lock()
        self.connected = True

        self._history: list[str] = []
        self._history_lock = threading.Lock()

    def add_to_history(self, utterance: str) -> None:
        """Append a transcribed utterance to this user's conversation history."""
        with self._history_lock:
            self._history.append(utterance)
            if len(self._history) > HISTORY_MAX:
                self._history.pop(0)

    def get_history(self) -> list[str]:
        """Return a snapshot of the current conversation history."""
        with self._history_lock:
            return list(self._history)

    def clear_history(self) -> None:
        """Reset history on disconnect or call end."""
        with self._history_lock:
            self._history.clear()

    async def handle_chunk(self, chunk: bytes, target: "UserSession") -> None:
        """Called for every incoming PCM chunk from this user."""
        should_process, audio = self.vad.add_chunk(chunk)
        if not should_process:
            return

        with self._lock:
            if self._busy:
                duration_s = len(audio) / (16000 * 2)
                print(f"[{self.user_id}] ⚠️  Busy — dropping {duration_s:.1f}s segment")
                return
            self._busy = True

        asyncio.create_task(self._run_pipeline(audio, target))

    async def _run_pipeline(self, audio: bytes, target: "UserSession") -> None:
        try:
            duration_s = len(audio) / (16000 * 2)
            print(f"[{self.user_id}] 🔄 Processing {duration_s:.1f}s → {target.user_id}")

            context_snapshot = self.get_history()

            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    self._sync_pipeline,
                    audio,
                    target.lang,
                    context_snapshot,
                ),
                timeout=300.0,
            )

            if result is None:
                return

            out_path, src_text, trans_text = result

            if not src_text or src_text.startswith("["):
                print(f"[{self.user_id}] No usable speech: {src_text}")
                return

            self.add_to_history(src_text)

            if self.connected:
                try:
                    await self.ws.send_json({
                        "type": "self_caption",
                        "text": src_text,
                    })
                except Exception:
                    pass

            if not target.connected:
                print(f"[{self.user_id}] Target disconnected — discarding result")
                if out_path and os.path.exists(out_path):
                    os.unlink(out_path)
                return

            if out_path and os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    audio_bytes = f.read()
                os.unlink(out_path)

                import struct as _struct
                meta = json.dumps({
                    "type": "audio_with_caption",
                    "text": trans_text,
                    "original": src_text,
                }).encode("utf-8")
                framed = _struct.pack(">I", len(meta)) + meta + audio_bytes
                await target.ws.send_bytes(framed)
            else:
                # TTS failed — send text caption only
                await target.ws.send_json({
                    "type": "caption",
                    "text": trans_text,
                    "original": src_text,
                })

            print(f'✅ {self.user_id}→{target.user_id}: '
                  f'"{src_text[:60]}" → "{trans_text[:60]}"')

        except asyncio.TimeoutError:
            print(f"❌ [{self.user_id}] Pipeline timeout (>300s)")

        except Exception as e:
            err = str(e)
            if any(k in err.lower() for k in ("websocket", "disconnect", "close", "send")):
                print(f"[{self.user_id}] Target disconnected mid-send")
            else:
                print(f"❌ [{self.user_id}] Pipeline error: {e}")

        finally:
            with self._lock:
                self._busy = False

    def _sync_pipeline(
        self,
        audio_data: bytes,
        target_lang: str,
        context: list[str],
    ):
        """
        Runs in ThreadPoolExecutor.
        Writes raw PCM to a WAV file, then runs ASR → context-aware MT → TTS.
        """
        wav_path = None
        try:
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data)

            file_size = os.path.getsize(wav_path)
            print(f"[Pipeline] WAV written: {file_size} bytes, "
                  f"{len(audio_data) // 32000:.1f}s")

            return pipeline.process_audio(
                wav_path,
                self.lang,
                target_lang,
                context=context,
            )

        except Exception as e:
            print(f"[Pipeline] Sync error: {e}")
            return None

        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass


# ── WebSocket endpoint ────────────────────────────────────────────────────────

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
            cfg = json.loads(raw)
        except asyncio.TimeoutError:
            await websocket.send_json({"type": "error", "message": "Handshake timed out"})
            return
        except json.JSONDecodeError:
            await websocket.send_json({"type": "error", "message": "Invalid JSON in handshake"})
            return

        lang = cfg.get("native_lang", "en")
        display_name = cfg.get("display_name", user_id)

        session = UserSession(user_id, websocket, lang)
        session.display_name = display_name
        room[user_id] = session

        print(f"📡 {user_id} joined room '{room_id}' [{lang}] ({len(room)}/{MAX_USERS_PER_ROOM})")
        await websocket.send_json({"type": "connected", "user_id": user_id, "room": room_id})

        # ── Peer notification + MT pre-load ──────────────────────────────────
        peer = _get_peer(room_id, user_id)
        if peer:
            # Tell existing peer that someone joined
            try:
                await peer.ws.send_json({
                    "type": "peer_joined",
                    "peer_id": user_id,
                    "display_name": display_name,
                })
                session.vad._reset()
            except Exception:
                pass

            # Tell joining user that a peer is already here
            try:
                await websocket.send_json({
                    "type": "peer_joined",
                    "peer_id": peer.user_id,
                    "display_name": getattr(peer, "display_name", peer.user_id),
                })
                session.vad._reset()
            except Exception:
                pass

            # Pre-load MT models for this session's language pair in background.
            # We await this so models are ready before audio starts flowing.
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                executor,
                _preload_mt_pairs,
                session.lang,
                peer.lang,
            )

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
            session.clear_history()

        room = rooms.get(room_id, {})
        room.pop(user_id, None)
        if not room:
            rooms.pop(room_id, None)

        peer = _get_peer(room_id, user_id)
        if peer:
            try:
                await peer.ws.send_json({
                    "type": "peer_left",
                    "peer_id": user_id,
                    "display_name": getattr(session, "display_name", user_id),
                })
            except Exception:
                pass

        print(f"🔌 {user_id} left room '{room_id}'")


def _get_peer(room_id: str, exclude_user_id: str) -> UserSession | None:
    """Return the other connected user in the room, or None."""
    room = rooms.get(room_id, {})
    for uid, sess in room.items():
        if uid != exclude_user_id and sess.connected:
            return sess
    return None


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        log_level="info",
        reload=False,
    )