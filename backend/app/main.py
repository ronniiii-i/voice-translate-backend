# backend/app/main.py
import os
import json
import uvicorn
import tempfile
import subprocess
import asyncio
import time
import wave
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from app.services.translation_pipeline import TranslationPipeline
from app.services.vad_processor import StreamingVAD
from fastapi.middleware.cors import CORSMiddleware


call_registry = {}
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("🚀 Initializing Translation Engine...")
    pipeline = TranslationPipeline()
    # Warm up common pairs
    try:
        pipeline.translator.translate("warmup", "en", "fr")
        pipeline.translator.translate("warmup", "fr", "en")
        print("✅ Models Warm. System Ready.")
    except: pass
    yield
    call_registry.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing, allow everything
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/call/{room_id}/{user_id}")
async def voice_bridge(websocket: WebSocket, room_id: str, user_id: str):
    await websocket.accept()
    
    # Initialize VAD for this specific user
    user_vad = StreamingVAD(silence_threshold=1.2, min_speech_duration=0.5)
    
    try:
        # 1. Handshake
        init_text = await websocket.receive_text()
        config = json.loads(init_text)
        
        call_registry[user_id] = {
            "ws": websocket,
            "native_lang": config.get("native_lang", "en"),
            "is_processing": False
        }
        
        print(f"📡 User {user_id} joined [{call_registry[user_id]['native_lang']}]")
        await websocket.send_json({"type": "connected", "user_id": user_id})

        # 2. Continuous Streaming Loop
        while True:
            # We use receive_bytes because the frontend is streaming audio blobs
            message = await websocket.receive()
            
            if "bytes" in message:
                audio_chunk = message["bytes"]
                
                # Use your StreamingVAD to decide when to process
                should_process, accumulated_audio = user_vad.add_chunk(audio_chunk)
                
                if should_process and not call_registry[user_id]["is_processing"]:
                    call_registry[user_id]["is_processing"] = True
                    
                    # Offload to background so the WebSocket stays open for more audio
                    asyncio.create_task(
                        process_and_send_audio(user_id, accumulated_audio, call_registry)
                    )
            
            elif "text" in message:
                # Handle control messages (like hangup) if needed
                pass

    except WebSocketDisconnect:
        print(f"❌ User {user_id} disconnected.")
    except Exception as e:
        print(f"⚠️ Socket Error: {e}")
    finally:
        if user_id in call_registry:
            del call_registry[user_id]

async def process_and_send_audio(user_id: str, audio_data: bytes, registry: dict):
    try:
        target_id = next((uid for uid in registry if uid != user_id), None)
        if not target_id: return

        source_lang = registry[user_id]["native_lang"]
        target_lang = registry[target_id]["native_lang"]

        # with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_raw:
        #     tmp_raw.write(audio_data)
        #     raw_path = tmp_raw.name
        
        # wav_path = raw_path.replace(".webm", ".wav")

        # # Transcode (Browser WebM -> Whisper WAV)
        # subprocess.run([
        #     "ffmpeg", "-y", "-loglevel", "error", "-i", raw_path,
        #     "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
        # ], check=True)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            with wave.open(tmp_wav, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2) # 16-bit
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_data)
            wav_path = tmp_wav.name

        # Pipeline
        output_audio, src_text, trans_text = pipeline.process_audio(
            wav_path, source_lang=source_lang, target_lang=target_lang
        )

        # Send to peer
        if os.path.exists(output_audio):
            with open(output_audio, "rb") as f:
                await registry[target_id]["ws"].send_bytes(f.read())
            
            await registry[target_id]["ws"].send_json({
                "type": "caption", "text": trans_text, "original": src_text
            })
            os.unlink(output_audio)

    except Exception as e:
        print(f"⚠️ Pipeline Error: {e}")
    finally:
        if user_id in registry:
            registry[user_id]["is_processing"] = False
        # Clean up temp files
        for p in [wav_path]:
            if os.path.exists(p): os.unlink(p)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)