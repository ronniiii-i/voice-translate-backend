import os
import json
import uvicorn
import tempfile
import subprocess
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from app.services.translation_pipeline import TranslationPipeline

# Structure: { "user_id": {"ws": websocket, "native_lang": "de"} }
call_registry = {}
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("🚀 Initializing Translation Engine...")
    print("💡 Loading models into RAM (this takes ~60-90s on 8GB RAM)...")
    
    pipeline = TranslationPipeline()
    
    try:
        pipeline.translator.translate("warmup", "en", "fr")
        print("✅ Models Warm. System Ready on Port 8000.")
    except Exception as e:
        print(f"⚠️ Warmup warning: {e}")
        
    yield
    call_registry.clear()

app = FastAPI(lifespan=lifespan)

@app.websocket("/ws/call/{room_id}/{user_id}")
async def voice_bridge(websocket: WebSocket, room_id: str, user_id: str):
    await websocket.accept()
    
    try:
        init_data = await websocket.receive_text()
        config = json.loads(init_data)
        
        call_registry[user_id] = {
            "ws": websocket,
            "native_lang": config.get("native_lang", "en")
        }
        print(f"📡 User {user_id} joined. Native Language: {call_registry[user_id]['native_lang']}")
        
        while True:
            audio_in = await websocket.receive_bytes()
            
            target_id = next((uid for uid in call_registry if uid != user_id), None)
            
            if target_id:
                source_lang = call_registry[user_id]["native_lang"]
                target_lang = call_registry[target_id]["native_lang"]
                
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_raw:
                    tmp_raw.write(audio_in)
                    raw_path = tmp_raw.name
                
                wav_path = raw_path.replace(".webm", ".wav")
                
                try:
                    transcode_cmd = [
                        "ffmpeg", "-y", "-i", raw_path,
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path
                    ]
                    subprocess.run(transcode_cmd, capture_output=True, check=True)

                    
                    print(f"🔄 {user_id}({source_lang}) -> {target_id}({target_lang})")
                    output_audio, src_text, trans_text = pipeline.process_audio(
                        wav_path, 
                        source_lang=source_lang, 
                        target_lang=target_lang
                    )
                    
                    
                    if os.path.exists(output_audio):
                        with open(output_audio, "rb") as f:
                            # Send translated voice to Target
                            await call_registry[target_id]["ws"].send_bytes(f.read())
                        
                        # Send captions to Target
                        await call_registry[target_id]["ws"].send_json({
                            "type": "caption",
                            "text": trans_text,
                            "original": src_text
                        })
                        
                        os.unlink(output_audio)

                except subprocess.CalledProcessError as e:
                    print(f"❌ Transcoding Error: {e.stderr.decode() if e.stderr else str(e)}")
                except Exception as e:
                    print(f"⚠️ Pipeline Error: {str(e)}")
                finally:
                    for p in [raw_path, wav_path]:
                        if os.path.exists(p):
                            os.unlink(p)
            else:
                await websocket.send_json({"info": "Waiting for a peer to join the room..."})
                    
    except WebSocketDisconnect:
        if user_id in call_registry:
            del call_registry[user_id]
        print(f"❌ User {user_id} left.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)