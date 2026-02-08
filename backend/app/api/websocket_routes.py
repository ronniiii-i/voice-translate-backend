from fastapi import WebSocket, WebSocketDisconnect
from app.services.translation_pipeline import TranslationPipeline
import json
import tempfile
import os
from pathlib import Path

async def websocket_translate(websocket: WebSocket):
    """
    WebSocket endpoint for streaming audio translation
    
    Protocol:
    1. Client sends JSON config: {"source_lang": "en", "target_lang": "fr"}
    2. Client sends audio chunks as binary data
    3. Client sends JSON end signal: {"action": "process"}
    4. Server processes and returns JSON: {"source_text": "...", "translated_text": "...", "audio_ready": true}
    5. Client requests audio: {"action": "get_audio"}
    6. Server streams back translated audio chunks
    """
    await websocket.accept()
    
    pipeline = TranslationPipeline()
    audio_buffer = bytearray()
    config = {"source_lang": "en", "target_lang": "fr"}
    output_audio_path = None
    
    try:
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                # JSON control message
                data = json.loads(message["text"])
                action = data.get("action")
                
                if action == "config":
                    # Update configuration
                    config["source_lang"] = data.get("source_lang", "en")
                    config["target_lang"] = data.get("target_lang", "fr")
                    await websocket.send_json({"status": "config_updated"})
                    
                elif action == "process":
                    # Process accumulated audio
                    if len(audio_buffer) == 0:
                        await websocket.send_json({"error": "No audio data received"})
                        continue
                    
                    # Save buffer to temp file
                    temp_input = tempfile.mktemp(suffix=".wav")
                    with open(temp_input, "wb") as f:
                        f.write(audio_buffer)
                    
                    try:
                        # Process through pipeline
                        output_audio_path, source_text, translated_text = pipeline.process_audio(
                            temp_input,
                            config["source_lang"],
                            config["target_lang"]
                        )
                        
                        # Send translation results
                        await websocket.send_json({
                            "source_text": source_text,
                            "translated_text": translated_text,
                            "audio_ready": True
                        })
                    finally:
                        os.unlink(temp_input)
                        audio_buffer.clear()
                    
                elif action == "get_audio":
                    # Stream translated audio back
                    if output_audio_path and os.path.exists(output_audio_path):
                        with open(output_audio_path, "rb") as f:
                            audio_data = f.read()
                        await websocket.send_bytes(audio_data)
                        
                        # Cleanup
                        os.unlink(output_audio_path)
                        output_audio_path = None
                    else:
                        await websocket.send_json({"error": "No audio available"})
                        
                elif action == "reset":
                    # Clear buffer and reset
                    audio_buffer.clear()
                    if output_audio_path and os.path.exists(output_audio_path):
                        os.unlink(output_audio_path)
                    output_audio_path = None
                    await websocket.send_json({"status": "reset_complete"})
                    
            elif "bytes" in message:
                # Audio chunk - accumulate in buffer
                audio_buffer.extend(message["bytes"])
                await websocket.send_json({
                    "status": "chunk_received",
                    "buffer_size": len(audio_buffer)
                })
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        # Cleanup on disconnect
        if output_audio_path and os.path.exists(output_audio_path):
            os.unlink(output_audio_path)