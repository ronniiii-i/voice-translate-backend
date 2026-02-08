import time
import subprocess
import os
from pathlib import Path
from app.models.asr_model import WhisperASR

def get_audio_duration(file_path):
    """Uses ffprobe to get duration of the wav file"""
    cmd = [
        "ffprobe", "-i", file_path, "-show_entries", "format=duration",
        "-v", "quiet", "-of", "csv=p=0"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())

def run_test():
    # 1. Setup paths
    test_file = Path(__file__).parent.parent.parent / "models/asr/whisper.cpp/samples/input.wav"
    if not test_file.exists():
        print(f"❌ Error: {test_file} not found. Run the ffmpeg command first!")
        return

    # 2. Initialize Model
    print("🚀 Loading Whisper Tiny...")
    asr = WhisperASR(model_size="tiny")
    
    # 3. Transcribe and Benchmark
    duration = get_audio_duration(str(test_file))
    print(f"Watch out! Processing {duration:.2f}s of audio...")
    
    start_time = time.time()
    text = asr.transcribe(str(test_file))
    end_time = time.time()
    
    elapsed = end_time - start_time
    rtf = elapsed / duration

    # 4. Results
    print("\n" + "="*30)
    print(f"RESULT: {text}")
    print("="*30)
    print(f"Processing Time: {elapsed:.2f}s")
    print(f"Real-time Factor: {rtf:.2f}x (Lower is better)")
    
    if rtf < 1.0:
        print("✅ SUCCESS: Your CPU is fast enough for real-time!")
    else:
        print("⚠️ WARNING: Processing is slower than speech. Consider 'tiny.en' model.")

if __name__ == "__main__":
    run_test()