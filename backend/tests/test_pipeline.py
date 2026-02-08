import time
from pathlib import Path
from app.services.translation_pipeline import TranslationPipeline

def test_full_pipeline():
    pipeline = TranslationPipeline()
    # Use available test audio from whisper.cpp samples
    test_audio = Path(__file__).parent.parent.parent / "models/asr/whisper.cpp/samples/input.wav"
    
    print(f"Testing with audio file: {test_audio}")
    
    start = time.time()
    output, src_text, trans_text = pipeline.process_audio(str(test_audio), "en", "fr")
    elapsed = time.time() - start
    
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Source: {src_text}")
    print(f"Translation: {trans_text}")
    print(f"Output audio: {output}")
    
    # Target: < 3x real-time for 5s audio
    assert elapsed < 15, f"Too slow: {elapsed}s for 5s audio"
    
if __name__ == "__main__":
    test_full_pipeline()