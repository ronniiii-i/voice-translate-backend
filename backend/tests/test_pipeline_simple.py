#!/usr/bin/env python3
"""Simplified pipeline test with better error handling"""
import time
from pathlib import Path
import sys

def test_asr_only():
    """Test ASR component only"""
    print("="*50)
    print("STEP 1: Testing ASR (Speech Recognition)")
    print("="*50)
    
    try:
        from app.models.asr_model import WhisperASR
        
        test_audio = Path(__file__).parent.parent.parent / "models/asr/whisper.cpp/samples/input.wav"
        print(f"Using audio file: {test_audio}")
        
        if not test_audio.exists():
            print(f"❌ Audio file not found: {test_audio}")
            return False
            
        asr = WhisperASR(model_size="tiny")
        print("✅ WhisperASR initialized")
        
        start = time.time()
        text = asr.transcribe(str(test_audio), language="en")
        elapsed = time.time() - start
        
        print(f"✅ Transcription completed in {elapsed:.2f}s")
        print(f"📝 Result: {text}")
        return True, text
        
    except Exception as e:
        print(f"❌ ASR Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_mt_only(source_text):
    """Test MT component only"""
    print("\n" + "="*50)
    print("STEP 2: Testing MT (Machine Translation)")
    print("="*50)
    
    try:
        from app.models.mt_model import ArgosTranslator
        
        translator = ArgosTranslator()
        print("✅ ArgosTranslator initialized")
        
        start = time.time()
        translated = translator.translate(source_text, "en", "fr")
        elapsed = time.time() - start
        
        print(f"✅ Translation completed in {elapsed:.2f}s")
        print(f"📝 English: {source_text}")
        print(f"📝 French: {translated}")
        return True, translated
        
    except ImportError as e:
        print(f"❌ MT Test Failed - Missing dependency: {e}")
        print("💡 Install with: pip install argostranslate")
        return False, None
    except Exception as e:
        print(f"❌ MT Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_tts_only(translated_text):
    """Test TTS component only"""
    print("\n" + "="*50)
    print("STEP 3: Testing TTS (Text-to-Speech)")
    print("="*50)
    
    try:
        from app.models.tts_model import PiperTTS
        import tempfile
        
        tts = PiperTTS()
        print("✅ PiperTTS initialized")
        
        output_path = tempfile.mktemp(suffix=".wav")
        
        start = time.time()
        result = tts.synthesize(translated_text, output_path, language="fr")
        elapsed = time.time() - start
        
        print(f"✅ TTS completed in {elapsed:.2f}s")
        print(f"📝 Output: {result}")
        return True, result
        
    except FileNotFoundError as e:
        print(f"❌ TTS Test Failed - Piper not found: {e}")
        print("💡 Install Piper TTS: https://github.com/rhasspy/piper")
        return False, None
    except Exception as e:
        print(f"❌ TTS Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_full_pipeline():
    """Test complete pipeline"""
    print("\n" + "="*50)
    print("FULL PIPELINE TEST")
    print("="*50)
    
    overall_start = time.time()
    
    # Step 1: ASR
    asr_result = test_asr_only()
    if not asr_result or not asr_result[0]:
        print("\n❌ Pipeline test stopped - ASR failed")
        return False
    
    source_text = asr_result[1]
    
    # Step 2: MT
    mt_result = test_mt_only(source_text)
    if not mt_result or not mt_result[0]:
        print("\n❌ Pipeline test stopped - MT failed")
        print("💡 You can still use ASR independently")
        return False
    
    translated_text = mt_result[1]
    
    # Step 3: TTS
    tts_result = test_tts_only(translated_text)
    if not tts_result or not tts_result[0]:
        print("\n❌ Pipeline test stopped - TTS failed")
        print("💡 You can still use ASR + MT")
        return False
    
    # Success!
    overall_elapsed = time.time() - overall_start
    
    print("\n" + "="*50)
    print("🎉 FULL PIPELINE SUCCESS!")
    print("="*50)
    print(f"⏱️  Total time: {overall_elapsed:.2f}s")
    print(f"📝 Source: {source_text}")
    print(f"📝 Translation: {translated_text}")
    print(f"🔊 Audio: {tts_result[1]}")
    
    return True

if __name__ == "__main__":
    print("Voice Translation Pipeline Test")
    print("Testing components individually before full pipeline...\n")
    
    success = test_full_pipeline()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some components failed - see errors above")
        sys.exit(1)
