"""
Voice Activity Detection (VAD) for automatic speech chunking
Uses webrtcvad for robust silence detection
"""

import webrtcvad
import struct
from collections import deque
import time


class VADProcessor:
    """
    Handles Voice Activity Detection and automatic audio chunking
    """
    
    def __init__(
        self, 
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        padding_duration_ms: int = 300,
        vad_aggressiveness: int = 3
    ):
        """
        Args:
            sample_rate: Audio sample rate (8000, 16000, 32000, or 48000)
            frame_duration_ms: Duration of each frame for VAD (10, 20, or 30 ms)
            padding_duration_ms: Duration of padding frames before/after speech
            vad_aggressiveness: VAD aggressiveness (0-3, 3 is most aggressive)
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        
        # Calculate frame size in bytes
        # For 16-bit audio: (sample_rate * frame_duration / 1000) * 2 bytes
        self.frame_size = int(sample_rate * frame_duration_ms / 1000) * 2
        
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        
        # Padding frames (silence before/after speech to include)
        self.num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        
        # Ring buffer for padding
        self.ring_buffer = deque(maxlen=self.num_padding_frames)
        
        # Speech state
        self.triggered = False
        self.voiced_frames = []
        
    def is_speech(self, frame: bytes) -> bool:
        """
        Check if a frame contains speech
        
        Args:
            frame: Raw PCM audio frame
            
        Returns:
            True if speech detected, False otherwise
        """
        try:
            return self.vad.is_speech(frame, self.sample_rate)
        except Exception as e:
            # If VAD fails, assume it's speech to be safe
            print(f"VAD error: {e}")
            return True
    
    def process_frame(self, frame: bytes) -> tuple[bool, bytes]:
        """
        Process a single audio frame and return accumulated speech if detected
        
        Args:
            frame: Raw PCM audio frame
            
        Returns:
            (is_speech_complete, accumulated_audio)
            - is_speech_complete: True when silence detected after speech
            - accumulated_audio: Bytes of the complete speech segment
        """
        is_speech = self.is_speech(frame)
        
        if not self.triggered:
            # Not currently in speech
            self.ring_buffer.append((frame, is_speech))
            num_voiced = sum(1 for f, speech in self.ring_buffer if speech)
            
            # If enough consecutive speech frames, trigger
            if num_voiced > 0.8 * self.ring_buffer.maxlen:
                self.triggered = True
                # Add buffered frames to voiced_frames
                self.voiced_frames.extend([f for f, s in self.ring_buffer])
                self.ring_buffer.clear()
        else:
            # Currently in speech
            self.voiced_frames.append(frame)
            self.ring_buffer.append((frame, is_speech))
            num_unvoiced = sum(1 for f, speech in self.ring_buffer if not speech)
            
            # If enough consecutive silence frames, end speech
            if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                # Speech segment complete
                accumulated = b''.join(self.voiced_frames)
                self.triggered = False
                self.voiced_frames = []
                self.ring_buffer.clear()
                return True, accumulated
        
        return False, b''
    
    def reset(self):
        """Reset the VAD state"""
        self.triggered = False
        self.voiced_frames = []
        self.ring_buffer.clear()


class StreamingVAD:
    """
    Simpler streaming VAD for WebSocket audio chunks
    Works with variable-length chunks
    """
    
    def __init__(
        self,
        silence_threshold: float = 1.5,  # seconds
        min_speech_duration: float = 0.3,  # seconds
    ):
        """
        Args:
            silence_threshold: Seconds of silence before considering speech ended
            min_speech_duration: Minimum duration to consider as valid speech
        """
        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration
        self.last_speech_time = None
        self.speech_start_time = None
        self.audio_buffer = []
        
    def is_silence(self, audio_chunk: bytes, sample_rate: int = 16000) -> bool:
        """
        Simple energy-based silence detection
        
        Args:
            audio_chunk: Raw PCM audio bytes
            sample_rate: Audio sample rate
            
        Returns:
            True if chunk is silence, False otherwise
        """
        if len(audio_chunk) < 2:
            return True
        
        try:
            # Convert bytes to samples (16-bit PCM)
            samples = struct.unpack(f"{len(audio_chunk)//2}h", audio_chunk)
            
            # Calculate RMS energy
            rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
            
            # Threshold for silence (tune this value)
            # Typical range: 50-500 for 16-bit audio
            SILENCE_THRESHOLD = 200
            
            return rms < SILENCE_THRESHOLD
            
        except Exception as e:
            print(f"Silence detection error: {e}")
            return False
    
    def add_chunk(self, audio_chunk: bytes) -> tuple[bool, bytes]:
        """
        Add audio chunk and check if we should process
        
        Args:
            audio_chunk: Raw audio bytes
            
        Returns:
            (should_process, accumulated_audio)
        """
        current_time = time.time()
        is_silence = self.is_silence(audio_chunk)
        
        if not is_silence:
            # Speech detected
            if self.speech_start_time is None:
                self.speech_start_time = current_time
            self.last_speech_time = current_time
            self.audio_buffer.append(audio_chunk)
            
        else:
            # Silence detected
            if self.last_speech_time is not None:
                silence_duration = current_time - self.last_speech_time
                
                # Check if silence threshold exceeded
                if silence_duration >= self.silence_threshold:
                    # Check if we have minimum speech duration
                    if self.speech_start_time is not None:
                        speech_duration = self.last_speech_time - self.speech_start_time
                        
                        if speech_duration >= self.min_speech_duration:
                            # Valid speech segment - process it
                            accumulated = b''.join(self.audio_buffer)
                            self.reset()
                            return True, accumulated
                    
                    # Speech too short, discard
                    self.reset()
        
        return False, b''
    
    def reset(self):
        """Reset the VAD state"""
        self.last_speech_time = None
        self.speech_start_time = None
        self.audio_buffer = []


# Example usage
if __name__ == "__main__":
    # Test the VAD
    vad = StreamingVAD(silence_threshold=1.0)
    
    # Simulate receiving audio chunks
    import random
    
    for i in range(100):
        # Simulate random audio chunk
        chunk_size = 1600  # 0.1 second at 16kHz, 16-bit
        # Random audio data
        chunk = bytes([random.randint(0, 255) for _ in range(chunk_size)])
        
        should_process, audio = vad.add_chunk(chunk)
        
        if should_process:
            print(f"Speech segment detected at iteration {i}")
            print(f"Total audio length: {len(audio)} bytes")