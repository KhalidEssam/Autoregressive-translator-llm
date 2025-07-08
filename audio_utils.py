import pyaudio
import numpy as np
import wave
import time
import os
from config import (  # Import constants explicitly
    RATE,
    CHUNK_SIZE,
    CHANNELS,
    AUDIO_FORMAT,
    SILENCE_THRESHOLD,  # Now correctly imported
    SILENCE_TIMEOUT,
    INPUT_AUDIO_PATH
)

def is_silent(audio_chunk: np.ndarray, threshold: int = SILENCE_THRESHOLD) -> bool:
    """Check if audio chunk is silent."""
    return np.max(np.abs(audio_chunk)) < threshold

def record_until_silence() -> str:
    """Record audio until silence timeout."""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=getattr(pyaudio, AUDIO_FORMAT),
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    frames = []
    last_sound_time = time.time()
    
    print("🎙️ Listening...")
    while True:
        chunk = np.frombuffer(stream.read(CHUNK_SIZE), dtype=np.int16)
        frames.append(chunk)
        
        if is_silent(chunk):
            if time.time() - last_sound_time > SILENCE_TIMEOUT:
                break
        else:
            last_sound_time = time.time()
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save to file
    with wave.open(INPUT_AUDIO_PATH, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(getattr(pyaudio, AUDIO_FORMAT)))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    return INPUT_AUDIO_PATH

def play_audio(file_path: str) -> None:
    """Play audio file cross-platform."""
    os.system(f"start {file_path}" if os.name == 'nt' else f"afplay {file_path}")



# record_until_silence()