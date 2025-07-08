# config.py
# Audio Recording
RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1
AUDIO_FORMAT = "paInt16"
SILENCE_THRESHOLD = 200
SILENCE_TIMEOUT = 3  # seconds

# File Paths
INPUT_AUDIO_PATH = "input.wav"
RESPONSE_AUDIO_PATH = "response.wav"
TTS_OUTPUT_DIR = "tts_output"

# Models
WHISPER_MODEL = "base"
TTS_MODEL = "tts_models/multilingual/multi-dataset/vits"