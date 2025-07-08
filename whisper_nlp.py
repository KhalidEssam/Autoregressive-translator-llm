import whisper
from config import WHISPER_MODEL  # Import the constant from config

class WhisperTranscriber:
    def __init__(self, model_name: str = WHISPER_MODEL):  # Now WHISPER_MODEL is defined
        self.model = whisper.load_model(model_name)
    
    def transcribe(self, audio_path: str) -> str:
        """Convert speech to text."""
        result = self.model.transcribe(audio_path)
        return result["text"]
    
# transcribe = WhisperTranscriber()

# result = transcribe.transcribe("input.wav")
# print(f"Transcription: {result}")
