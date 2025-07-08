from TTS.api import TTS
import os
from config import TTS_MODEL, TTS_OUTPUT_DIR

class CoquiTTS:
    def __init__(self, model_name: str = TTS_MODEL):
        try:
            # Initialize TTS with minimal parameters
            self.tts = TTS(model_name)
            print(f"Successfully loaded TTS model: {model_name}")
        except Exception as e:
            print(f"Error details: {str(e)}")
            raise RuntimeError(f"TTS init failed: {e}")

    def speak(self, text: str, output_path: str, language: str = "en") -> None:
        """Convert text to speech and save to file."""
        try:
            os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
            full_output_path = os.path.join(TTS_OUTPUT_DIR, output_path)
            self.tts.tts_to_file(text=text, file_path=full_output_path)
            print(f"Successfully generated speech file: {full_output_path}")
        except Exception as e:
            print(f"Error during speech generation: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        tts = CoquiTTS()
        tts.speak("Hello, how are you?", "response.wav")
    except Exception as e:
        print(f"Error in main: {str(e)}")





