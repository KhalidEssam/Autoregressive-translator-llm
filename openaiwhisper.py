import whisper
import os

model = whisper.load_model("tiny")

audiofile = "e:\work\medical_assistant\speech.mp3"
fileexists = os.path.isfile(audiofile)

if (fileexists):
    result = model.transcribe(audiofile)
    print(result["text"])
else:
    print("File does not exist")