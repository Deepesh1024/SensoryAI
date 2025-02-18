import os
from groq import Groq
client = Groq(api_key="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")
filename = os.path.dirname(__file__) + "/audio.m4a"

def transcribe_audio(filename):
    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
        file=(filename, file.read()),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",
        )
        # print(transcription.text)
        return transcription.text
