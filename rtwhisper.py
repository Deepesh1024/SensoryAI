import os
import speech_recognition as sr
from groq import Groq
from dotenv import load_dotenv
class Config:
    def __init__(self, env_variable_name="GROQ_API_KEY"):
        load_dotenv()  # Load environment variables from .env if present
        self.api_key = os.environ.get(env_variable_name)
        if not self.api_key:
            raise EnvironmentError(
                f"Environment variable '{env_variable_name}' not found."
            )
class GroqClientManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = self._create_groq_client()
    def _create_groq_client(self) -> Groq:
        return Groq(api_key=self.api_key)
class AudioRecorder:
    def __init__(self, pause_threshold: float = 2.0):
        self.pause_threshold = pause_threshold
    def record_until_silence(self, filename: str = "temp.wav") -> None:
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        recognizer.pause_threshold = self.pause_threshold
        print(
            f"Recording... (will end after {self.pause_threshold}s of silence)\n"
            "Speak now, or remain silent to stop."
        )
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio_data = recognizer.listen(source)
        print("Recording ended. Saving audio file...")
        wav_bytes = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
        with open(filename, "wb") as f:
            f.write(wav_bytes)
        print(f"File saved as {filename}")
class AudioTranscriber:
    def __init__(self, groq_client: Groq):
        self.groq_client = groq_client
    def transcribe_audio(self, filename: str) -> str:
        with open(filename, "rb") as f:
            file_data = f.read()
        transcription = self.groq_client.audio.transcriptions.create(
            file=(filename, file_data),
            model="distil-whisper-large-v3-en",
            response_format="verbose_json",
        )
        return transcription.text
class FileManager:
    @staticmethod
    def delete_file(filename: str) -> None:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"\nTemporary file '{filename}' has been deleted.")
        else:
            print(f"\nFile '{filename}' not found (already deleted or never created).")
class TranscriptionApp:
    def __init__(
        self,
        env_variable_name="GROQ_API_KEY",
        output_filename="temp.wav",
        pause_threshold=2.0
    ):
        config = Config(env_variable_name)
        self.groq_manager = GroqClientManager(config.api_key)
        self.audio_recorder = AudioRecorder(pause_threshold)
        self.audio_transcriber = AudioTranscriber(self.groq_manager.client)
        self.output_filename = output_filename
    def run(self):
        self.audio_recorder.record_until_silence(filename=self.output_filename)
        print("\nSending audio to Groq for transcription...")
        text = self.audio_transcriber.transcribe_audio(self.output_filename)
        print("\nTranscription result:")
        print(text)
        FileManager.delete_file(self.output_filename)
# def main():
#     app = TranscriptionApp(
#         env_variable_name="GROQ_API_KEY",  
#         output_filename="temp.wav",
#         pause_threshold=2.0
#     )
#     app.run()
# if __name__ == "__main__":
#     main()