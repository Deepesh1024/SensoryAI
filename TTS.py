import openai
import os
from dotenv import load_dotenv
import subprocess

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from .env
api_key = os.getenv("OPENAI_API_KEY")

# Set OpenAI API key
openai.api_key = api_key

# Function to generate speech from text
def text_to_speech(text, output_file="output.mp3"):
    response = openai.audio.speech.create(
        model="tts-1",  # Use "tts-1-hd" for higher quality
        voice="onyx",  # Other options: "echo", "fable", "onyx", "nova", "shimmer"
        input=text
    )
    
    # Save audio file
    with open(output_file, "wb") as f:
        f.write(response.content)

    print(f"Audio saved as {output_file}")
    play_audio("output.mp3")



def play_audio(output_file):
    subprocess.run(["afplay", output_file])
# Example usage
# text_to_speech("Hiii! My name is Deepesh! I am testing BroCode here.")
