#model to convert speech to text

from openai import OpenAI
from pydub import AudioSegment
import os

def split_audio(file_path, chunk_size_mb):
    # Convert MB to bytes
    chunk_size_bytes = chunk_size_mb * 1024 * 1024

    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    file_size_bytes = len(audio.raw_data)
    
    # Calculate the duration of each chunk
    chunk_duration_ms = len(audio) * chunk_size_bytes // file_size_bytes
    
    # Split the audio file into chunks
    for i, start in enumerate(range(0, len(audio), chunk_duration_ms)):
        chunk = audio[start:start + chunk_duration_ms]
        chunk.export(f"chunk_{i + 1}.mp3", format="mp3")
        print(f"Exported chunk_{i + 1}.mp3")

# Path to the audio file
audio_file_path = "audio.mp3"
# Desired chunk size in MB
chunk_size_mb = 25

# Split the audio file
split_audio(audio_file_path, chunk_size_mb)

#speech to text
client = OpenAI(api_key= "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")

audio_file = open("audio.mp3", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file, 
  response_format="text"
)
print(transcription)