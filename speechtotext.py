#model to convert speech to text

from openai import OpenAI
client = OpenAI(api_key= "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")

audio_file = open("audio.mp3", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file, 
  response_format="text"
)
print(transcription)