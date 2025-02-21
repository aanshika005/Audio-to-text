#model to convert speech to text
import os
import math
from pydub import AudioSegment
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Parameters for chunking
max_bytes = 25 * 1024 * 1024  # 25MB (adjust as needed)
audio_ext = 'mp3'  # The audio file extension
options = {'verbose': True} #debugging or tracking the execution of a program

# Function to add a chunk to the list
def add_chunk(audio_chunk, audio_path, audio_ext, audio_chunks):
  audio_chunk_path = f"{audio_path[:audio_path.rfind('.')]}" \
                       f"_{len(audio_chunks) + 1}.{audio_ext}"
  audio_chunk.export(audio_chunk_path, format=audio_ext)
  audio_chunks.append(audio_chunk_path)

# Function to split large chunks into smaller subchunks
def raw_split(big_chunk, max_chunk_milliseconds, audio_path, audio_ext, audio_chunks):
  subchunks = math.ceil(len(big_chunk) / max_chunk_milliseconds)
  for subchunk_i in range(subchunks):
    chunk_start = max_chunk_milliseconds * subchunk_i
    chunk_end = min(max_chunk_milliseconds * (subchunk_i + 1), len(big_chunk))
    add_chunk(big_chunk[chunk_start:chunk_end], audio_path, audio_ext, audio_chunks)

# Function to split the audio file into manageable chunks
def split_audio(audio_path, max_bytes, audio_ext, options):
  audio_chunks = []
  audio_bytes = os.path.getsize(audio_path)

  if audio_bytes >= max_bytes:
    if options['verbose']:
      print(f"Audio exceeds maximum allowed file size. Splitting audio into chunks...")

      audio_segment_file = AudioSegment.from_file(audio_path, format = audio_ext)

      min_chunks = math.ceil(audio_bytes / max_bytes)
      max_chunk_milliseconds = int(len(audio_segment_file) // min_chunks)

      chunks = audio_segment_file.split_to_mono()

      current_chunk = chunks[0] if chunks else audio_segment_file

      for next_chunk in chunks[1:]:
        if len(current_chunk) > max_chunk_milliseconds:
          raw_split(current_chunk, max_chunk_milliseconds, audio_path, audio_ext, audio_chunks)
          current_chunk = next_chunk
        elif len(current_chunk) + len(next_chunk) <= max_chunk_milliseconds:
          current_chunk += next_chunk
        else:
          add_chunk(current_chunk, audio_path, audio_ext, audio_chunks)
          current_chunk = next_chunk

      if len(current_chunk) > max_chunk_milliseconds:
        raw_split(current_chunk, max_chunk_milliseconds, audio_path, audio_ext, audio_chunks)
      else:
        add_chunk(current_chunk, audio_path, audio_ext, audio_chunks)

      if options['verbose']:
        print(f'Total chunks: {len(audio_chunks)}\n')
  else:
    print("No need to split")
    audio_chunks.append(audio_path)
  return audio_chunks

def transcribe(audio):
  device = "cpu"
  torch_dtype = torch.float32
  model_id = "openai/whisper-large-v3-turbo"

  model = AutoModelForSpeechSeq2Seq.from_pretrained(
  model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)

  model.to(device)
  processor = AutoProcessor.from_pretrained(model_id)

  model.config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
  language= "en", task = "transcribe")
  model.generation_config.forced_decoder_ids = processor.tokenizer.get_decoder_prompt_ids(
  language= "en", task = "transcribe")

  pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
  )

  result = pipe(audio, return_timestamps=True)

  return result.get("text", "ERROR Transcribing")


def process_audio_files(folder_path):
  transcribed_texts = []
  # Check if folder exists
  if not os.path.exists(folder_path):
    print(f"Error: Folder '{folder_path}' does not exist.")
    return

  # Iterate through files in the folder
  for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if it's a file and has a valid audio extension
    if os.path.isfile(file_path) and filename.lower().endswith(audio_ext):
      print(f"Reading: {filename}")
      try:
        print("sent to chunker")
        audio_chunks = split_audio(file_path, max_bytes, audio_ext, options)

        for chunk in audio_chunks:
          text = transcribe(chunk)
          transcribed_texts.append(text)

        final_transcription = " ".join(transcribed_texts)
        
        text_filename = f"{os.path.splitext(file_path)[0]}.txt"
        with open(text_filename, "w", encoding="utf-8") as text_file:
          text_file.write(final_transcription)
          print(f"Transcription saved to {text_filename}\n")

        # Clean up chunk files
        for chunk in audio_chunks:
          if chunk != file_path:
            os.remove(chunk)
            
        transcribed_texts = []
        final_transcription = ""
    
      except Exception as e:
        print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
  folder_path = input("Enter the path to the folder containing audio files: ")
  process_audio_files(folder_path)


