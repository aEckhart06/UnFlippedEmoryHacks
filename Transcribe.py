from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
client = OpenAI()

def segments_to_text(segments, file_path: str="segments.md"):
    segments_text = ""
    for segment in segments:
        # Truncate timestamps to 2 decimal places
        start_time = f"{segment.start:.2f}"
        end_time = f"{segment.end:.2f}"
        segments_text += f"Start: {start_time}s End: {end_time}s Text: {segment.text}\n"

    return segments_text

def transcribe(audio_file):
    # Check if the audio file exists
    if audio_file is None:
        print(f"Error: No audio file provided.")
        return None

    print("Transcribing...")
    transcript = client.audio.translations.create(
        model="whisper-1",
        file=audio_file,
        response_format="verbose_json"
    )

    return segments_to_text(transcript.segments)
