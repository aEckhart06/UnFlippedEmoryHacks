from Transcribe import transcribe
from openai import OpenAI
from dotenv import load_dotenv
import os
import argparse
load_dotenv()

client = OpenAI()

def call_model(transcript,current_timestamp, question):
    PROMPT = f"""
        You are a helpful professor that can answer questions about the information provided in the transcript of a video. You are given a timestamp of the video and a question. 
        You must answer the question based on the timestamp and the transcript of the video.

        Here is the transcript of the video with timestamps:
        {transcript}

        Here is the current timestamp of the video:
        {current_timestamp}

        Here is the question:
        {question}

        In your response, do not acknowledge that you were given a timestamp or the transcript. Just answer the question as if you are the professor.
        """
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": PROMPT
            }
        ]
    )

    return(completion.choices[0].message.content)
