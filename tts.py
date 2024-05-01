import streamlit as st
from pathlib import Path
from openai import OpenAI
from io import BytesIO
import os

def text_to_speech(text):
    client = OpenAI()
    
    # Create a BytesIO object to store the audio data
    audio_buffer = BytesIO()
    
    # Generate speech using OpenAI's API
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    
    # Write the response content to the BytesIO object
    audio_buffer.write(response.content)
    
    # Reset the buffer's position to the beginning
    audio_buffer.seek(0)
    
    return audio_buffer
