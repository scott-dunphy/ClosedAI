import streamlit as st
from pathlib import Path
from openai import OpenAI
from io import BytesIO

def text_to_speech(text):
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    client = OpenAI()
    
    # Create a BytesIO object to store the audio data
    audio_buffer = BytesIO()
    
    # Generate speech using OpenAI's API
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    
    # Stream the audio data to the BytesIO object
    response.stream_to_buffer(audio_buffer)
    
    # Reset the buffer's position to the beginning
    audio_buffer.seek(0)
    
    return audio_buffer

