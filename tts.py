import streamlit as st
from pathlib import Path
from openai import OpenAI
from io import BytesIO

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
    
    # Stream the audio data to the BytesIO object
    response.stream_to_buffer(audio_buffer)
    
    # Reset the buffer's position to the beginning
    audio_buffer.seek(0)
    
    return audio_buffer

# Streamlit app code
st.title("Text to Speech")

# Get the text input from the user
text = st.text_area("Enter the text to convert to speech:")

# Button to trigger text-to-speech conversion
if st.button("Convert to Speech"):
    if text:
        # Convert text to speech
        audio_buffer = text_to_speech(text)
        
        # Play the audio in the Streamlit app
        st.audio(audio_buffer, format='audio/mpeg')
    else:
        st.warning("Please enter some text to convert to speech.")
