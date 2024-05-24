import streamlit as st
from pathlib import Path
from openai import OpenAI
from io import BytesIO
import os

client = OpenAI()

# Set up the API keys
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]



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

def generate_podcast_style(ai_response):
    prompt = f"Convert the following into a podcast style narrative. Don't actually make it a podcast though, I just want it in narrative form to make it more interesting to the listener:"
    completion_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI author and editor who takes boring text and converts it into interest narrative."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=750,
        n=1,
        stop=None,
        temperature=0.7
    )
    follow_up_questions = completion_response.choices[0].message.content.strip().split("|")
    return follow_up_questions
