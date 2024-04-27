import os
import streamlit as st
import openai
from openai import OpenAI
from pinecone import Pinecone

# Set up the API keys
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

client = OpenAI()

# Initialize Pinecone client and index
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    st.error(f"Error initializing Pinecone: {str(e)}")
    st.stop()

class ThreadRunner:
    def __init__(self, index_name):
        self.index_name = index_name

    def query_pinecone(self, text_query):
        try:
            embedding_response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=[text_query]
            )
            query_vector = embedding_response.data[0].embedding
            results = index.query(vector=query_vector, top_k=6, include_metadata=True)
            if results['matches']:
                formatted_results = [match['metadata']['text'] for match in results['matches']]
                response = "\n".join(formatted_results)
            return response
        except Exception as e:
            st.error(f"Error querying Pinecone: {str(e)}")
            return None

    def generate_response(self, user_query, pinecone_results):
        try:
            prompt = f"User Query: {user_query}\n\nRelevant Documents:\n{pinecone_results}\n\nAssistant:"
            completion_response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.7
            )
            return completion_response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "Sorry, I couldn't generate a response. Please try again."

runner = ThreadRunner(index)

st.title('AI NCREIF Query Tool with Pinecone Integration and Chat Completions')
def handle_query(user_query):  # Ensure this function is correctly receiving 'user_query'
    if user_query:
        with st.container():
            st.write(f"**User**: {user_query}")
            pinecone_results = runner.query_pinecone(user_query)
            if pinecone_results:
                ai_response = runner.generate_response(user_query, pinecone_results)
                with st.container():
                    st.write(f"**Assistant**: {ai_response}")
            else:
                with st.container():
                    st.write("**Assistant**: No relevant documents found. Please refine your query or try different keywords.")

user_query = st.chat_input("Enter your query:")
if user_query:
    handle_query(user_query)
