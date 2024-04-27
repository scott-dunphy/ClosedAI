import os
import streamlit as st
import openai
from pinecone import Pinecone

# Initialize Streamlit secrets for API keys
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = "closedai"

# Initialize Pinecone client and index
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
except Exception as e:
    st.error(f"Error initializing Pinecone: {str(e)}")
    st.stop()

class ThreadRunner:
    def __init__(self, index):
        self.index = index

    def query_pinecone(self, text_query):
        """
        Generate query vector using OpenAI Embedding model and query Pinecone index.
        """
        try:
            embedding_response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=[text_query]
            )
            query_vector = embedding_response['data'][0]['embedding']
            results = self.index.query(vector=query_vector, top_k=6, include_metadata=True)
            return results
        except Exception as e:
            st.error(f"Error querying Pinecone: {str(e)}")
            return None

    def generate_response(self, user_query, pinecone_results):
        """
        Generate a response using OpenAI's ChatCompletion model based on user query and Pinecone results.
        """
        try:
            prompt = f"User Query: {user_query}\n\nRelevant Documents:\n{pinecone_results}\n\nAssistant:"
            completion_response = openai.ChatCompletion.create(
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

# Chat interface
chat_container = st.chat()

def handle_query(user_query):
    if user_query:
        chat_container.user_message(user_query)
        # First, we query Pinecone to get relevant documents
        pinecone_results = runner.query_pinecone(user_query)
        if pinecone_results and pinecone_results['matches']:
            results_text = "\n".join([f"ID: {match['id']}, Score: {match['score']}" for match in pinecone_results['matches']])
            # Generate a response based on Pinecone's results
            ai_response = runner.generate_response(user_query, results_text)
            chat_container.system_message(ai_response)
        else:
            chat_container.system_message("No relevant documents found. Please refine your query or try different keywords.")

# User input for the chat
st.chat_input("Enter your query:", on_submit=handle_query)
