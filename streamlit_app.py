import os
import streamlit as st
import openai  # Adjusted import for clarity
from pinecone import Pinecone

# Initialize Streamlit secrets for API keys
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = "closedai"

# Initialize OpenAI and Pinecone clients
openai.api_key = os.getenv("OPENAI_API_KEY")  # Ensure the API key is set for OpenAI
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

class ThreadRunner:
    def __init__(self, index):
        self.index = index

    def query_pinecone(self, text_query):
        # Generate query vector using OpenAI Embedding model
        embedding_response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[text_query]
        )
        query_vector = embedding_response['data'][0]['embedding']

        # Query Pinecone with this vector
        results = self.index.query(vector=[query_vector], top_k=5, include_metadata=True)
        return results

# Initialize the ThreadRunner instance right after its class definition
runner = ThreadRunner(index)

st.title('AI NCREIF Query Tool with Pinecone Integration')

def run_query_and_display_results():
    query = st.session_state.get('query', '')
    if query:
        try:
            results = runner.query_pinecone(query)
            st.session_state['results'] = results if results else "No results found."
        except Exception as e:
            st.session_state['results'] = f"Error querying Pinecone: {str(e)}"

query = st.text_input("Enter your query:", key="query", on_change=run_query_and_display_results)

if 'results' in st.session_state:
    st.write("Results:", st.session_state['results'])
