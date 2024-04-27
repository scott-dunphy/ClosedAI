import os
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from pinecone import ServerlessSpec
import time

# Set API key and user credentials from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = "closedai"

client = OpenAI()

# Create an assistant
assistant = client.beta.assistants.create(
    instructions="""
        Assist in retrieving and analyzing documents by querying a Pinecone vector database.
    """,
    model="gpt-4-turbo-preview",
    tools=[{"type": "code_interpreter"}]
)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

class ThreadRunner:
    def __init__(self, client, index):
        self.client = client
        self.thread = None

    def query_pinecone(self, text_query):
        # Generate query vector using OpenAI Embedding model (or any other model)
        embedding_response = client.Embeddings.create(
            model="text-embedding-ada-002",
            input=[text_query]
        )
        query_vector = embedding_response['data'][0]['embedding']

        # Query Pinecone with this vector
        results = index.query(vector=[query_vector], top_k=5, include_metadata=True)
        return results

# Initialize the ThreadRunner instance right after its class definition
runner = ThreadRunner(client, index)


st.title('AI NCREIF Query Tool with Pinecone Integration')

def run_query_and_display_results():
    query = st.session_state.get('query', '')
    if query:
        try:
            results = runner.query_pinecone(query)
            if results:
                st.session_state['results'] = results
            else:
                st.session_state['results'] = "No results found."
        except Exception as e:
            st.session_state['results'] = f"Error querying Pinecone: {str(e)}"
    else:
        st.session_state['results'] = "Please enter a query."

query = st.text_input("Enter your query:", key="query", on_change=run_query_and_display_results)

if 'results' in st.session_state:
    st.write("Results:", st.session_state['results'])
