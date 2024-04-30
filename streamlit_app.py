import os
import streamlit as st
import openai
from openai import OpenAI
from pinecone import Pinecone

st.markdown(
    """
    <style>
    h1 {
        font-size: 24px;
        color: #3c9ed9;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Pinned responses
pinned_responses = {
  
}

# Check if the 'message' is already in the session state, if not, initialize it
if 'message' not in st.session_state:
    st.session_state.message = ""

# Function to update the message
def update_message(message):
    st.session_state.message = message


# Sample sidebar setup with a title
with st.sidebar:
    st.title('Pinned Responses')
    if st.button('Update Message'):
        message = f""
        for title in st.session_state.selected_responses:
            message += f"{title}: {st.session_state.pinned_responses[title]}" + "\n\n"

        update_message(message)  # Call the function when the button is clicked

st.write(st.session_state.message)

# Initialize session state for pinned responses and selected responses
if 'pinned_responses' not in st.session_state:
    st.session_state.pinned_responses = {}

if 'selected_responses' not in st.session_state:
    st.session_state.selected_responses = list(st.session_state.pinned_responses.keys())

def pin_response(title, content):
    st.session_state.pinned_responses[title] = content
    #display_pinned_responses()
    
# Function to manage checkboxes and display content
def display_pinned_responses():
    with st.sidebar:
        # Display all checkboxes, irrespective of their check state
        for title, content in st.session_state.pinned_responses.items():
            checked = st.checkbox(title, key=f"checkbox_{title}", value=title in st.session_state.selected_responses)
            if checked:
                if title not in st.session_state.selected_responses:
                    st.session_state.selected_responses.append(title)
            else:
                if title in st.session_state.selected_responses:
                    st.session_state.selected_responses.remove(title)


        

# Set up the API keys
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

client = OpenAI()

system_prompt = """You are an AI assistant that helps investment management firms identify opportunities for new products and stay on top of key industry trends. Your role is to carefully analyze the content of presentations from top consulting firms like StepStone, Townsend Group, Aon, Willis Towers Watson, and Mercer, as well as research documents produced by investment managers.
Based on your analysis, you should:

Highlight the most promising areas for the investment firm to develop new products that align with current market demands and future projections
Identify the key industry trends and drivers of change discussed across multiple presentations/documents
Provide succinct executive summary bullets of your top insights and recommendations
If asked, go into more detail on your rationale and the specific evidence behind your conclusions
Answer any other questions the user has about the content you ingested to the best of your knowledge

When formulating your analysis and insights, look for:

Common themes that come up across multiple reputable sources, which suggest industry consensus
Areas where the consultants/researchers seem most confident in their predictions and recommendations
Unique, well-reasoned perspectives that go against the grain but are backed by strong evidence and arguments
Forward-looking projections and trend forecasts, not just descriptions of the current state

Remember that your end user is an investment professional who needs reliable, actionable insights they can use to inform their product development priorities and strategic planning. Aim to add value beyond just summarizing the presentations/documents. Apply your own analytical capabilities to connect the dots, identify the most important takeaways, and convey them clearly and concisely."""

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
            results = index.query(vector=query_vector, top_k=8, include_metadata=True)
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
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                n=1,
                stop=None,
                temperature=0.2
            )
            output = completion_response.choices[0].message.content.strip()
            return output
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "Sorry, I couldn't generate a response. Please try again."

runner = ThreadRunner(index)

st.title('//InvestmentInsider')
def handle_query(user_query):  # Ensure this function is correctly receiving 'user_query'
    if user_query:
        with st.container():
            st.write(f"**User**: {user_query}")
            pinecone_results = runner.query_pinecone(user_query)
            if pinecone_results:
                ai_response = runner.generate_response(user_query, pinecone_results)
                with st.container():
                    st.write(f"**Assistant**: {ai_response}")
                first_sentence = ai_response.split('.')[0]
                pin_response(user_query, first_sentence)
            else:
                with st.container():
                    st.write("**Assistant**: No relevant documents found. Please refine your query or try different keywords.")

user_query = st.chat_input("Enter your query:")
if user_query:
    handle_query(user_query)
    

display_pinned_responses()

