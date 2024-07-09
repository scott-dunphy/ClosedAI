import os
import streamlit as st
import openai
from openai import OpenAI
from pinecone import Pinecone
from tts import text_to_speech



if st.button("Generate Podcast!"):
    audio_buffer = text_to_speech(st.session_state.ai_response)
    st.audio(audio_buffer, format='audio/mpeg')
    
st.markdown(
    """
    <style>
    h1 {
        font-size: 24px;
        color: #3c9ed9;
        text-align: left;
    }
    </style>
    <h1>\\\\ ProductIQ</h1>
    """,
    unsafe_allow_html=True
)

# Pinned responses
pinned_responses = {
  
}

# Check if the 'message' is already in the session state, if not, initialize it
if 'message' not in st.session_state:
    st.session_state.message = ""

if 'ai_response' not in st.session_state:
    st.session_state.ai_response = ""

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

def generate_audio(audio_text):
    if audio_text:
        pass
    return
        

# Set up the API keys
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai.api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]

client = OpenAI()

system_prompt = """You are an AI consultant tasked with helping investment management firms identify opportunities for new products and stay ahead of key industry trends. Your role is to thoroughly analyze presentations from leading consulting firms such as StepStone, Townsend Group, Aon, Willis Towers Watson, and Mercer, as well as research documents produced by investment managers.

Based on your analysis, prepare a comprehensive report that includes the following:

Executive Summary:
- Provide a high-level overview of the most promising areas for the investment firm to develop new products that align with current market demands and future projections.
- Highlight the key industry trends and drivers of change identified across multiple presentations and documents.
- Present concise bullet points summarizing your top insights and recommendations.

Detailed Analysis:
- Dive deeper into the rationale and specific evidence behind your conclusions, citing relevant sources and data points.
- Identify common themes that emerge across multiple reputable sources, indicating industry consensus.
- Discuss areas where consultants and researchers express the highest confidence in their predictions and recommendations.
- Explore unique and well-reasoned perspectives that challenge conventional wisdom but are supported by strong evidence and arguments.
- Focus on forward-looking projections and trend forecasts rather than merely describing the current state of the industry.

Recommendations:
- Provide clear and actionable recommendations for the investment firm to capitalize on the identified opportunities and navigate the evolving industry landscape.
- Prioritize recommendations based on their potential impact, feasibility, and alignment with the firm's strategic objectives.
- Offer guidance on the next steps the firm should take to implement your recommendations effectively.

Throughout the report, employ a clear and professional tone that demonstrates your expertise and credibility as a management consultant. Use industry-specific terminology and concepts accurately, but ensure that the content remains accessible to investment professionals who may not be experts in all areas.

Remember, your end goal is to provide the investment firm with reliable, actionable insights they can use to inform their product development priorities and strategic planning. Aim to add significant value beyond simply summarizing the presentations and documents. Apply your analytical skills to identify the most important takeaways, connect related ideas, and convey your findings and recommendations persuasively.

Structure the report in a logical and visually appealing format, using headings, subheadings, and bullet points to enhance readability. Ensure that the report is well-organized, concise, and easy to navigate, allowing the investment firm's decision-makers to quickly grasp the key points and take appropriate action."""



    

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
                model="gpt-4o",
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
            st.session_state.ai_response = output
            return output
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "Sorry, I couldn't generate a response. Please try again."

def generate_follow_up_questions(ai_response):
    prompt = f"Based on the following response, generate two recommended follow-up questions (separate them with a '|'):\n\n{ai_response}\n\nFollow-up questions:"
    completion_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI consultant tasked with helping investment management firms identify opportunities."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7
    )
    follow_up_questions = completion_response.choices[0].message.content.strip().split("|")
    return follow_up_questions


runner = ThreadRunner(index)



def handle_query(user_query):
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

                    # Generate and display follow-up question buttons
                    follow_up_questions = generate_follow_up_questions(ai_response)
                    for question in follow_up_questions:
                        st.button(question, key=f"follow_up_{question}", on_click=handle_query, args=(question,))
            else:
                with st.container():
                    st.write("**Assistant**: No relevant documents found. Please refine your query or try different keywords.")

user_query = st.chat_input("Enter your query:")
if user_query:
    handle_query(user_query)

display_pinned_responses()
