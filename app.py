# Import required libraries
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
#from transformers import pipeline  # Hugging Face's pipeline for text generation
import utils
from PIL import Image
import os
def initialize_session_state():
    """
    Session State is a way to share variables between reruns, for each user session.
    """
    st.session_state.setdefault('history', [])
    st.session_state.setdefault('generated', ["Hello! I am here to provide answers to questions extracted from uploaded PDF files."])
    st.session_state.setdefault('past', ["Hello Buddy!"])

def create_conversational_chain(llm, vector_store):
    """
    Create a conversational chain using Google's Gemini model and vector store
    Args:
        llm: Instance of Google's Gemini model
        vector_store: FAISS Vector store containing indexed PDF document chunks
    Returns:
        chain: ConversationalRetrievalChain instance for handling Q&A
    """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key = 'answer')
    
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2,'fetch_k':2}),
                                                 memory=memory)
    return chain

def display_chat(conversation_chain):
    """
    Handle chat UI and interactions using Streamlit components.
    Creates two containers:
        - Input container: For user question input form
        - Reply container: For displaying chat history with avatars
    Args:
        conversation_chain: Instance of ConversationalRetrievalChain for processing queries
    """
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask me questions from uploaded PDF", key='input')
            submit_button = st.form_submit_button(label='Send ⬆️')
        
        # Check if user submits question with user input and generate response of the question
        if submit_button and user_input:
            generate_response(user_input, conversation_chain)
    
    # Display generated response to Streamlit web UI
    display_generated_responses(reply_container)

def generate_response(user_input, conversation_chain):
    """
    Generate LLM response for user questions using vector database retrieval.
    Maintains chat history in Streamlit session state for contextual conversations.
    Args:
        user_input (str): User's question text
        conversation_chain: ConversationalRetrievalChain instance
    """
    with st.spinner('Spinning a snazzy reply...'):
        output = conversation_chat(user_input, conversation_chain, st.session_state['history'])

    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

def conversation_chat(user_input, conversation_chain, history):
    """
    Process user input through LLM with error handling and timeouts
    Args:
        user_input (str): User's question
        conversation_chain: ConversationalRetrievalChain instance
        history: List of previous QA pairs
    Returns:
        str: LLM generated response or error message
    """

    try:
        # Add a timeout to the chain invocation
        result = conversation_chain.invoke(
            {
                "question": user_input, 
                "chat_history": history  # Keep only recent history
            },
            config={"timeout": 30}  # 15 second timeout
        )
        
        # Get and truncate answer
        #answer = result.get("answer", "")
        
        # Store in history
        history.append((user_input, result['answer']))
        return result['answer']
        
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            return "The model is taking too long to respond. Please try asking a shorter question."
        elif "busy" in error_msg.lower():
            return "The model is currently busy. Please wait a moment and try again."
        else:
            return f"Error: {error_msg}"


def display_generated_responses(reply_container):
    """
    Display generated LLM response to Streamlit Web UI
    Args:
    - reply_container: Streamlit container created at previous step
    """
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="adventurer")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

def main():
    """
    First function to call when we start Streamlit app
    """
    # Step 1: Initialize session state
    initialize_session_state()
    
    st.title("Chat Bot")
    st.sidebar.title("Configuration")
    api_token = st.sidebar.text_input(
        "Enter Google API Token:",
        type="password",
        help="Get your token from Google Cloud"
    )
    
    if api_token:
        os.environ['GOOGLE_API_TOKEN'] = api_token
    else:
        st.warning("Please enter your Google API token to continue.")
        st.markdown("""
        To get your token:
        1. Sign up at Google Cloud PlatForm
        2. Create a new token and paste it above
        """)
        return
    image = Image.open('chatbot.jpg')
    st.image(image, width=150)
    
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>

            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    # Step 2: Initialize Streamlit
    st.sidebar.title("Upload Pdf")
    #file_uploader, the data are copied to the Streamlit backend via the browser, 
    #and contained in a BytesIO buffer in Python memory (i.e. RAM, not disk).
    pdf_files = st.sidebar.file_uploader("", accept_multiple_files=True)
    
    # Step 3: Create instance of Google's Gemini model using pipeline    
    llm = utils1.create_llm()  # Using Google's Gemini model
    
    # Step 4: Create Vector Store and store uploaded Pdf file to in-memory Vector Database FAISS
    # and return instance of vector store
    vector_store = utils1.create_vector_store(pdf_files)

    if vector_store:
        # Step 5: If Vector Store is created successfully with chunks of PDF files
        # then Create the chain object
        chain = create_conversational_chain(llm, vector_store)

        # Step 6 - Display Chat to Web UI
        display_chat(chain)
    else:
        print('Initialized App.')

if __name__ == "__main__":
    main()
