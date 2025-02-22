from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
#from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
from typing import List
from tqdm import tqdm


def create_llm(temperature: float = 0.4):
    """
    Initialize Google's Gemini Pro model for text generation
    Args:
        temperature (float): Controls randomness in generation (0.0 = deterministic, 1.0 = creative)
    Returns:
        ChatGoogleGenerativeAI: Configured Gemini model instance
    Raises:
        Exception: If model initialization fails or API token is missing
    """

    """
    try:

        huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        
        if not huggingface_api_token:
            raise ValueError("Please set HUGGINGFACE_API_TOKEN environment variable")
        
        llm = HuggingFaceEndpoint(
            #endpoint_url="https://api-inference.huggingface.co/models/gpt2",  # Smaller model
            endpoint_url = "https://api-inference.huggingface.co/models/distilgpt2",
            huggingfacehub_api_token = huggingface_api_token,
            task="text-generation",
            temperature = temperature,
            top_p = 0.9,
            top_k = 50,
            model_kwargs={
                "max_length": 150,          # Control response length
                #"min_length": 30,           # Ensure minimum meaningful response
                "no_repeat_ngram_size": 3,  # Prevent repetitive phrases
                "num_return_sequences": 1,   # Get single response
                #"pad_token_id": 50256
            }
        )
        #test = llm("test")
        return llm
    except Exception as e:
        raise Exception(f"Failed to initialize LLM: {str(e)}")

    """
    try:
        google_api_token = os.getenv("GOOGLE_API_TOKEN")
        
        if not google_api_token:
            raise ValueError("Please set GOOGLE_API_TOKEN environment variable")
        
        llm = ChatGoogleGenerativeAI(
            model = "gemini-1.5-pro-latest",
            google_api_key = google_api_token
        )
        return llm
    except Exception as e:
        raise Exception(f"Failed to initialize LLM: {str(e)}")
    
def create_vector_store(pdf_files: List):
    """
    Create FAISS vector store from uploaded PDF files for semantic search
    Args:
        pdf_files (List): List of uploaded PDF file objects from Streamlit
    Returns:
        FAISS: Vector store instance containing document embeddings, or None if no files
    Process:
        1. Creates temporary files for PDF processing
        2. Splits documents into chunks (4000 chars, 200 char overlap)
        3. Generates embeddings using all-MiniLM-L6-v2 model
        4. Indexes chunks in FAISS for efficient similarity search
    """
    vector_store = None

    if pdf_files:
        text = []

        for file in tqdm(pdf_files, desc="Processing files"):
            # Get the file extension to identify the type of file
            file_extension = os.path.splitext(file.name)[1]
            
            # Write the PDF file to a temporary location on disk
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            
            # Load the PDF using PyPDFLoader
            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            
            # If loader is available, load the documents
            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        # Split the loaded text into chunks using a character-based splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents(text)


        # Initialize embeddings using Hugging Face embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})

        # Create an in-memory FAISS vector store and store the document chunks using embeddings
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    return vector_store
