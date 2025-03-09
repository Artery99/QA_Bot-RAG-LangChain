from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
import gradio as gr
import warnings

# Suppress warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings('ignore')

# ==================== Load PDF Document ====================
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()  # Load pages as a list of Document objects
    full_text = "\n".join([doc.page_content for doc in documents])  # Merge pages
    print("Extracted Text (First 1000 characters):\n", full_text[:1000])  # Print first 1000 chars
    return documents  # Return the document for further processing


# ==================== Split Document Text ====================
def split_document_text(document_text):
    # Define the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,   # Split into chunks of 500 characters
        chunk_overlap=50, # Overlap of 50 characters for context
        length_function=len
    )
    
    # Split the document text into chunks
    chunks = text_splitter.split_text(document_text)
    
    return chunks


# ==================== Embed Document Chunks ====================
def embed_document_chunks(chunks):
    # Initialize embedding model (HuggingFace Embeddings model)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Convert document chunks into LangChain Document format
    documents = [Document(page_content=chunk, metadata={"source": "new-Policies.txt"}) for chunk in chunks]
    
    # Create Chroma vector database and add the document chunks
    vector_db = Chroma.from_documents(documents, embedding_model)
    
    return vector_db


# ==================== Create and Configure a Vector Database ====================
def create_vector_database(file):
    # Read the document content from the file
    with open(file, 'r') as f:
        document_text = f.read()

    # Split the document into chunks
    chunks = split_document_text(document_text)
    
    # Create a vector database using the embedded chunks
    vector_db = embed_document_chunks(chunks)
    
    return vector_db


# ==================== Develop a Retriever to Fetch Document Segments ====================
def create_retriever(file):
    vector_db = create_vector_database(file)
    
    # Create a retriever using Chroma's built-in retriever
    retriever = vector_db.as_retriever()
    
    return retriever