from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os

# ==================== Create and Configure a Vector Database ====================

# Path to the 'new-Policies.txt' file
file_path = 'new-Policies.txt'

# Read the document content from the file
with open(file_path, 'r') as file:
    document_text = file.read()

# Initialize embedding model (e.g., HuggingFace Embeddings model)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Convert document into LangChain Document format
document = Document(page_content=document_text, metadata={"source": "new-Policies.txt"})

# Create Chroma vector database and add the document
vector_db = Chroma.from_documents([document], embedding_model)

# Create a retriever using Chroma's built-in retriever
retriever = vector_db.as_retriever()

# Conduct a similarity search using the query 'Smoking policy' and get the top 5 results
query = "Smoking policy"
search_results = retriever.get_relevant_documents(query)

# Print the top 5 results
print("\nTop 5 Similarity Search Results for Query:", query)
for i, result in enumerate(search_results):
    print(f"\nResult {i+1}:")
    print(result.page_content[:500])  # Show only the first 500 characters for readability
