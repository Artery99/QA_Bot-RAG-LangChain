# QA RAG Chatbot

This project implements a Retrieval-Augmented Generation (RAG) chatbot using IBM Watsonx and LangChain. The chatbot allows users to upload documents (PDF or TXT) and ask questions about the content, which the bot answers by retrieving and processing relevant information from the uploaded document.

## Features

- **Document Upload**: Users can upload PDF or TXT documents.
- **Text Splitting**: Large documents are split into smaller chunks for efficient processing.
- **Retrieval**: Relevant document segments are retrieved based on user queries.
- **Answer Generation**: The bot uses a language model to generate answers based on retrieved information.
- **Gradio Interface**: The app is equipped with a user-friendly web interface to interact with the bot.

## Files in the Project

1. **QA_bot_Project.py**  
   This is the main script for running the QA bot. It initializes the LLM (Language Model), document loaders, text splitters, embedding models, and the retriever. It also sets up a Gradio interface for user interaction.

2. **QA_bot_test.py**  
   A testing script that demonstrates loading PDF files, splitting the content into chunks, embedding these chunks, and creating a vector database for efficient retrieval.

3. **PDF_Loader.py**  
   This script loads PDF documents and prints the first 1000 characters of the text to verify correct extraction. It is used as part of the document processing pipeline.

4. **text_splitter.py**  
   This file contains the logic for splitting large documents into smaller, manageable chunks using `RecursiveCharacterTextSplitter`.

5. **Vector_Store.py**  
   This script defines how text embeddings are generated using Watsonx and how to create and configure a vector database (using Chroma) to store the embedded document chunks.

6. **Retriever.py**  
   Defines how a retriever is created using the Chroma vector store, to perform similarity searches and retrieve relevant document segments for answering user queries.

## Set up IBM Watsonx Credentials
Make sure to have an active IBM Watsonx account. You need to set up your API credentials for WatsonxLLM and WatsonxEmbeddings.

## Run the application
Run the main application script to start the Gradio web interface:
```bash
python QA_Bot_Project.py
```
The app will be hosted locally on http://0.0.0.0:7871.

Thank you!
