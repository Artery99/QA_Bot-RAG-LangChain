from langchain_community.document_loaders import PyPDFLoader

# ==================== Load PDF Document ====================
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()  # Load pages as a list of Document objects
    full_text = "\n".join([doc.page_content for doc in documents])  # Merge pages
    print("Extracted Text (First 1000 characters):\n", full_text[:1000])  # Print first 1000 chars
    return documents  # Return the document for further processing

# Example usage
pdf_path = "A_Comprehensive_Review_of_Low_Rank_Adaptation_in_Large_Language_Models_for_Efficient_Parameter_Tuning-1.pdf"
loaded_documents = load_pdf(pdf_path)
