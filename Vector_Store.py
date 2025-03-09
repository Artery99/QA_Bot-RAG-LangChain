from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings

# ==================== Embed Text ====================
def embed_text(query):
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }
    
    # Initialize Watsonx embedding model
    embedding_model = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params,
    )
    
    # Generate embedding
    embedding = embedding_model.embed_query(query)
    
    # Print first five values of the embedding
    print("Embedding for query:", query)
    print("First 5 embedding values:", embedding[:5])
    
    return embedding

# Example usage
query = "How are you?"
embedding_result = embed_text(query)
