from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
    # Create the Ollama embedding instance
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # You can change the model here based on availability
    return embeddings

# Example usage
embedding_function = get_embedding_function()
embeddings = embedding_function.embed_documents(["Your document text here"])
print(embeddings)
