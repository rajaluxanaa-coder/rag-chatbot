class Config:
    # Using FREE Hugging Face models
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "distilgpt2"  # Free!
    CHROMA_PERSIST_DIR = "./vector_db"
    COLLECTION_NAME = "knowledge_base"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K_RESULTS = 3