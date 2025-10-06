"""
LLM and Vector Store Configuration
Handles OpenAI embeddings, ChatGPT, and Pinecone vector database setup.
"""
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import time

load_dotenv()

# ===== Configuration =====
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 1536))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "dsi314")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Embedding model configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
# LLM model configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))

# ===== Validation =====
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# ===== OpenAI Embeddings =====
print(f"Initializing OpenAI embeddings: {EMBEDDING_MODEL} (dimension: {EMBEDDING_DIMENSION})")
embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=OPENAI_API_KEY,
    dimensions=EMBEDDING_DIMENSION
)

# ===== OpenAI LLM =====
print(f"Initializing OpenAI LLM: {LLM_MODEL} (temperature: {LLM_TEMPERATURE})")
llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_key=OPENAI_API_KEY,
    temperature=LLM_TEMPERATURE
)

# ===== Pinecone Setup =====
print(f"Connecting to Pinecone (index: {INDEX_NAME})...")
pc = Pinecone(api_key=PINECONE_API_KEY)

def initialize_pinecone_index(recreate=False):
    """
    Initialize Pinecone index. Creates if doesn't exist, validates if exists.
    
    Args:
        recreate: If True, delete and recreate the index (useful for dimension changes)
    
    Returns:
        bool: True if successful
    """
    try:
        existing_indexes = pc.list_indexes().names()
        
        if INDEX_NAME in existing_indexes:
            if recreate:
                print(f"Deleting existing index '{INDEX_NAME}'...")
                pc.delete_index(INDEX_NAME)
                # Wait for deletion to complete
                time.sleep(5)
            else:
                # Validate existing index
                index = pc.Index(INDEX_NAME)
                stats = index.describe_index_stats()
                current_dim = stats.get("dimension", 0)
                
                if current_dim != EMBEDDING_DIMENSION:
                    raise ValueError(
                        f"Dimension mismatch: Pinecone index has dimension {current_dim}, "
                        f"but EMBEDDING_DIMENSION is set to {EMBEDDING_DIMENSION}. "
                        f"Either delete the index manually or set recreate=True"
                    )
                
                print(f"Using existing Pinecone index '{INDEX_NAME}' (dimension: {current_dim})")
                return True
        
        # Create new index
        if INDEX_NAME not in pc.list_indexes().names():
            print(f"Creating new Pinecone index '{INDEX_NAME}' (dimension: {EMBEDDING_DIMENSION})...")
            pc.create_index(
                name=INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
            
            # Wait for index to be ready
            print("Waiting for index to be ready...")
            max_wait = 60
            start_time = time.time()
            while INDEX_NAME not in pc.list_indexes().names():
                if time.time() - start_time > max_wait:
                    raise TimeoutError(f"Index creation timeout after {max_wait}s")
                time.sleep(2)
            
            print(f"Index '{INDEX_NAME}' created successfully")
        
        return True
        
    except Exception as e:
        print(f"Error initializing Pinecone index: {str(e)}")
        raise

# Initialize index on import
try:
    initialize_pinecone_index(recreate=False)
except Exception as e:
    print(f"WARNING: Pinecone initialization failed: {str(e)}")
    print("You may need to manually create or delete the index")

def get_index_stats():
    """Get statistics about the current Pinecone index."""
    try:
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        return {
            "dimension": stats.get("dimension"),
            "total_vectors": stats.get("total_vector_count", 0),
            "namespaces": stats.get("namespaces", {}),
            "index_fullness": stats.get("index_fullness", 0)
        }
    except Exception as e:
        return {"error": str(e)}

def clear_index():
    """Delete all vectors from the index (useful for testing)."""
    try:
        index = pc.Index(INDEX_NAME)
        index.delete(delete_all=True)
        print(f"Cleared all vectors from index '{INDEX_NAME}'")
        return True
    except Exception as e:
        print(f"Error clearing index: {str(e)}")
        return False

# Export configuration for easy access
config = {
    "embedding_dimension": EMBEDDING_DIMENSION,
    "embedding_model": EMBEDDING_MODEL,
    "llm_model": LLM_MODEL,
    "llm_temperature": LLM_TEMPERATURE,
    "index_name": INDEX_NAME,
    "pinecone_cloud": PINECONE_CLOUD,
    "pinecone_region": PINECONE_REGION
}

print("\n=== Configuration Summary ===")
for key, value in config.items():
    print(f"{key}: {value}")
print("=" * 30 + "\n")