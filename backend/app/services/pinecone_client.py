from pinecone import Pinecone
import os

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to index
index_name = "dsi314"  
index = pc.Index(index_name)

# Upsert embedding
def upsert_embeddings(id, vector):
    index.upsert([(id, vector)])

# Query embeddings
def query_embeddings(vector, top_k=3):
    result = index.query(vector=vector, top_k=top_k)
    return result
