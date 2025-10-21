import os
from pinecone import Pinecone
import random
import json

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# stats = index.describe_index_stats()
# print(json.dumps(stats, indent=2))

# --- 4. Query with a random vector ---
random_vector = [random.random() for _ in range(1536)]
result = index.query(vector=random_vector, top_k=5, include_metadata=True)

print("🧭 Sample Document from Pinecone Index:\n")

if result.matches:
    sample_doc = result.matches[0]  
    formatted_doc = {
        "id": sample_doc.id,
        "score": sample_doc.score,
        "metadata": sample_doc.metadata
    }
    print(json.dumps(formatted_doc, indent=2, ensure_ascii=False))
else:
    print("No matches found in the index.")
