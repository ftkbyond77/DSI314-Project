import os
import random
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone

# -----------------------------
# Load .env
# -----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

INDEX_NAME = "dsi314"

if INDEX_NAME not in pc.list_indexes().names():
    print(f"❌ Index '{INDEX_NAME}' not found")
else:
    index = pc.Index(INDEX_NAME)

    stats = index.describe_index_stats()
    total_vectors = stats['total_vector_count']
    print(f"Total vectors in index '{INDEX_NAME}': {total_vectors}")

    if total_vectors == 0:
        print("❌ Index ไม่มี vector ให้ดึง")
    else:
        # -----------------------------
        # สุ่ม vector ขนาด dimension ของ index
        # -----------------------------
        # ดึง dimension ของ index
        dimension = stats['dimension']
        random_vector = np.random.rand(dimension).tolist()

        # query K nearest neighbors (approximate random sampling)
        K = 10
        result = index.query(vector=random_vector, top_k=K, include_metadata=True)
        print("Sampled vectors:")
        for match in result['matches']:
            print(match['id'], match.get('metadata'))
