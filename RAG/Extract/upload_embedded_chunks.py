import json
import os
import sys
from chunk import DocumentChunk, PineconeUploader, EMBEDDING_DIM

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, "embedded_chunks.json")

def upload_from_json(json_file=JSON_PATH):
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        return

    print(f"📂 Loading {json_file} ...")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"✅ Loaded {len(data)} chunks from JSON")

    chunks = [
        DocumentChunk(
            chunk_id=item["chunk_id"],
            text=item["text"],
            embedding=item["embedding"],
            metadata=item.get("metadata", {})
        )
        for item in data
    ]

    print(f"🚀 Uploading {len(chunks)} chunks to Pinecone index...")
    uploader = PineconeUploader()
    uploader.upload_chunks(chunks)

    print("🎉 Upload complete.")

if __name__ == "__main__":
    upload_from_json()
