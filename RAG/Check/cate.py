import os
from dotenv import load_dotenv
from pinecone import Pinecone
from collections import defaultdict

# ------------------------------------------------
# 1. โหลด .env ด้านนอก 1 ชั้น
# ------------------------------------------------
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

if not api_key or not index_name:
    raise ValueError("❌ Missing PINECONE_API_KEY or PINECONE_INDEX_NAME in .env file")

# ------------------------------------------------
# 2. เชื่อมต่อ Pinecone
# ------------------------------------------------
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

# ------------------------------------------------
# 3. เตรียมตัวนับข้อมูล
# ------------------------------------------------
file_chunk_count = defaultdict(int)     # นับ chunk ต่อไฟล์
upload_to_files = defaultdict(set)      # เชื่อม upload_id → set of files

def normalize_metadata(meta):
    """
    แปลง metadata ให้เป็น schema กลาง:
    - category = upload_id
    - doc_name = file (ชื่อเอกสารต้นฉบับ)
    """
    doc_name = "unknown"
    category = "unknown"

    if isinstance(meta, dict):
        # 1. upload_id
        category = str(meta.get("upload_id", "unknown")).strip()

        # 2. file
        raw_file = None
        if "file" in meta and meta["file"]:
            raw_file = meta["file"]
        elif "metadata" in meta and isinstance(meta["metadata"], dict):
            raw_file = meta["metadata"].get("file")

        if raw_file:
            # strip space/newline และ convert เป็น string
            doc_name = str(raw_file).strip().replace("\n", "").replace("\r", "")
    
    return category, doc_name

# ------------------------------------------------
# 4. โหลด vectors ทั้งหมดจาก Pinecone
# ------------------------------------------------
print("🔄 กำลังโหลด vectors ทั้งหมดจาก Pinecone...")

total_vectors = 0
for batch in index.list():
    for vector in batch:
        meta = vector.get("metadata", {}) if isinstance(vector, dict) else {}
        category, doc_name = normalize_metadata(meta)

        file_chunk_count[doc_name] += 1
        upload_to_files[category].add(doc_name)

print(f"✅ โหลดข้อมูลทั้งหมด {total_vectors:,} vectors สำเร็จ\n")

# ------------------------------------------------
# 5. แสดงผลรวม chunks ต่อเอกสาร
# ------------------------------------------------
print("📘 จำนวน chunk ต่อเอกสาร:\n")
for file_name, c in sorted(file_chunk_count.items(), key=lambda x: x[1], reverse=True):
    print(f"{file_name:<40} {c:>5} chunks")

# ------------------------------------------------
# 6. สรุปผลตาม upload_id (หมวดหมู่)
# ------------------------------------------------
print("\n📂 สรุปตาม upload_id:\n")
for upload_id, files in upload_to_files.items():
    total_chunks = sum(file_chunk_count[f] for f in files)
    print(f"Upload ID: {upload_id}")
    print(f"  - เอกสารทั้งหมด: {len(files)} ไฟล์")
    print(f"  - จำนวน chunks รวม: {total_chunks}\n")

# ------------------------------------------------
# 7. Return summary สำหรับเรียกใช้จากโมดูลอื่น
# ------------------------------------------------
summary = {
    "by_file": dict(file_chunk_count),
    "by_upload": {
        upload_id: {
            "file_count": len(files),
            "chunk_count": sum(file_chunk_count[f] for f in files),
        }
        for upload_id, files in upload_to_files.items()
    },
}

if __name__ == "__main__":
    print("✅ Summary dictionary ready for further use")
