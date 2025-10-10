# core/tasks.py — OPTIMIZED FOR LARGE PDFs WITH TOKEN & RATE LIMIT MANAGEMENT + TEXT SANITIZATION

from celery import shared_task
from .models import Upload, Chunk, Plan
from .pdf_utils import extract_text_from_pdf, chunk_text, sanitize_text
from .llm_config import embeddings, llm, INDEX_NAME
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import os
from dotenv import load_dotenv
import json
import warnings
import re
import time
import math
from collections import defaultdict
from typing import List, Dict

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

# ==================== CONFIGURATION ====================
MAX_TOKENS_PER_REQUEST = 180000  # Safe limit for Claude (leave buffer)
MAX_CHUNKS_PER_BATCH = 50  # Process chunks in batches
RATE_LIMIT_DELAY = 0.15  # Delay between API calls (500 RPM = ~0.12s)
MAX_RETRIES = 3
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# ==================== TOKEN ESTIMATION ====================
def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars = 1 token"""
    return len(text) // 4

def smart_truncate_context(context: str, max_tokens: int = 150000) -> str:
    """Intelligently truncate context to fit token limits"""
    estimated = estimate_tokens(context)
    if estimated <= max_tokens:
        return context
    
    # Calculate truncation ratio
    ratio = max_tokens / estimated
    target_length = int(len(context) * ratio * 0.95)  # 5% buffer
    
    # Truncate at document boundary if possible
    truncated = context[:target_length]
    last_doc_sep = truncated.rfind("\n===\n")
    if last_doc_sep > target_length * 0.7:  # Keep at least 70%
        truncated = truncated[:last_doc_sep]
    
    return truncated + "\n\n[Context truncated due to length]"

# ==================== BATCH PROCESSING ====================
def process_chunks_in_batches(chunks: List[Dict], batch_size: int = MAX_CHUNKS_PER_BATCH):
    """Split chunks into manageable batches"""
    for i in range(0, len(chunks), batch_size):
        yield chunks[i:i + batch_size]

# ==================== PROMPT TEMPLATE ====================
prompt_template = """You are an AI study planner. Analyze the document chunks below and create a prioritized study plan.

Retrieved Documents:
{context}

Optional User Constraints:
{user_prompt_constraints}

Your task:
Generate a JSON array ranking PDF files by study priority. You must:
- Generate exactly one entry for each unique PDF file listed in Total files.
- Avoid duplicates or missing files.
- Summarize the entire document's content from all its chunks first, then determine the overall priority and reason for the whole file.
- Consider user constraints if they are provided (e.g., urgency, focus topics, exam schedules).

Each entry needs:
- file: the PDF filename
- priority: number where 1 is highest priority
- reason: Detailed explanation (3–4 sentences) following this reasoning structure:
  1. **What**: Describe what this document covers.
  2. **Why**: Explain why it is important or urgent (based on content, prerequisites, or user constraints).
  3. **Compare**: Briefly compare its importance relative to other documents in the list.
  4. **Suggest**: Optionally suggest how or when to study it.
  5. **Additional reasoning**: Any other factors that affect its priority (e.g., length, difficulty, dependency on other materials).

Prioritization Rules:
1. **Analyze content holistically** — combine all chunks of each document before making a decision.
2. **Core & fundamental concepts** rank higher than applied or specialized topics.
3. **Introductory or prerequisite topics** should come before dependent or advanced topics.
4. **Exam-related or time-sensitive content** should receive higher priority.
5. Consider optional user constraints (e.g., deadlines, focus areas) to adjust priorities.
6. If difficulty can be inferred (intro vs advanced), weigh difficulty appropriately.
7. **Document weight by chunk count (log-scale)** — Use log(1 + chunk_count) to slightly increase priority for longer documents.

Return ONLY a valid JSON array with no extra text or formatting.
Example format: [
  {{
    "file": "fundamentals.pdf",
    "priority": 1,
    "reason": "What: This document covers the core principles and basic concepts of the subject. Why: It is crucial because understanding these fundamentals is necessary to comprehend advanced topics and it may also be closely related to exam content. Compare: Compared to other documents, this one provides the foundational knowledge that supports all other materials. Suggest: It should be studied first to build a strong base before progressing. Additional reasoning: The document is extensive and thorough, making it a critical reference throughout the course."
  }}
]
"""


PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "user_prompt_constraints"]
)

# ==================== UPLOAD PROCESSING ====================
@shared_task
def process_upload(upload_id):
    """Process a single PDF upload with batching for large files and text sanitization"""
    try:
        upload = Upload.objects.get(id=upload_id)
        print(f"Processing upload {upload_id}: {upload.filename}")
        
        # Extract text from PDF (with OCR if needed)
        text_data = extract_text_from_pdf(upload.file.path)
        chunks = chunk_text(text_data)
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            print(f"!Warning: No chunks extracted from {upload.filename}. File may be corrupted or image-only without OCR.")
            upload.status = "failed"
            upload.save()
            return
        
        print(f"Extracted {total_chunks} chunks from {upload.filename}")
        
        if text_data.get("ocr_pages", 0) > 0:
            print(f"!OCR was used on {text_data['ocr_pages']} pages")

        # Delete existing chunks for this upload
        Chunk.objects.filter(upload=upload).delete()
        
        # Initialize vector store
        vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        
        # Delete existing vectors for this upload
        try:
            vector_store.delete(filter={"upload_id": upload.id})
            print(f"Deleted existing vectors for upload_id {upload.id}")
        except Exception as e:
            print(f"Warning: Could not delete existing vectors: {str(e)}")

        # Update upload metadata
        upload.pages = text_data["total_pages"]
        upload.status = "processing"
        upload.save()

        # Process chunks in batches to avoid overwhelming the system
        processed_count = 0
        for batch_num, chunk_batch in enumerate(process_chunks_in_batches(chunks)):
            batch_texts = []
            batch_ids = []
            batch_metadatas = []
            
            for i, chunk_data in enumerate(chunk_batch):
                global_idx = batch_num * MAX_CHUNKS_PER_BATCH + i
                chunk_text_content = chunk_data["text"]
                
                # CRITICAL: Sanitize text before saving to database
                chunk_text_content = sanitize_text(chunk_text_content)
                
                # Skip if sanitization resulted in empty text
                if not chunk_text_content or len(chunk_text_content) < 10:
                    print(f"!Skipping empty chunk at index {global_idx}")
                    continue
                
                start_page = chunk_data["start_page"]
                end_page = chunk_data["end_page"]
                chunk_id = f"{upload.id}_{global_idx}"
                
                # Create chunk in database with sanitized text
                try:
                    Chunk.objects.create(
                        upload=upload,
                        chunk_id=chunk_id,
                        text=chunk_text_content,
                        start_page=start_page,
                        end_page=end_page
                    )
                except ValueError as e:
                    if "NUL" in str(e) or "0x00" in str(e):
                        print(f"!NUL character still present in chunk {global_idx}, applying aggressive sanitization")
                        # Additional aggressive sanitization
                        chunk_text_content = ''.join(c for c in chunk_text_content if c.isprintable() or c in '\t\n\r')
                        chunk_text_content = chunk_text_content.replace('\x00', '')
                        
                        if chunk_text_content and len(chunk_text_content) >= 10:
                            Chunk.objects.create(
                                upload=upload,
                                chunk_id=chunk_id,
                                text=chunk_text_content,
                                start_page=start_page,
                                end_page=end_page
                            )
                        else:
                            print(f"!Chunk {global_idx} empty after aggressive sanitization, skipping")
                            continue
                    else:
                        raise
                
                # Prepare for batch vector store insertion
                batch_texts.append(chunk_text_content)
                batch_ids.append(chunk_id)
                batch_metadatas.append({
                    "upload_id": upload.id,
                    "file": upload.filename,
                    "start_page": start_page,
                    "end_page": end_page
                })
            
            # Add batch to vector store
            if batch_texts:  # Only add if we have valid texts
                try:
                    vector_store.add_texts(
                        texts=batch_texts,
                        ids=batch_ids,
                        metadatas=batch_metadatas
                    )
                    processed_count += len(batch_texts)
                    print(f"Batch {batch_num + 1}: Added {len(batch_texts)} chunks ({processed_count}/{total_chunks})")
                    
                    # Rate limiting
                    time.sleep(RATE_LIMIT_DELAY)
                    
                except Exception as e:
                    print(f"Failed to add batch {batch_num} to vector store: {str(e)}")
                    raise
            else:
                print(f"Batch {batch_num + 1}: No valid chunks to add (all were sanitized away)")

        # Mark as processed
        upload.status = "processed"
        upload.save()
        print(f"Successfully processed upload {upload_id}: {upload.filename} ({processed_count} chunks)")
        
    except Exception as e:
        print(f"!Error in process_upload for upload_id {upload_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            upload = Upload.objects.get(id=upload_id)
            upload.status = "failed"
            upload.save()
        except:
            pass
        raise

# ==================== STUDY PLAN GENERATION ====================
@shared_task
def generate_study_plan(user_id, upload_ids, user_prompt_constraints=""):
    """Generate a prioritized study plan with smart chunking and token management"""
    try:
        print(f"Generating study plan for user {user_id} with uploads {upload_ids}")
        
        # Get processed uploads
        uploads = Upload.objects.filter(
            id__in=upload_ids,
            user_id=user_id,
            status="processed"
        )
        
        if not uploads.exists():
            return {"error": "No processed uploads found"}

        filenames = [upload.filename for upload in uploads]
        print(f"Processing files: {filenames}")

        # Initialize Pinecone vector store
        vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )

        # Calculate total chunks for smart retrieval
        total_chunks = Chunk.objects.filter(upload__in=uploads).count()
        print(f"Total chunks available: {total_chunks}")
        
        # Adaptive k based on total chunks (use sampling for very large documents)
        if total_chunks > 1000:
            k = min(100, len(filenames) * 15)  # Sample more aggressively
            fetch_k = 300
        elif total_chunks > 500:
            k = min(150, len(filenames) * 20)
            fetch_k = 400
        else:
            k = min(200, len(filenames) * 25)
            fetch_k = 500

        # Build retriever with adaptive settings
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": 0.5,
                "filter": {"upload_id": {"$in": [up.id for up in uploads]}}
            }
        )

        print(f"Retrieving relevant documents (k={k}, fetch_k={fetch_k})...")
        time.sleep(RATE_LIMIT_DELAY)  # Rate limiting
        docs = retriever.invoke("comprehensive study plan priority analysis")
        print(f"Retrieved {len(docs)} documents")

        # Format documents with smart truncation
        def format_docs(docs):
            file_chunks = defaultdict(list)
            for doc in docs:
                meta = doc.metadata
                filename = meta.get("file", "Unknown")
                file_chunks[filename].append(
                    f"Pages: {meta.get('start_page', '?')}-{meta.get('end_page', '?')}\n"
                    f"Content: {doc.page_content[:400]}...\n"
                )

            formatted = []
            seen_files = set(file_chunks.keys())
            
            for filename in sorted(seen_files):
                chunks_list = file_chunks[filename]
                # For large documents, sample chunks evenly
                if len(chunks_list) > 10:
                    step = len(chunks_list) // 10
                    chunks_list = [chunks_list[i] for i in range(0, len(chunks_list), step)][:10]
                
                chunks_str = "\n---\n".join(chunks_list)
                formatted.append(
                    f"Document: {filename}\n"
                    f"Total Chunks: {len(file_chunks[filename])}\n"
                    f"Sample Chunks:\n{chunks_str}"
                )
            
            result = f"Total files: {', '.join(sorted(seen_files))}\n\n"
            result += "\n===\n".join(formatted)
            
            # Smart truncation
            result = smart_truncate_context(result, MAX_TOKENS_PER_REQUEST - 10000)
            return result

        context = format_docs(docs)
        print(f"Context size: ~{estimate_tokens(context)} tokens")

        # Build full prompt
        full_prompt = PROMPT.format(
            context=context,
            user_prompt_constraints=user_prompt_constraints or "No specific constraints."
        )

        # Call LLM with retry logic
        from langchain_core.messages import HumanMessage
        
        for attempt in range(MAX_RETRIES):
            try:
                print(f"Invoking LLM (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(RATE_LIMIT_DELAY)  # Rate limiting
                
                response = llm.invoke([HumanMessage(content=full_prompt)])
                result = response.content.strip()
                print(f"LLM invocation completed (length={len(result)} chars)")
                break
                
            except Exception as e:
                print(f"!LLM call attempt {attempt + 1} failed: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

        # Extract JSON safely
        json_str = None
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            print("Found JSON in code block")
        else:
            json_match = re.search(r'\[.*?\]', result, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                print("Found JSON array directly")
            else:
                json_str = result
                print("Using raw result as JSON")

        try:
            plan_json = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON decode failed: {e}")
            print("Falling back to heuristic plan...")
            plan_json = create_fallback_plan(filenames)
            plan_json.insert(0, {
                "file": "SYSTEM_MESSAGE",
                "priority": 0,
                "reason": f"LLM returned invalid JSON. Using fallback. Error: {str(e)}. User constraints: {user_prompt_constraints}"
            })

        # Validate and enhance plan
        plan_json = validate_and_enhance_plan(plan_json, filenames, uploads)

        # Save to database
        Plan.objects.filter(user_id=user_id, upload__id__in=upload_ids).delete()
        Plan.objects.create(
            user_id=user_id,
            upload=uploads.first(),
            plan_json=plan_json
        )

        print("Study plan generated and saved successfully")
        return plan_json

    except Exception as e:
        print(f"!Unexpected error in generate_study_plan: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to generate plan: {str(e)}"}

# ==================== HELPER FUNCTIONS ====================
def validate_and_enhance_plan(plan_json, filenames, uploads):
    """Validate plan structure and add missing files"""
    if not isinstance(plan_json, list):
        raise ValueError("Generated plan is not a list")
    
    # Build lookup for existing entries
    existing_files = {item.get("file") for item in plan_json if isinstance(item, dict)}
    
    # Add missing files
    missing_files = set(filenames) - existing_files
    if missing_files:
        print(f"Adding {len(missing_files)} missing files to plan")
        for filename in missing_files:
            upload = next((u for u in uploads if u.filename == filename), None)
            chunk_count = Chunk.objects.filter(upload=upload).count() if upload else 0
            
            plan_json.append({
                "file": filename,
                "priority": 999,
                "reason": f"Document not analyzed by LLM. Contains {chunk_count} chunks. Requires manual review.",
                "chunk_count": chunk_count
            })
    
    # Sort by priority
    plan_json.sort(key=lambda x: x.get("priority", 999))
    
    # Renumber priorities
    for idx, item in enumerate(plan_json, 1):
        if item.get("file") != "SYSTEM_MESSAGE":
            item["priority"] = idx
    
    return plan_json

def create_fallback_plan(filenames):
    """Create a basic priority plan based on filename heuristics"""
    print(f"Creating heuristic plan for {len(filenames)} files")
    plan = []
    
    for filename in filenames:
        priority = 5
        reason = "Standard document. Fallback entry due to generation issues."
        
        lower_name = filename.lower()
        
        if any(kw in lower_name for kw in ["intro", "fundamental", "basic", "101", "chapter 1", "ch1"]):
            priority = 1
            reason = "Introductory or fundamental content. Essential foundation material that should be studied first."
        elif any(kw in lower_name for kw in ["exam", "test", "midterm", "final", "quiz", "review"]):
            priority = 2
            reason = "Exam preparation material. Time-sensitive content for test preparation."
        elif any(kw in lower_name for kw in ["data", "statistics", "analysis", "method"]):
            priority = 2
            reason = "Core technical content. Important analytical methods and techniques."
        elif any(kw in lower_name for kw in ["finance", "financial", "accounting"]):
            priority = 3
            reason = "Domain-specific knowledge. Specialized content that builds on fundamentals."
        elif any(kw in lower_name for kw in ["case", "report", "application", "example"]):
            priority = 4
            reason = "Applied case studies. Practical applications best studied after theory."
        elif any(kw in lower_name for kw in ["advanced", "special", "project", "research"]):
            priority = 5
            reason = "Advanced topics. Specialized content requiring prerequisite knowledge."
        
        plan.append({
            "file": filename,
            "priority": priority,
            "reason": reason
        })
    
    plan.sort(key=lambda x: (x["priority"], x["file"]))
    
    for idx, item in enumerate(plan, 1):
        item["priority"] = idx
    
    return plan