"""
Test script for uploading and processing PDFs with study plan generation.
Usage: python test_upload.py [folder_path]
"""
import os
import sys
import json
import django
import hashlib
import time
from pathlib import Path

# Django setup
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "student_assistant.settings")
django.setup()

from core.pdf_utils import extract_text_from_pdf, chunk_text
from core.llm_config import embeddings, INDEX_NAME, get_index_stats, llm
from core.tasks import process_upload
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from core.models import Upload, Chunk, Plan
import warnings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
import re

warnings.filterwarnings("ignore", category=DeprecationWarning)

User = get_user_model()

# Prompt from tasks.py - with escaped literal curly braces
prompt_template = """You are an AI study planner. Analyze the document chunks below and create a prioritized study plan.

Retrieved Documents:
{context}

Your task: Generate a JSON array ranking PDF files by study priority. Each entry needs:
- file: the PDF filename
- priority: number where 1 is highest priority  
- reason: brief explanation

Prioritization rules:
1. Fundamental/technical content ranks higher
2. Introductory material before advanced
3. Exam content gets high priority
4. Prerequisites before dependent topics

Return ONLY a valid JSON array with no extra text or formatting.
Example format: [{{ "file": "example.pdf", "priority": 1, "reason": "Core fundamentals" }}]
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context"]
)

def calculate_file_hash(file_path):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def print_stats():
    """Print current system statistics."""
    print("\n" + "="*60)
    print("SYSTEM STATISTICS")
    print("="*60)
    
    # Database stats
    total_uploads = Upload.objects.count()
    processed_uploads = Upload.objects.filter(status="processed").count()
    total_chunks = Chunk.objects.count()
    total_plans = Plan.objects.count()
    
    print(f"Database:")
    print(f"  - Total uploads: {total_uploads}")
    print(f"  - Processed uploads: {processed_uploads}")
    print(f"  - Total chunks: {total_chunks}")
    print(f"  - Total plans: {total_plans}")
    
    # Pinecone stats
    print(f"\nPinecone Index ({INDEX_NAME}):")
    stats = get_index_stats()
    if "error" not in stats:
        print(f"  - Dimension: {stats['dimension']}")
        print(f"  - Total vectors: {stats['total_vectors']}")
        print(f"  - Index fullness: {stats['index_fullness']*100:.2f}%")
    else:
        print(f"  - Error: {stats['error']}")
    
    print("="*60 + "\n")

def format_docs(docs):
    """Format documents for prompt context."""
    formatted = []
    seen_files = set()
    for doc in docs:
        meta = doc.metadata
        filename = meta.get("file", "Unknown")
        seen_files.add(filename)
        formatted.append(
            f"Document: {filename}\n"
            f"Pages: {meta.get('start_page', '?')}-{meta.get('end_page', '?')}\n"
            f"Content Preview: {doc.page_content[:500]}...\n"
        )
    result = f"Total files: {', '.join(sorted(seen_files))}\n\n"
    result += "\n---\n".join(formatted)
    return result

def generate_study_plan_sync(user_id, upload_ids):
    """Synchronous study plan generation using LCEL chain (adapted from debug_chain)."""
    try:
        print(f"Generating study plan synchronously for user {user_id} with uploads {upload_ids}")
        
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

        # Build retriever with filter for current uploads
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": min(15, len(filenames) * 4),
                "filter": {"upload_id": {"$in": [up.id for up in uploads]}}
            }
        )

        print("Building LCEL chain...")
        def retrieve_and_format(input_query):
            docs = retriever.invoke(input_query)
            return format_docs(docs)
        
        rag_chain = (
            {"context": RunnableLambda(retrieve_and_format)}
            | PROMPT
            | llm
            | StrOutputParser()
        )

        print("Retrieving relevant documents and generating plan...")
        result = rag_chain.invoke("study plan")
        print(f"Chain invocation completed (length={len(result)} chars)")

        # --- Extract JSON safely ---
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
                "reason": f"LLM returned invalid JSON. Using fallback. Error: {str(e)}"
            })

        # --- Validate JSON structure ---
        if not isinstance(plan_json, list):
            raise ValueError("Generated plan is not a list")
        for idx, item in enumerate(plan_json):
            if not isinstance(item, dict):
                raise ValueError(f"Item {idx} is not a dict")
            if not all(k in item for k in ["file", "priority", "reason"]):
                raise ValueError(f"Item {idx} missing required keys: {item}")

        print(f"Validated plan with {len(plan_json)} entries")

        # --- Save to database (optional, for consistency) ---
        Plan.objects.filter(user_id=user_id, upload__id__in=upload_ids).delete()
        Plan.objects.create(
            user_id=user_id,
            upload=uploads.first(),
            plan_json=plan_json
        )

        print("✅ Study plan generated and saved successfully")
        return plan_json

    except Exception as e:
        print(f"Unexpected error in generate_study_plan_sync: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to generate plan: {str(e)}"}

def create_fallback_plan(filenames):
    """Create a basic priority plan based on filename heuristics."""
    print(f"Creating heuristic plan for {len(filenames)} files")
    plan = []
    
    for filename in filenames:
        priority = 5  # Default middle priority
        reason = "Standard document"
        
        # Apply heuristics
        lower_name = filename.lower()
        
        if any(kw in lower_name for kw in ["intro", "fundamental", "basic", "101", "chapter 1"]):
            priority = 1
            reason = "Introductory or fundamental content"
        elif any(kw in lower_name for kw in ["exam", "test", "midterm", "final", "quiz"]):
            priority = 2
            reason = "Exam preparation material"
        elif any(kw in lower_name for kw in ["data prep", "dsi310", "statistics", "analysis"]):
            priority = 2
            reason = "Core technical content"
        elif any(kw in lower_name for kw in ["finance", "financial", "accounting"]):
            priority = 3
            reason = "Domain-specific knowledge"
        elif any(kw in lower_name for kw in ["case", "report", "fraud", "application"]):
            priority = 4
            reason = "Applied case studies and reports"
        elif any(kw in lower_name for kw in ["advanced", "special", "project"]):
            priority = 5
            reason = "Advanced or specialized topics"
        
        plan.append({
            "file": filename,
            "priority": priority,
            "reason": reason
        })
    
    # Sort by priority, then alphabetically
    plan.sort(key=lambda x: (x["priority"], x["file"]))
    
    # Renumber priorities sequentially
    current_priority = 1
    last_priority = None
    for item in plan:
        if last_priority is None or item["priority"] != last_priority:
            last_priority = item["priority"]
        item["priority"] = current_priority
        current_priority += 1
    
    return plan

def test_process_folder(folder_path="Files_Test", force_reprocess=False):
    """
    Process all PDF files in a folder and generate study plan.
    
    Args:
        folder_path: Path to folder containing PDFs
        force_reprocess: If True, reprocess even if file hash matches
    
    Returns:
        JSON string with study plan or error
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING FOLDER: {folder_path}")
    print(f"{'='*60}\n")
    
    # Validate folder
    if not os.path.exists(folder_path):
        error_msg = f"Folder '{folder_path}' not found"
        print(f"ERROR: {error_msg}")
        return json.dumps({"error": error_msg}, indent=2)

    # Find PDF files
    pdf_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")])
    
    if not pdf_files:
        error_msg = f"No PDF files found in '{folder_path}'"
        print(f"ERROR: {error_msg}")
        return json.dumps({"error": error_msg}, indent=2)
    
    print(f"Found {len(pdf_files)} PDF file(s):")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf}")
    print()

    # Get or create test user
    try:
        user, created = User.objects.get_or_create(username="test_user")
        if created:
            print(f"Created new test user")
        else:
            print(f"Using existing test user")
    except Exception as e:
        error_msg = f"Failed to create/get test user: {str(e)}"
        print(f"ERROR: {error_msg}")
        return json.dumps({"error": error_msg}, indent=2)

    upload_ids = []
    processed_any = False
    skipped_count = 0

    # Process each PDF
    print(f"\n{'='*60}")
    print("PROCESSING PDFs")
    print(f"{'='*60}\n")
    
    for idx, pdf in enumerate(pdf_files, 1):
        print(f"[{idx}/{len(pdf_files)}] Processing: {pdf}")
        pdf_path = os.path.join(folder_path, pdf)
        
        try:
            # Calculate file hash
            file_hash = calculate_file_hash(pdf_path)
            print(f"  - File hash: {file_hash}")
            
            # Check if already uploaded
            existing_upload = Upload.objects.filter(user=user, filename=pdf).first()
            
            if existing_upload and not force_reprocess:
                # Compare hash
                try:
                    with open(existing_upload.file.path, "rb") as f:
                        existing_hash = hashlib.md5(f.read()).hexdigest()
                    
                    if existing_hash == file_hash:
                        print(f"  - Status: SKIPPED (unchanged)")
                        upload_ids.append(existing_upload.id)
                        skipped_count += 1
                        continue
                    else:
                        print(f"  - Status: CHANGED (will reprocess)")
                        # Delete old upload
                        existing_upload.file.delete(save=False)
                        existing_upload.delete()
                except Exception as e:
                    print(f"  - Warning: Could not compare hash: {str(e)}")
            
            # Upload file
            with open(pdf_path, "rb") as f:
                file_content = ContentFile(f.read(), name=pdf)
                upload = Upload.objects.create(
                    user=user,
                    file=file_content,
                    filename=pdf,
                    status="uploaded"
                )
            
            print(f"  - Upload ID: {upload.id}")
            print(f"  - Status: UPLOADED (processing...)")
            
            # Process upload (async task)
            process_upload.delay(upload.id)
            upload_ids.append(upload.id)
            processed_any = True
            
        except Exception as e:
            print(f"  - ERROR: {str(e)}")
            continue
        
        print()

    # Summary
    print(f"{'='*60}")
    print(f"Summary: {len(upload_ids)} files ready, {skipped_count} skipped")
    print(f"{'='*60}\n")
    
    if not upload_ids:
        error_msg = "No files processed or found"
        print(f"ERROR: {error_msg}")
        return json.dumps({"error": error_msg}, indent=2)

    # Wait for processing to complete
    if processed_any:
        print(f"{'='*60}")
        print("WAITING FOR PROCESSING")
        print(f"{'='*60}\n")
        
        for upload_id in upload_ids:
            upload = Upload.objects.get(id=upload_id)
            
            # Skip if already processed
            if upload.status == "processed":
                continue
            
            print(f"Waiting for upload {upload_id} ({upload.filename})...")
            timeout = 180  # 3 minutes
            start_time = time.time()
            
            while upload.status not in ["processed", "failed"] and time.time() - start_time < timeout:
                time.sleep(2)
                upload.refresh_from_db()
                elapsed = int(time.time() - start_time)
                print(f"  - Status: {upload.status} ({elapsed}s elapsed)")
            
            if upload.status == "processed":
                print(f"  - SUCCESS: Processing completed")
                # Get chunk count
                chunk_count = Chunk.objects.filter(upload=upload).count()
                print(f"  - Chunks created: {chunk_count}")
            elif upload.status == "failed":
                print(f"  - FAILED: Processing failed")
            else:
                error_msg = f"Timeout waiting for upload_id {upload_id} to process"
                print(f"  - ERROR: {error_msg}")
                return json.dumps({"error": error_msg}, indent=2)
            print()

    # Generate study plan synchronously using LCEL chain
    print(f"{'='*60}")
    print("GENERATING STUDY PLAN (USING LCEL CHAIN)")
    print(f"{'='*60}\n")
    
    try:
        print(f"Requesting study plan for {len(upload_ids)} file(s)...")
        plan_json = generate_study_plan_sync(user.id, upload_ids)
        
        if isinstance(plan_json, dict) and "error" in plan_json:
            print(f"ERROR: {plan_json['error']}")
        else:
            print(f"SUCCESS: Study plan generated with {len(plan_json)} items")
        
    except Exception as e:
        error_msg = f"Failed to generate plan: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return json.dumps({"error": error_msg}, indent=2)

    # Print final stats
    print_stats()
    
    # Return formatted JSON
    return json.dumps(plan_json, indent=2, ensure_ascii=False)

def cleanup_test_data():
    """Clean up all test data (useful for fresh start)."""
    print("\n" + "="*60)
    print("CLEANUP: Removing all test data")
    print("="*60)
    
    try:
        user = User.objects.filter(username="test_user").first()
        if user:
            # Delete all uploads (cascades to chunks and plans)
            upload_count = Upload.objects.filter(user=user).count()
            Upload.objects.filter(user=user).delete()
            print(f"Deleted {upload_count} uploads")
            
            # Optionally delete user
            # user.delete()
            # print("Deleted test user")
        
        print("Cleanup completed")
        
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

def main():
    """Main entry point."""
    # Parse arguments
    folder_path = sys.argv[1] if len(sys.argv) > 1 else "Files_Test"
    force_reprocess = "--force" in sys.argv
    
    if "--cleanup" in sys.argv:
        cleanup_test_data()
        return
    
    if "--stats" in sys.argv:
        print_stats()
        return
    
    # Run test
    result = test_process_folder(folder_path, force_reprocess)
    
    print("\n" + "="*60)
    print("FINAL STUDY PLAN")
    print("="*60)
    print(result)
    print("="*60 + "\n")

if __name__ == "__main__":
    main()