from celery import shared_task
from .models import Upload, Chunk, Plan
from .pdf_utils import extract_text_from_pdf, chunk_text
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

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

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

@shared_task
def process_upload(upload_id):
    """Process a single PDF upload: extract text, chunk, and store in vector database."""
    try:
        upload = Upload.objects.get(id=upload_id)
        print(f"Processing upload {upload_id}: {upload.filename}")
        
        # Extract text from PDF
        text_data = extract_text_from_pdf(upload.file.path)
        chunks = chunk_text(text_data)
        print(f"Extracted {len(chunks)} chunks from {upload.filename}")

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
            print(f"Warning: Could not delete existing vectors for upload_id {upload.id}: {str(e)}")

        # Update upload metadata
        upload.pages = text_data["total_pages"]
        upload.status = "processing"
        upload.save()

        # Store chunks in database and vector store
        for i, chunk_data in enumerate(chunks):
            chunk_text_content = chunk_data["text"]
            start_page = chunk_data["start_page"]
            end_page = chunk_data["end_page"]
            chunk_id = f"{upload.id}_{i}"
            
            # Create chunk in database
            chunk = Chunk.objects.create(
                upload=upload,
                chunk_id=chunk_id,
                text=chunk_text_content,
                start_page=start_page,
                end_page=end_page
            )

            # Add to vector store
            try:
                vector_store.add_texts(
                    texts=[chunk_text_content],
                    ids=[chunk_id],
                    metadatas=[{
                        "upload_id": upload.id,
                        "file": upload.filename,
                        "start_page": start_page,
                        "end_page": end_page
                    }]
                )
            except Exception as e:
                print(f"Failed to add chunk {chunk_id} to vector store: {str(e)}")
                raise

        # Mark as processed
        upload.status = "processed"
        upload.save()
        print(f"Successfully processed upload {upload_id}: {upload.filename}")
        
    except Exception as e:
        print(f"Error in process_upload for upload_id {upload_id}: {str(e)}")
        # Mark as failed
        try:
            upload = Upload.objects.get(id=upload_id)
            upload.status = "failed"
            upload.save()
        except:
            pass
        raise

@shared_task
def generate_study_plan(user_id, upload_ids):
    """Generate a prioritized study plan for the given uploads using RAG (LangChain v0.3+ compatible)."""
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

        # Build retriever with filter for current uploads
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": min(15, len(filenames) * 4),
                "filter": {"upload_id": {"$in": [up.id for up in uploads]}}
            }
        )

        print("Retrieving relevant documents...")
        docs = retriever.invoke("study plan")   
        print(f"Retrieved {len(docs)} documents")

        # --- Format documents ---
        def format_docs(docs):
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

        context = format_docs(docs)

        # --- Build full prompt ---
        full_prompt = PROMPT.format(context=context)

        # --- Call LLM directly ---
        from langchain_core.messages import HumanMessage
        print("Invoking LLM for study plan generation...")
        response = llm.invoke([HumanMessage(content=full_prompt)])
        result = response.content.strip()
        print(f"LLM invocation completed (length={len(result)} chars)")

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

        # --- Save to database ---
        Plan.objects.filter(user_id=user_id, upload__id__in=upload_ids).delete()
        Plan.objects.create(
            user_id=user_id,
            upload=uploads.first(),
            plan_json=plan_json
        )

        print("✅ Study plan generated and saved successfully")
        return plan_json

    except Exception as e:
        print(f"Unexpected error in generate_study_plan: {str(e)}")
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