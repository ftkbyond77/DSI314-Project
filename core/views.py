# core/views.py — OPTIMIZED FOR LARGE PDFs WITH RATE LIMITING

from django.contrib.auth import authenticate, login, logout
from .forms import LoginForm, RegistrationForm
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth import get_user_model
from .models import Upload, Plan, Chunk
from .tasks import process_upload
from langchain_pinecone import PineconeVectorStore
from .llm_config import embeddings, INDEX_NAME, llm
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from collections import defaultdict
import json, os, re, time, warnings, math

warnings.filterwarnings("ignore", message=".*pydantic_v1.*")
User = get_user_model()

# ==================== CONFIGURATION ====================
MAX_TOKENS_PER_REQUEST = 180000
RATE_LIMIT_DELAY = 0.15  # 500 RPM = 0.12s minimum
MAX_RETRIES = 3
MAX_WAIT_TIME = 300  # 5 minutes for processing

# ==================== TOKEN ESTIMATION ====================
def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars = 1 token"""
    return len(text) // 4

def smart_truncate_context(context: str, max_tokens: int = 150000) -> str:
    """Intelligently truncate context to fit token limits"""
    estimated = estimate_tokens(context)
    if estimated <= max_tokens:
        return context
    
    ratio = max_tokens / estimated
    target_length = int(len(context) * ratio * 0.95)
    truncated = context[:target_length]
    
    last_doc_sep = truncated.rfind("\n===\n")
    if last_doc_sep > target_length * 0.7:
        truncated = truncated[:last_doc_sep]
    
    return truncated + "\n\n[Context truncated due to length - large document]"

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

# ==================== FORMAT DOCS ====================
def format_docs(docs, max_samples_per_file=8):
    """Combine chunks by filename with smart sampling for large documents"""
    file_chunks = defaultdict(list)
    
    for doc in docs:
        meta = doc.metadata
        filename = meta.get("file", "Unknown")
        snippet = doc.page_content.strip()
        if snippet:
            file_chunks[filename].append({
                "content": snippet,
                "pages": f"{meta.get('start_page', '?')}-{meta.get('end_page', '?')}"
            })
    
    # Build context with intelligent sampling
    formatted = []
    total_chunks = sum(len(chunks) for chunks in file_chunks.values())
    
    for fname, chunks in sorted(file_chunks.items()):
        chunk_count = len(chunks)
        
        # Smart sampling for large documents
        if chunk_count > max_samples_per_file:
            # Sample evenly across the document
            step = chunk_count // max_samples_per_file
            sampled_chunks = [chunks[i] for i in range(0, chunk_count, step)][:max_samples_per_file]
        else:
            sampled_chunks = chunks
        
        # Build preview
        preview_parts = []
        for idx, chunk in enumerate(sampled_chunks):
            preview_parts.append(
                f"  Chunk {idx+1} (Pages {chunk['pages']}):\n  {chunk['content'][:300]}..."
            )
        
        preview = "\n\n".join(preview_parts)
        
        formatted.append(
            f"📄 File: {fname}\n"
            f"Total Chunks: {chunk_count}\n"
            f"Sampled Chunks: {len(sampled_chunks)}\n"
            f"Content Preview:\n{preview}\n"
        )
    
    context = (
        f"Total unique files: {len(file_chunks)}\n"
        f"Total chunks retrieved: {total_chunks}\n\n"
        + "\n===\n".join(formatted)
    )
    
    return context, file_chunks

# ==================== FALLBACK PLAN ====================
def create_fallback_plan(filenames):
    plan = []
    for filename in filenames:
        priority = 5
        reason = "Standard document requiring review."
        lower_name = filename.lower()
        
        if any(kw in lower_name for kw in ["intro", "fundamental", "basic", "101", "ch1"]):
            priority, reason = 1, "Introductory content - foundational material"
        elif any(kw in lower_name for kw in ["exam", "test", "midterm", "final", "review"]):
            priority, reason = 2, "Exam preparation - time-sensitive content"
        elif any(kw in lower_name for kw in ["data", "statistics", "analysis", "method"]):
            priority, reason = 2, "Core technical content - essential methods"
        elif any(kw in lower_name for kw in ["finance", "accounting", "business"]):
            priority, reason = 3, "Domain knowledge - specialized content"
        elif any(kw in lower_name for kw in ["case", "report", "application", "example"]):
            priority, reason = 4, "Applied studies - practical applications"
        elif any(kw in lower_name for kw in ["advanced", "special", "project", "research"]):
            priority, reason = 5, "Advanced topics - requires prerequisites"
        
        plan.append({"file": filename, "priority": priority, "reason": reason})
    
    plan.sort(key=lambda x: (x["priority"], x["file"]))
    for i, item in enumerate(plan, 1):
        item["priority"] = i
    return plan

# ==================== GENERATE STUDY PLAN (SYNC) ====================
def generate_study_plan_sync(user, upload_ids, constraint_prompt=""):
    """Generate study plan with optimizations for large documents"""
    try:
        uploads = Upload.objects.filter(id__in=upload_ids, user=user, status="processed")
        if not uploads.exists():
            return {"error": "No processed uploads found"}

        filenames = [u.filename for u in uploads]
        
        # Calculate total chunks for adaptive retrieval
        total_chunks = Chunk.objects.filter(upload__in=uploads).count()
        total_pages = sum(u.pages or 0 for u in uploads)
        print(f"📊 Processing {len(filenames)} files, {total_chunks} chunks, {total_pages} pages")

        # Initialize Pinecone
        vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        
        # Adaptive k based on document size
        if total_chunks > 2000:  # Very large corpus
            k = min(80, len(filenames) * 10)
            max_samples = 6
        elif total_chunks > 1000:
            k = min(120, len(filenames) * 15)
            max_samples = 8
        elif total_chunks > 500:
            k = min(160, len(filenames) * 20)
            max_samples = 10
        else:
            k = min(200, len(filenames) * 25)
            max_samples = 12
        
        print(f"📊 Retrieval settings: k={k}, max_samples={max_samples}")
        
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": min(k * 3, 600),
                "lambda_mult": 0.5,
                "filter": {"upload_id": {"$in": [up.id for up in uploads]}}
            }
        )

        # Retrieve & format docs with rate limiting
        def retrieve_and_format(_):
            time.sleep(RATE_LIMIT_DELAY)
            docs = retriever.invoke("comprehensive study plan priority analysis")
            context, _ = format_docs(docs, max_samples_per_file=max_samples)
            # Apply token limit
            context = smart_truncate_context(context, MAX_TOKENS_PER_REQUEST - 15000)
            return context

        constraint_text = f"User constraints: {constraint_prompt}" if constraint_prompt else "No specific constraints provided."

        # Build RAG chain
        rag_chain = (
            {
                "context": RunnableLambda(retrieve_and_format),
                "user_prompt_constraints": lambda _: constraint_text
            }
            | PROMPT 
            | llm 
            | StrOutputParser()
        )

        # Invoke with retry logic
        result = None
        for attempt in range(MAX_RETRIES):
            try:
                print(f"🤖 Invoking LLM (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(RATE_LIMIT_DELAY)
                result = rag_chain.invoke("study plan")
                print(f"✅ LLM response received ({len(result)} chars)")
                break
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1} failed: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

        # Extract JSON
        json_str = None
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\[.*?\]', result, re.DOTALL)
            json_str = json_match.group(0) if json_match else result

        try:
            plan_json = json.loads(json_str)
        except Exception as e:
            print(f"❌ JSON decode failed: {e}")
            plan_json = create_fallback_plan(filenames)
            plan_json.insert(0, {
                "file": "SYSTEM_MESSAGE",
                "priority": 0,
                "reason": f"LLM returned invalid JSON. Using fallback heuristics. Constraints: {constraint_prompt}"
            })

        # Merge duplicates and enhance with chunk data
        merged = {}
        for entry in plan_json:
            fname = entry.get("file")
            if not fname or fname == "SYSTEM_MESSAGE":
                if fname == "SYSTEM_MESSAGE":
                    merged["SYSTEM_MESSAGE"] = entry
                continue
            
            if fname not in merged:
                merged[fname] = entry
            else:
                # Merge duplicate entries
                merged[fname]["reason"] += " / " + entry.get("reason", "")
                merged[fname]["priority"] = min(
                    merged[fname].get("priority", 999),
                    entry.get("priority", 999)
                )

        # Add chunk counts and calculate adjusted priorities
        for upload in uploads:
            fname = upload.filename
            chunk_count = Chunk.objects.filter(upload=upload).count()
            
            if fname in merged:
                merged[fname]["chunk_count"] = chunk_count
                merged[fname]["pages"] = upload.pages or 0
                # Log-scale bonus for longer documents
                length_bonus = math.log(1 + chunk_count) * 0.1
                merged[fname]["adjusted_priority"] = merged[fname]["priority"] - length_bonus
            else:
                # Missing file
                merged[fname] = {
                    "file": fname,
                    "priority": 999,
                    "chunk_count": chunk_count,
                    "pages": upload.pages or 0,
                    "reason": f"Document not analyzed by LLM. Contains {chunk_count} chunks across {upload.pages or 0} pages. Requires manual review.",
                    "adjusted_priority": 999
                }

        # Final sort and renumber
        final_plan = sorted(merged.values(), key=lambda x: (
            0 if x.get("file") == "SYSTEM_MESSAGE" else x.get("adjusted_priority", 999),
            x.get("file", "")
        ))
        
        # Renumber priorities (skip SYSTEM_MESSAGE)
        priority_counter = 1
        for item in final_plan:
            if item.get("file") != "SYSTEM_MESSAGE":
                item["priority"] = priority_counter
                priority_counter += 1

        # Save plan
        Plan.objects.filter(user_id=user.id, upload__id__in=upload_ids).delete()
        Plan.objects.create(user_id=user.id, upload=uploads.first(), plan_json=final_plan)

        print(f"✅ Study plan generated successfully with {len(final_plan)} entries")
        return final_plan

    except Exception as e:
        import traceback
        print(f"❌ Unexpected error in generate_study_plan_sync: {str(e)}")
        traceback.print_exc()
        return {"error": f"Failed to generate plan: {str(e)}"}

# ==================== UPLOAD VIEW ====================
@login_required
def upload_page(request):
    if request.method == 'POST':
        files = request.FILES.getlist('files')
        constraint_prompt = request.POST.get('constraint_prompt', '').strip()

        if not files:
            messages.error(request, 'Please select at least one PDF file.')
            return render(request, 'core/upload.html', {})

        if len(files) > 10:
            messages.error(request, 'Maximum 10 files allowed.')
            return render(request, 'core/upload.html', {})

        # Validate and prepare uploads
        upload_ids, new_uploads = [], []
        total_size = 0
        
        for file in files:
            if not file.name.lower().endswith('.pdf'):
                messages.error(request, f'Only PDF files allowed: {file.name}')
                return render(request, 'core/upload.html', {})
            
            # Check file size (optional: add size limit)
            file_size = file.size / (1024 * 1024)  # MB
            total_size += file_size
            
            if file_size > 100:  # 100MB per file
                messages.warning(request, f'Large file detected: {file.name} ({file_size:.1f}MB). Processing may take longer.')
            
            # Check for existing processed upload
            existing = Upload.objects.filter(
                user=request.user,
                filename=file.name,
                status='processed'
            ).first()
            
            if existing:
                print(f"♻️ Reusing existing upload: {file.name}")
                upload_ids.append(existing.id)
            else:
                upload = Upload.objects.create(
                    user=request.user,
                    file=file,
                    filename=file.name,
                    status='uploaded'
                )
                new_uploads.append(upload)
                upload_ids.append(upload.id)
                print(f"📤 New upload created: {file.name} ({file_size:.1f}MB)")

        # Process new uploads asynchronously
        for upload in new_uploads:
            process_upload.delay(upload.id)

        # Show processing message for large uploads
        if total_size > 50:
            messages.info(
                request,
                f'Processing {len(new_uploads)} large file(s) ({total_size:.1f}MB total). '
                f'This may take several minutes. Please wait...'
            )

        # Wait for processing with progress updates
        if new_uploads:
            max_wait = MAX_WAIT_TIME
            start = time.time()
            last_status_time = start
            
            while time.time() - start < max_wait:
                all_done = True
                processing_status = []
                
                for u in new_uploads:
                    u.refresh_from_db()
                    if u.status not in ['processed', 'failed']:
                        all_done = False
                        processing_status.append(f"{u.filename}: {u.status}")
                
                if all_done:
                    break
                
                # Log progress every 10 seconds
                if time.time() - last_status_time > 10:
                    print(f"⏳ Processing status: {', '.join(processing_status)}")
                    last_status_time = time.time()
                
                time.sleep(3)
            
            # Check for failures
            failed = [u for u in new_uploads if u.status == 'failed']
            if failed:
                failed_names = ', '.join([u.filename for u in failed])
                messages.error(request, f"Failed to process: {failed_names}")
            
            # Check for timeouts
            still_processing = [u for u in new_uploads if u.status not in ['processed', 'failed']]
            if still_processing:
                timeout_names = ', '.join([u.filename for u in still_processing])
                messages.warning(
                    request,
                    f"Processing timeout for: {timeout_names}. "
                    f"These files may still be processing in the background."
                )

        # Generate study plan
        print(f"🎯 Generating study plan for {len(upload_ids)} uploads...")
        plan_result = generate_study_plan_sync(request.user, upload_ids, constraint_prompt)
        
        if isinstance(plan_result, dict) and 'error' in plan_result:
            messages.error(request, f"Plan generation failed: {plan_result['error']}")
            request.session['upload_status'] = {'error': plan_result['error']}
        else:
            success_msg = f'Study plan generated for {len(files)} file(s)!'
            if constraint_prompt:
                success_msg += f' (with custom constraints)'
            messages.success(request, success_msg)
            request.session['upload_status'] = {'success': True}

        return redirect('result_page')

    # GET request - show upload form
    user_uploads = Upload.objects.filter(user=request.user, status='processed').order_by('-created_at')[:10]
    return render(request, 'core/upload.html', {'recent_uploads': user_uploads})

# ==================== RESULT PAGE ====================
@login_required
def result_page(request):
    upload_status = request.session.pop('upload_status', None)
    latest_plan = Plan.objects.filter(user=request.user).order_by('-created_at').first()
    
    if latest_plan:
        plan = latest_plan.plan_json
        # Add statistics
        total_files = len([p for p in plan if p.get('file') != 'SYSTEM_MESSAGE'])
        total_chunks = sum(p.get('chunk_count', 0) for p in plan if p.get('file') != 'SYSTEM_MESSAGE')
        total_pages = sum(p.get('pages', 0) for p in plan if p.get('file') != 'SYSTEM_MESSAGE')
        
        stats = {
            'total_files': total_files,
            'total_chunks': total_chunks,
            'total_pages': total_pages,
            'generated_at': latest_plan.created_at
        }
    else:
        plan = [{
            "file": "No Plan Available",
            "priority": 0,
            "reason": "Upload PDF files to generate a personalized study plan."
        }]
        stats = None
    
    return render(request, 'core/result.html', {
        'plan': plan,
        'stats': stats,
        'upload_status': upload_status
    })

# ==================== AUTH VIEWS ====================
def register_page(request):
    form = RegistrationForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        messages.success(request, f'Account created for {user.username}! Please log in.')
        return redirect('login_page')
    return render(request, 'core/register.html', {'form': form})

def login_page(request):
    form = LoginForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        username = form.cleaned_data.get('username')
        password = form.cleaned_data.get('password')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            messages.success(request, f'Welcome back, {username}!')
            return redirect('upload_page')
        messages.error(request, 'Invalid username or password.')
    return render(request, 'core/login.html', {'form': form})

@login_required
def logout_view(request):
    logout(request)
    messages.success(request, 'You have been logged out.')
    return redirect('login_page')