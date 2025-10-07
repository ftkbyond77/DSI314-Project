# core/views.py — COMPLETE RAG + Pinecone VERSION
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
import json, os, re, time, warnings

warnings.filterwarnings("ignore", message=".*pydantic_v1.*")
User = get_user_model()

# ------------------ PROMPT TEMPLATE ------------------
PROMPT_TEMPLATE = """You are an AI study planner. Analyze the document chunks below and create a prioritized study plan.

Each file has a number of content chunks (indicating its depth). More chunks may indicate broader or more detailed content.

Retrieved Documents:
{context}

Your task: Generate a JSON array ranking PDF files by study priority. Each entry must include:
- file: the PDF filename
- priority: integer where 1 = highest priority
- reason: short justification

Prioritization rules:
1. More chunks may indicate higher coverage → possibly higher priority
2. Fundamental or technical content ranks higher
3. Introductory material before advanced
4. Exam or summary files are high priority
5. Prerequisites before dependent topics

Return ONLY a valid JSON array, no explanations or markdown.
Example: [{{"file": "Finance 101.pdf", "priority": 1, "reason": "Core concepts"}}]
"""

PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "constraint_prompt"])

# ------------------ FORMAT DOCS ------------------
def format_docs(docs):
    """Combine chunks by filename, count chunks, return context for LLM."""
    file_chunks = defaultdict(list)
    for doc in docs:
        meta = doc.metadata
        filename = meta.get("file", "Unknown")
        snippet = doc.page_content.strip()
        if snippet:
            file_chunks[filename].append(snippet)
    # Build preview
    formatted = []
    for fname, chunks in file_chunks.items():
        preview = "\n".join(chunks[:3])  # preview 3 chunks per file
        formatted.append(f"📄 File: {fname}\nChunks: {len(chunks)}\nPreview:\n{preview}\n")
    context = f"Total unique files: {len(file_chunks)}\n\n" + "\n---\n".join(formatted)
    return context, file_chunks

# ------------------ FALLBACK PLAN ------------------
def create_fallback_plan(filenames):
    plan = []
    for filename in filenames:
        priority = 5
        reason = "Standard document"
        lower_name = filename.lower()
        if any(kw in lower_name for kw in ["intro", "fundamental", "basic", "101"]):
            priority, reason = 1, "Introductory content"
        elif any(kw in lower_name for kw in ["exam", "test", "midterm", "final"]):
            priority, reason = 2, "Exam preparation"
        elif any(kw in lower_name for kw in ["data", "statistics", "analysis"]):
            priority, reason = 2, "Core technical content"
        elif any(kw in lower_name for kw in ["finance", "accounting"]):
            priority, reason = 3, "Domain knowledge"
        elif any(kw in lower_name for kw in ["case", "report", "application"]):
            priority, reason = 4, "Applied studies"
        plan.append({"file": filename, "priority": priority, "reason": reason})
    plan.sort(key=lambda x: (x["priority"], x["file"]))
    for i, item in enumerate(plan, 1):
        item["priority"] = i
    return plan

# ------------------ GENERATE STUDY PLAN ------------------
def generate_study_plan_sync(user, upload_ids, constraint_prompt=""):
    try:
        uploads = Upload.objects.filter(id__in=upload_ids, user=user, status="processed")
        if not uploads.exists():
            return {"error": "No processed uploads found"}

        filenames = [u.filename for u in uploads]

        # Connect Pinecone
        vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": min(15, len(filenames)*4),
                "filter": {"upload_id": {"$in": [up.id for up in uploads]}}
            }
        )

        # Retrieve & format docs
        def retrieve_and_format(_):
            docs = retriever.invoke("study plan")
            return format_docs(docs)[0]

        constraint_text = f"Additional constraint: {constraint_prompt}" if constraint_prompt else "No additional constraints."

        # Build RAG chain
        rag_chain = ({"context": RunnableLambda(retrieve_and_format), "constraint_prompt": lambda _: constraint_text}
                     | PROMPT | llm | StrOutputParser())

        result = rag_chain.invoke("study plan")

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
        except Exception:
            plan_json = create_fallback_plan(filenames)
            plan_json.insert(0, {"file": "SYSTEM_MESSAGE", "priority": 0, "reason": "LLM returned invalid JSON. Using fallback."})

        # Merge duplicates
        merged = {}
        for entry in plan_json:
            fname = entry.get("file")
            if not fname: continue
            if fname not in merged:
                merged[fname] = entry
            else:
                merged[fname]["reason"] += " / " + entry.get("reason", "")
                merged[fname]["priority"] = min(merged[fname].get("priority", 999), entry.get("priority", 999))

        # Recalculate priority based on chunk count
        docs = retriever.invoke("study plan")
        _, file_chunks = format_docs(docs)
        for fname, entry in merged.items():
            chunk_bonus = len(file_chunks.get(fname, []))
            entry["chunk_count"] = chunk_bonus
            entry["adjusted_priority"] = entry["priority"] - chunk_bonus*0.1

        # Add missing uploads
        for up in uploads:
            if up.filename not in merged:
                merged[up.filename] = {"file": up.filename, "priority": 999, "chunk_count": len(file_chunks.get(up.filename, [])), "reason": "Not referenced by LLM", "adjusted_priority": 999}

        # Final sort
        final_plan = sorted(merged.values(), key=lambda x: x.get("adjusted_priority", 999))

        # Save plan
        Plan.objects.filter(user_id=user.id, upload__id__in=upload_ids).delete()
        Plan.objects.create(user_id=user.id, upload=uploads.first(), plan_json=final_plan)

        return final_plan

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Failed to generate plan: {str(e)}"}

# ------------------ UPLOAD VIEW ------------------
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

        upload_ids, new_uploads = [], []
        for file in files:
            if not file.name.lower().endswith('.pdf'):
                messages.error(request, f'Only PDF files allowed: {file.name}')
                return render(request, 'core/upload.html', {})
            existing = Upload.objects.filter(user=request.user, filename=file.name, status='processed').first()
            if existing:
                upload_ids.append(existing.id)
            else:
                upload = Upload.objects.create(user=request.user, file=file, filename=file.name, status='uploaded')
                new_uploads.append(upload)
                upload_ids.append(upload.id)

        for upload in new_uploads:
            process_upload.delay(upload.id)

        # Wait for processing
        max_wait = 180
        start = time.time()
        while new_uploads and time.time() - start < max_wait:
            all_done = True
            for u in new_uploads:
                u.refresh_from_db()
                if u.status not in ['processed', 'failed']:
                    all_done = False
            if all_done: break
            time.sleep(3)

        failed = [u for u in new_uploads if u.status == 'failed']
        if failed:
            messages.error(request, f"Failed: {', '.join([u.filename for u in failed])}")

        plan_result = generate_study_plan_sync(request.user, upload_ids, constraint_prompt)
        if isinstance(plan_result, dict) and 'error' in plan_result:
            messages.error(request, f"Plan generation failed: {plan_result['error']}")
            request.session['upload_status'] = {'error': plan_result['error']}
        else:
            messages.success(request, f'Study plan generated for {len(files)} file(s)!')
            request.session['upload_status'] = {'success': True}

        return redirect('result_page')

    return render(request, 'core/upload.html', {})

# ------------------ RESULT PAGE ------------------
@login_required
def result_page(request):
    upload_status = request.session.pop('upload_status', None)
    latest_plan = Plan.objects.filter(user=request.user).order_by('-created_at').first()
    plan = latest_plan.plan_json if latest_plan else [{"file": "No Plan", "priority": 0, "reason": "Upload files to generate a study plan."}]
    return render(request, 'core/result.html', {'plan': plan})

# ------------------ AUTH VIEWS ------------------
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
