# core/tasks_agentic_optimized.py - FINAL PRODUCTION
# LLM-Based Ranking + Full Async + Validation After Sorting + File Wait Logic

from typing import List, Dict, Tuple
from celery import shared_task
from .models import Upload, Chunk, Plan, StudyPlanHistory
from .agent_tools_advanced import (
    EnhancedTaskAnalysisTool,
    FlexibleSchedulingTool,
    ToolLogger
)
from .llm_config import llm
from langchain_core.messages import HumanMessage, SystemMessage
import json
import time
import re

from .llm_reasoning_integration import generate_knowledge_grounded_reasoning
from .knowledge_weighting import refresh_knowledge_base_cache


# ==================== HELPER FUNCTIONS ====================

def _clean_task_name(task_name: str) -> str:
    """Clean task name for consistency."""
    clean_name = re.sub(r'[^\w\s]', '', task_name)
    return clean_name.strip()


def _convert_time_to_hours(time_dict: Dict) -> float:
    """Convert flexible time to total hours."""
    total = 0.0
    total += time_dict.get("years", 0) * 365 * 24
    total += time_dict.get("months", 0) * 30 * 24
    total += time_dict.get("weeks", 0) * 7 * 24
    total += time_dict.get("days", 0) * 24
    total += time_dict.get("hours", 0)
    return total


# ==================== LLM-BASED RANKING SYSTEM ====================

def _prepare_tasks_for_llm_ranking(tasks: List[Dict], sort_method: str) -> List[Dict]:
    """
    Prepare tasks with initial scores but LET LLM DO FINAL RANKING.
    """
    print(f"   Preparing {len(tasks)} tasks for LLM-based ranking...")
    
    for task in tasks:
        analysis = task['analysis']
        kb_grounding = analysis.get('knowledge_grounding', {})
        
        urgency = analysis.get('urgency_score', 5)
        complexity = analysis.get('complexity', 5)
        pages = analysis.get('pages', 0)
        foundational = 10 if analysis.get('is_foundational', False) else 0
        kb_gap = (1 - kb_grounding.get('knowledge_relevance_score', 0.5)) * 10
        
        if sort_method == 'urgency':
            task['guidance_score'] = urgency
        elif sort_method == 'complexity':
            task['guidance_score'] = complexity
        elif sort_method == 'hybrid':
            task['guidance_score'] = (
                0.30 * urgency +
                0.25 * complexity +
                0.20 * foundational +
                0.15 * kb_gap +
                0.10 * min(pages / 100, 1) * 10
            )
        else:
            task['guidance_score'] = analysis.get('knowledge_adjusted_score', 
                                                 analysis.get('preferred_score', 5))
    
    # Sort by guidance score to give LLM a good starting point
    tasks_sorted = sorted(tasks, key=lambda t: (-t['guidance_score'], -t['analysis'].get('pages', 0), t['task']))
    
    for idx, task in enumerate(tasks_sorted, 1):
        task['temp_priority'] = idx
    
    return tasks_sorted


def _validate_priorities(tasks: List[Dict]) -> Tuple[bool, str]:
    """Validate that priorities are sequential."""
    if not tasks:
        return True, ""
    
    priorities = sorted([task.get('priority', 999) for task in tasks])
    expected = list(range(1, len(tasks) + 1))
    
    if priorities != expected:
        return False, f"Expected: {expected}, Got: {priorities}"
    
    return True, ""


def _force_sequential_priorities(tasks: List[Dict]) -> List[Dict]:
    """Force sequential priorities if validation fails."""
    print(f"   Forcing sequential priorities...")
    sorted_tasks = sorted(tasks, key=lambda t: (t.get('priority', 999), t['task']))
    for idx, task in enumerate(sorted_tasks, 1):
        task['priority'] = idx
    return sorted_tasks


def _extract_llm_priorities(prepared_tasks: List[Dict], reasoning: Dict) -> List[Dict]:
    """Extract LLM-assigned priorities from reasoning XML with validation."""
    print(f"   Extracting LLM-assigned priorities...")
    
    task_reasoning = reasoning.get("tasks", {})
    
    for task in prepared_tasks:
        task_name = task['task']
        
        if task_name in task_reasoning:
            reasoning_text = task_reasoning[task_name]
            priority_match = re.search(r'<task_analysis\s+priority="(\d+)">', reasoning_text)
            if priority_match:
                task['priority'] = int(priority_match.group(1))
            else:
                task['priority'] = task.get('temp_priority', 999)
        else:
            task['priority'] = task.get('temp_priority', 999)
    
    is_valid, _ = _validate_priorities(prepared_tasks)
    
    if not is_valid:
        print(f"   Priorities invalid, applying auto-correction...")
        prepared_tasks = _force_sequential_priorities(prepared_tasks)
    
    final_tasks = sorted(prepared_tasks, key=lambda t: t['priority'])
    return final_tasks


# ==================== OPTIMIZED PLANNING TASK ====================

@shared_task(bind=True, max_retries=2, time_limit=1200)
def generate_optimized_plan_async(self, user_id, upload_ids, user_goal, 
                                  time_input, constraints, sort_method, project_name=None):
    """
    FINAL PRODUCTION: LLM-based ranking + Full async + Validation.
    Includes wait logic for file processing.
    """
    try:
        print(f"[Task {self.request.id}] Starting LLM-based planning (PRODUCTION)")
        start_time = time.time()
        ToolLogger.clear_logs()

        from django.contrib.auth import get_user_model
        User = get_user_model()
        
        # ==================== Phase 0: Wait for Files ====================
        MAX_WAIT_SECONDS = 600
        wait_start = time.time()
        
        print(f"   Waiting for {len(upload_ids)} files to complete processing...")
        
        while True:
            # Check status of all requested uploads
            pending_count = Upload.objects.filter(
                id__in=upload_ids, 
                status__in=['uploaded', 'processing']
            ).count()
            
            if pending_count == 0:
                print(f"   All files processed (or failed). Proceeding.")
                break
                
            if time.time() - wait_start > MAX_WAIT_SECONDS:
                print(f"   Wait timeout reached ({MAX_WAIT_SECONDS}s). Proceeding with available files.")
                break
                
            self.update_state(
                state='PROCESSING',
                meta={
                    'current': 0, 
                    'total': 5, 
                    'status': f'Processing files... ({pending_count} remaining)'
                }
            )
            time.sleep(2)

        # ==================== Phase 1: Get Uploads (FIXED) ====================
        # Fetch ALL uploads to ensure we account for every file the user submitted
        all_uploads = Upload.objects.filter(
            id__in=upload_ids,
            user_id=user_id
        ).select_related('user')
        
        uploads_map = {u.id: u for u in all_uploads}
        
        if not uploads_map:
             return {"success": False, "error": "No uploads found."}

        print(f"   Found {len(uploads_map)} uploads (Total requested: {len(upload_ids)})")

        # ==================== Phase 2: Task Analysis (FIXED) ====================
        self.update_state(
            state='PROCESSING',
            meta={'current': 1, 'total': 5, 'status': 'Analyzing tasks...'}
        )
        
        task_tool = EnhancedTaskAnalysisTool()
        analyzed_tasks = []
        
        # Iterate through the ORIGINAL requested IDs to ensure 1:1 mapping
        for uid in upload_ids:
            upload = uploads_map.get(uid)
            
            # Case 1: File completely missing from DB
            if not upload:
                print(f"   [WARN] Upload ID {uid} missing from DB. Skipping.")
                continue
            
            # Case 2: File Processed Successfully
            if upload.status == 'processed':
                chunks = list(Chunk.objects.filter(upload=upload).order_by('start_page')[:3])
                content = " ".join([chunk.text[:500] for chunk in chunks])
                
                metadata = {
                    "pages": upload.pages or 0, 
                    "chunk_count": Chunk.objects.filter(upload=upload).count(), 
                    "deadline": None,
                    "source_type": "textbook"
                }
                
                analysis_result = task_tool._run(
                    task_name=upload.filename, 
                    content_summary=content, 
                    metadata=metadata, 
                    sort_preference=sort_method,
                    use_knowledge_grounding=True
                )
                analyzed_tasks.append(json.loads(analysis_result))
                
            # Case 3: File Failed or Timed Out (Fallback)
            else:
                print(f"   [WARN] File not processed: {upload.filename} (Status: {upload.status}). Adding fallback.")
                analyzed_tasks.append({
                    "task": upload.filename,
                    "analysis": {
                        "category": "Processing Failed",
                        "complexity": 10, # High complexity to flag it
                        "urgency_score": 10, # High urgency to flag it
                        "pages": 0,
                        "chunks": 0,
                        "estimated_hours": 1,
                        "is_foundational": False,
                        "knowledge_grounding": {
                            "confidence": 0,
                            "knowledge_relevance_score": 0,
                            "knowledge_depth": "none"
                        },
                        "summary": f"File status is '{upload.status}'. content unavailable for analysis."
                    }
                })

        print(f"   Analyzed {len(analyzed_tasks)} tasks (Success + Fallback)")

        # ==================== Phase 3: PREPARE FOR LLM RANKING ====================
        self.update_state(
            state='PROCESSING',
            meta={'current': 2, 'total': 5, 'status': 'Preparing for LLM-based ranking...'}
        )
        
        prepared_tasks = _prepare_tasks_for_llm_ranking(analyzed_tasks, sort_method)

        # ==================== Phase 4: LLM RANKING ====================
        self.update_state(
            state='PROCESSING',
            meta={'current': 3, 'total': 5, 'status': 'LLM generating rankings...'}
        )
        
        reasoning = generate_knowledge_grounded_reasoning(
            tasks=prepared_tasks,
            schedule_data={"schedule": [], "available_hours": 0},
            user_goal=user_goal,
            sort_method=sort_method
        )
        
        final_tasks = _extract_llm_priorities(prepared_tasks, reasoning)

        # ==================== Phase 5: Generate Schedule ====================
        self.update_state(
            state='PROCESSING',
            meta={'current': 4, 'total': 5, 'status': 'Creating schedule...'}
        )
        
        total_hours_available = _convert_time_to_hours(time_input)
        
        if total_hours_available > 0:
            scheduling_tool = FlexibleSchedulingTool()
            schedule_result = scheduling_tool._run(
                prioritized_tasks=json.dumps(final_tasks),
                available_time=time_input,
                constraints=constraints,
                sort_method=sort_method
            )
            schedule_data = json.loads(schedule_result)
            schedule = schedule_data.get("schedule", [])
        else:
            schedule_data = {"total_allocated_hours": 0, "utilization_percent": 0}
            schedule = []

        # ==================== Phase 6: Build plan_json ====================
        self.update_state(
            state='PROCESSING',
            meta={'current': 5, 'total': 5, 'status': 'Finalizing plan...'}
        )
        
        total_time = time.time() - start_time
        plan_json = []
        
        if schedule:
            plan_json.append({
                "file": "WEEKLY SCHEDULE",
                "priority": 0,
                "reason": "Optimized schedule",
                "schedule": schedule,
                "metadata": {
                    "type": "schedule",
                    "total_hours": schedule_data.get("total_allocated_hours", 0),
                    "utilization_percent": schedule_data.get("utilization_percent", 0),
                    "sort_method": sort_method,
                    "kb_grounded": True,
                    "llm_ranked": True
                }
            })
        
        for task in final_tasks:
            analysis = task["analysis"]
            kb = analysis.get('knowledge_grounding', {})
            
            cat = analysis.get("category", "General")
            
            plan_json.append({
                "file": task["task"],
                "priority": task["priority"],
                "reason": reasoning.get("tasks", {}).get(task["task"], "Analysis unavailable"),
                "chunk_count": analysis.get("chunks", 0),
                "pages": analysis.get("pages", 0),
                "estimated_hours": analysis.get("estimated_hours", 0),
                "category": cat,
                "complexity": analysis.get("complexity", 5),
                "urgency": analysis.get("urgency_score", 5),
                "guidance_score": task.get("guidance_score", 0),
                "kb_relevance": kb.get("knowledge_relevance_score", 0),
                "kb_confidence": kb.get("confidence", 0),
                "llm_ranked": True,
                "priorities_validated": True
            })

        # ==================== Phase 7: Save to Database ====================
        final_project_name = project_name
        if not final_project_name or final_project_name.strip() == "":
            count = StudyPlanHistory.objects.filter(user_id=user_id).count()
            final_project_name = f"Plan #{count + 1}"

        valid_uploads = [u for u in all_uploads if u.status == 'processed']

        plan = Plan.objects.create(
            user_id=user_id,
            upload=all_uploads[0] if all_uploads else None,  
            version=1,
            plan_json=plan_json
        )
        
        history = StudyPlanHistory.objects.create(
            user_id=user_id,
            project_name=final_project_name,
            plan_json=plan_json,                
            user_goal=user_goal,
            sort_method=sort_method,
            constraints=json.dumps(constraints), 
            time_input=json.dumps(time_input),
            total_hours=total_hours_available,
            execution_time=total_time,
            total_files=len(all_uploads), # Count ALL uploads
            total_pages=sum(u.pages or 0 for u in valid_uploads),
            total_chunks=sum(Chunk.objects.filter(upload=u).count() for u in valid_uploads),
            ocr_pages_total=sum(u.ocr_pages for u in valid_uploads)
        )

        history.uploads.set(all_uploads)
        
        return {
            "success": True, 
            "plan_id": plan.id, 
            "history_id": history.id, 
            "total_files": len(all_uploads),
            "execution_time": round(total_time, 2)
        }

    except Exception as exc:
        import traceback
        print(f"[Task {self.request.id}] FAILED: {str(exc)}")
        traceback.print_exc()
        return {"success": False, "error": str(exc)}


@shared_task(bind=True)
def process_large_upload_batch(self, upload_id, batch_start, batch_end, batch_size=50):
    """Optimized batch processing for very large PDFs."""
    try:
        from .pdf_utils import extract_text_from_pdf, chunk_text, sanitize_text
        from .llm_config import embeddings, INDEX_NAME
        from langchain_pinecone import PineconeVectorStore
        import os

        upload = Upload.objects.get(id=upload_id)
        print(f"Processing batch {batch_start}-{batch_end} of {upload.filename}")

        text_data = extract_text_from_pdf(upload.file.path)
        pages_batch = text_data["pages"][batch_start:batch_end]
        chunks = chunk_text({"pages": pages_batch, "total_pages": len(pages_batch)})

        all_chunks_data = []
        vector_texts = []
        vector_ids = []
        vector_metadatas = []

        for i, chunk_data in enumerate(chunks):
            chunk_text_sanitized = sanitize_text(chunk_data["text"])
            if not chunk_text_sanitized or len(chunk_text_sanitized) < 10:
                continue

            chunk_id = f"{upload_id}_{batch_start}_{i}"

            all_chunks_data.append(
                Chunk(
                    upload=upload,
                    chunk_id=chunk_id,
                    text=chunk_text_sanitized,
                    start_page=chunk_data["start_page"],
                    end_page=chunk_data["end_page"]
                )
            )

            vector_texts.append(chunk_text_sanitized)
            vector_ids.append(chunk_id)
            vector_metadatas.append({
                "upload_id": upload_id,
                "file": upload.filename,
                "start_page": chunk_data["start_page"],
                "end_page": chunk_data["end_page"]
            })

            if len(all_chunks_data) >= batch_size:
                Chunk.objects.bulk_create(all_chunks_data, batch_size=batch_size)
                all_chunks_data = []

        if all_chunks_data:
            Chunk.objects.bulk_create(all_chunks_data, batch_size=batch_size)

        if vector_texts:
            vector_store = PineconeVectorStore(
                index_name=INDEX_NAME,
                embedding=embeddings,
                pinecone_api_key=os.getenv("PINECONE_API_KEY")
            )
            vector_store.add_texts(
                texts=vector_texts,
                ids=vector_ids,
                metadatas=vector_metadatas
            )

        print(f"Batch complete: {len(vector_texts)} chunks")

        return {
            "success": True,
            "chunks_processed": len(vector_texts),
            "batch_range": f"{batch_start}-{batch_end}"
        }

    except Exception as e:
        print(f"Batch processing failed: {e}")
        return {"success": False, "error": str(e)}


@shared_task
def cleanup_old_logs():
    ToolLogger.clear_logs()


@shared_task
def cleanup_failed_uploads():
    from django.utils import timezone
    from datetime import timedelta
    
    cutoff = timezone.now() - timedelta(hours=24)
    failed = Upload.objects.filter(
        status='failed',
        created_at__lt=cutoff
    )
    
    count = failed.count()
    failed.delete()
    return {"deleted": count}