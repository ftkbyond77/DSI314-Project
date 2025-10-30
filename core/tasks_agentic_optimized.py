# core/tasks_agentic_optimized.py - FINAL PRODUCTION
# LLM-Based Ranking + Full Async + Validation After Sorting

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


def _determine_task_type(task_name: str, task_analysis: Dict) -> str:
    """Determine study activity type."""
    name_lower = task_name.lower()
    category = task_analysis.get('category', '').lower()
    complexity = task_analysis.get('complexity', 5)
    
    if any(word in name_lower for word in ['exam', 'test', 'quiz', 'assessment', 'midterm', 'final']):
        return "Exam Prep"
    elif any(word in name_lower for word in ['assignment', 'homework', 'project', 'submission']):
        return "Assignment"
    elif any(word in name_lower for word in ['lab', 'practical', 'experiment', 'hands-on']):
        return "Practical"
    elif any(word in name_lower for word in ['review', 'revision', 'summary', 'recap']):
        return "Review"
    elif any(word in name_lower for word in ['workshop', 'tutorial', 'exercise', 'practice']):
        return "Workshop"
    elif complexity >= 7 or 'theory' in category or 'concept' in category:
        return "Theory"
    elif complexity <= 4:
        return "Review"
    else:
        return "Theory"


# ==================== LLM-BASED RANKING SYSTEM ====================

def _prepare_tasks_for_llm_ranking(tasks: List[Dict], sort_method: str) -> List[Dict]:
    """
    Prepare tasks with initial scores but LET LLM DO FINAL RANKING.
    
    This provides guidance scores but LLM makes final decisions.
    """
    
    print(f"   🤖 Preparing {len(tasks)} tasks for LLM-based ranking...")
    
    # Calculate guidance scores (NOT final priorities)
    for task in tasks:
        analysis = task['analysis']
        kb_grounding = analysis.get('knowledge_grounding', {})
        
        urgency = analysis.get('urgency_score', 5)
        complexity = analysis.get('complexity', 5)
        pages = analysis.get('pages', 0)
        foundational = 10 if analysis.get('is_foundational', False) else 0
        kb_gap = (1 - kb_grounding.get('knowledge_relevance_score', 0.5)) * 10
        
        # Calculate guidance scores based on method
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
    
    # Sort by guidance score (initial ordering for LLM)
    tasks_sorted = sorted(tasks, key=lambda t: (-t['guidance_score'], -t['analysis'].get('pages', 0), t['task']))
    
    # Assign TEMPORARY priorities (LLM will reassign)
    for idx, task in enumerate(tasks_sorted, 1):
        task['temp_priority'] = idx
    
    print(f"   ✅ Tasks prepared with guidance scores (LLM will determine final priorities)")
    
    return tasks_sorted


# ==================== VALIDATION FUNCTIONS ====================

def _validate_priorities(tasks: List[Dict]) -> Tuple[bool, str]:
    """
    Validate that priorities are sequential (1, 2, 3, ..., N).
    
    Returns: (is_valid, error_message)
    """
    if not tasks:
        return True, ""
    
    priorities = sorted([task.get('priority', 999) for task in tasks])
    expected = list(range(1, len(tasks) + 1))
    
    if priorities != expected:
        error_msg = f"Priority validation FAILED! Expected: {expected}, Got: {priorities}"
        
        # Show duplicates
        from collections import Counter
        duplicates = [p for p, count in Counter(priorities).items() if count > 1]
        if duplicates:
            error_msg += f" | Duplicates: {duplicates}"
        
        # Show missing
        missing = set(expected) - set(priorities)
        if missing:
            error_msg += f" | Missing: {sorted(missing)}"
        
        return False, error_msg
    
    return True, ""


def _force_sequential_priorities(tasks: List[Dict]) -> List[Dict]:
    """
    Force sequential priorities if validation fails.
    
    Sorts by existing priority, then reassigns 1, 2, 3...
    """
    print(f"   🔧 Forcing sequential priorities...")
    
    # Sort by existing priority
    sorted_tasks = sorted(tasks, key=lambda t: (t.get('priority', 999), t['task']))
    
    # Reassign sequential
    for idx, task in enumerate(sorted_tasks, 1):
        old_priority = task.get('priority', 999)
        task['priority'] = idx
        if old_priority != idx:
            print(f"      Corrected: {task['task'][:30]}... ({old_priority} → {idx})")
    
    return sorted_tasks


# ==================== PRIORITY EXTRACTION WITH VALIDATION ====================

def _extract_llm_priorities(prepared_tasks: List[Dict], reasoning: Dict) -> List[Dict]:
    """
    Extract LLM-assigned priorities from reasoning XML with validation.
    
    LLM's priorities in XML override guidance scores.
    Validates after extraction and forces correction if needed.
    """
    
    print(f"   🔍 Extracting LLM-assigned priorities...")
    
    task_reasoning = reasoning.get("tasks", {})
    extraction_success = 0
    
    for task in prepared_tasks:
        task_name = task['task']
        
        if task_name in task_reasoning:
            reasoning_text = task_reasoning[task_name]
            
            # Extract priority from XML
            priority_match = re.search(r'<task_analysis\s+priority="(\d+)">', reasoning_text)
            if priority_match:
                llm_priority = int(priority_match.group(1))
                task['priority'] = llm_priority
                extraction_success += 1
                print(f"      {task_name[:30]}... → Priority #{llm_priority} (LLM)")
            else:
                # Fallback to temp priority
                task['priority'] = task.get('temp_priority', 999)
                print(f"      {task_name[:30]}... → Priority #{task['priority']} (fallback)")
        else:
            task['priority'] = task.get('temp_priority', 999)
    
    print(f"   📊 Extracted {extraction_success}/{len(prepared_tasks)} priorities from LLM")
    
    # CRITICAL: Validate priorities after extraction
    print(f"   🔍 Validating priorities after LLM extraction...")
    is_valid, error_msg = _validate_priorities(prepared_tasks)
    
    if not is_valid:
        print(f"   ⚠️ {error_msg}")
        print(f"   🔧 Applying automatic correction...")
        
        # Force sequential priorities
        prepared_tasks = _force_sequential_priorities(prepared_tasks)
        
        # Validate again
        is_valid_after, error_msg_after = _validate_priorities(prepared_tasks)
        if is_valid_after:
            print(f"   ✅ Priorities corrected successfully")
        else:
            print(f"   ❌ Priority correction failed: {error_msg_after}")
            raise ValueError(f"Unable to assign valid priorities: {error_msg_after}")
    else:
        print(f"   ✅ Priorities validated: Sequential 1-{len(prepared_tasks)}")
    
    # Sort by validated priority
    final_tasks = sorted(prepared_tasks, key=lambda t: t['priority'])
    
    return final_tasks


# ==================== OPTIMIZED PLANNING TASK ====================

@shared_task(bind=True, max_retries=2, time_limit=600)
def generate_optimized_plan_async(self, user_id, upload_ids, user_goal, 
                                  time_input, constraints, sort_method):
    """
    FINAL PRODUCTION: LLM-based ranking + Full async + Validation.
    
    Key Features:
    - LLM does the ranking (not algorithmic sort)
    - Full async processing (no ThreadPoolExecutor)
    - High-quality, meaningful reasoning with long context
    - VALIDATION AFTER SORTING (ensures sequential priorities)
    - Fast recall, high accuracy, production-grade
    """
    try:
        print(f"🚀 [Task {self.request.id}] Starting LLM-based planning (PRODUCTION)")
        start_time = time.time()
        ToolLogger.clear_logs()

        from django.contrib.auth import get_user_model
        User = get_user_model()
        
        # ==================== Phase 1: Get Uploads ====================
        uploads = Upload.objects.filter(
            id__in=upload_ids,
            user_id=user_id,
            status='processed'
        ).select_related('user')
        
        if not uploads.exists():
            return {"success": False, "error": "No processed uploads found"}

        print(f"   📁 Found {uploads.count()} processed uploads")

        # ==================== Phase 2: Task Analysis with KB Grounding ====================
        self.update_state(
            state='PROCESSING',
            meta={'current': 1, 'total': 5, 'status': 'Analyzing tasks with KB grounding...'}
        )
        
        task_tool = EnhancedTaskAnalysisTool()
        analyzed_tasks = []
        
        for upload in uploads:
            # Get content for KB comparison
            chunks = list(Chunk.objects.filter(upload=upload).order_by('start_page')[:3])
            content = " ".join([chunk.text[:500] for chunk in chunks])
            
            metadata = {
                "pages": upload.pages or 0, 
                "chunk_count": Chunk.objects.filter(upload=upload).count(), 
                "deadline": None,
                "source_type": "textbook"
            }
            
            # Analysis with KB grounding enabled
            analysis_result = task_tool._run(
                task_name=upload.filename, 
                content_summary=content, 
                metadata=metadata, 
                sort_preference=sort_method,
                use_knowledge_grounding=True
            )
            
            analyzed_tasks.append(json.loads(analysis_result))

        print(f"   ✅ Analyzed {len(analyzed_tasks)} tasks with KB grounding")

        # ==================== Phase 3: PREPARE FOR LLM RANKING ====================
        self.update_state(
            state='PROCESSING',
            meta={'current': 2, 'total': 5, 'status': 'Preparing for LLM-based ranking...'}
        )
        
        # Prepare tasks with guidance scores (LLM will do final ranking)
        prepared_tasks = _prepare_tasks_for_llm_ranking(analyzed_tasks, sort_method)
        
        print(f"   🤖 Tasks ready for LLM ranking (method: {sort_method})")

        # ==================== Phase 4: LLM RANKING + REASONING ====================
        self.update_state(
            state='PROCESSING',
            meta={'current': 3, 'total': 5, 'status': 'LLM generating rankings & reasoning...'}
        )
        
        # CRITICAL: LLM does ranking AND reasoning (full async)
        reasoning = generate_knowledge_grounded_reasoning(
            tasks=prepared_tasks,  # Tasks with guidance scores
            schedule_data={"schedule": [], "available_hours": 0},  # Temporary, will regenerate
            user_goal=user_goal,
            sort_method=sort_method
        )
        
        # Extract LLM-assigned priorities with VALIDATION
        final_tasks = _extract_llm_priorities(prepared_tasks, reasoning)
        
        print(f"   ✅ LLM ranking complete with validated priorities")

        # ==================== Phase 5: Generate Schedule ====================
        self.update_state(
            state='PROCESSING',
            meta={'current': 4, 'total': 5, 'status': 'Creating optimized schedule...'}
        )
        
        total_hours_available = _convert_time_to_hours(time_input)
        
        if total_hours_available > 0:
            scheduling_tool = FlexibleSchedulingTool()
            schedule_result = scheduling_tool._run(
                prioritized_tasks=json.dumps(final_tasks),  # Use validated priorities
                available_time=time_input,
                constraints=constraints,
                sort_method=sort_method
            )
            schedule_data = json.loads(schedule_result)
            schedule_data['available_hours'] = total_hours_available
            schedule_data['constraints'] = constraints
            schedule = schedule_data.get("schedule", [])
            
            # Regenerate schedule reasoning with final priorities
            reasoning_updated = generate_knowledge_grounded_reasoning(
                tasks=final_tasks,
                schedule_data=schedule_data,
                user_goal=user_goal,
                sort_method=sort_method
            )
            reasoning = reasoning_updated  # Use updated reasoning
            
            print(f"   ✅ Created schedule with {len(schedule)} items")
        else:
            schedule_data = {
                "schedule": [], 
                "reasoning": "No time availability provided.",
                "total_allocated_hours": 0, 
                "utilization_percent": 0,
                "available_hours": 0,
                "constraints": ""
            }
            schedule = []
            print(f"   ⚠️ No time provided - skipping schedule")

        # ==================== Phase 6: Build plan_json ====================
        self.update_state(
            state='PROCESSING',
            meta={'current': 5, 'total': 5, 'status': 'Finalizing plan...'}
        )
        
        total_time = time.time() - start_time
        
        plan_json = []
        
        # Add schedule section
        if schedule:
            plan_json.append({
                "file": "📅 WEEKLY SCHEDULE",
                "priority": 0,
                "reason": reasoning.get("schedule", "Optimized schedule"),
                "schedule": schedule,
                "metadata": {
                    "type": "schedule",
                    "total_hours": schedule_data.get("total_allocated_hours", 0),
                    "utilization_percent": schedule_data.get("utilization_percent", 0),
                    "sort_method": sort_method,
                    "kb_grounded": reasoning.get("kb_grounded", False),
                    "llm_ranked": True,
                    "priorities_validated": True
                }
            })
        
        # Add LLM-prioritized tasks with reasoning
        for task in final_tasks:
            analysis = task["analysis"]
            kb_grounding = analysis.get('knowledge_grounding', {})
            
            plan_json.append({
                "file": task["task"],
                "priority": task["priority"],  # LLM-assigned + validated priority
                "reason": reasoning.get("tasks", {}).get(task["task"], "AI analysis"),
                "chunk_count": analysis.get("chunks", 0),
                "pages": analysis.get("pages", 0),
                "estimated_hours": analysis.get("estimated_hours", 0),
                "category": analysis.get("category", "general"),
                "complexity": analysis.get("complexity", 5),
                "urgency": analysis.get("urgency_score", 5),
                "guidance_score": task.get("guidance_score", 0),
                # KB metrics
                "kb_relevance": kb_grounding.get("knowledge_relevance_score", 0.5),
                "kb_confidence": kb_grounding.get("confidence", 0),
                "kb_depth": kb_grounding.get("knowledge_depth", "unknown"),
                "knowledge_adjusted": analysis.get("knowledge_adjusted_score") is not None,
                "llm_ranked": True,
                "priorities_validated": True
            })

        # ==================== Phase 7: Save to Database ====================
        plan = Plan.objects.create(
            user_id=user_id,
            upload=uploads.first(),  
            version=Plan.objects.filter(user_id=user_id).count() + 1,
            plan_json=plan_json
        )
        
        history = StudyPlanHistory.objects.create(
            user_id=user_id,
            plan_json=plan_json,                
            user_goal=user_goal,
            sort_method=sort_method,
            constraints=json.dumps(constraints), 
            time_input=json.dumps(time_input),
            total_hours=_convert_time_to_hours(time_input),
            execution_time=total_time,
            total_files=len(uploads),
            total_pages=sum(u.pages or 0 for u in uploads),
            total_chunks=sum(Chunk.objects.filter(upload=u).count() for u in uploads),
            ocr_pages_total=sum(u.ocr_pages for u in uploads)
        )

        history.uploads.set(uploads)
        
        tool_logs = ToolLogger.get_logs()
        tool_summary = ToolLogger.get_summary()
        
        # ==================== Phase 8: Calculate Statistics ====================
        kb_stats = {
            "avg_confidence": sum(
                t.get('analysis', {}).get('knowledge_grounding', {}).get('confidence', 0)
                for t in final_tasks
            ) / len(final_tasks) if final_tasks else 0,
            "tasks_with_kb": sum(
                1 for t in final_tasks 
                if t.get('analysis', {}).get('knowledge_grounding', {}).get('confidence', 0) > 0.3
            ),
            "avg_kb_relevance": sum(
                t.get('analysis', {}).get('knowledge_grounding', {}).get('knowledge_relevance_score', 0)
                for t in final_tasks
            ) / len(final_tasks) if final_tasks else 0,
            "kb_grounded_reasoning": reasoning.get("kb_grounded", False),
            "llm_ranked": True,
            "priorities_validated": True
        }
        
        print(f"✅ [Task {self.request.id}] COMPLETED in {total_time:.2f}s")
        print(f"   📊 Summary:")
        print(f"      Tasks: {len(final_tasks)} (LLM-ranked + validated)")
        print(f"      Schedule: {len(schedule)} items")
        print(f"      KB-grounded: {kb_stats['tasks_with_kb']}/{len(final_tasks)}")
        print(f"      Avg KB confidence: {kb_stats['avg_confidence']:.3f}")
        print(f"      Sort method: {sort_method}")
        print(f"      Ranking: LLM-based ✓")
        print(f"      Validation: Passed ✓")
        
        return {
            "success": True, 
            "plan_id": plan.id, 
            "history_id": history.id, 
            "total_time": round(total_time, 2),
            "total_tasks": len(final_tasks),
            "schedule_items": len(schedule),
            "execution_time": round(total_time, 2),
            "tool_logs": tool_logs[-10:],
            "tool_summary": tool_summary,
            "kb_statistics": kb_stats,
            "reasoning": reasoning.get("full_explanation", ""),
            "sort_method": sort_method,
            "llm_ranked": True,
            "priorities_validated": True,
            "validation_log": reasoning.get("validation_log", [])
        }

    except Exception as exc:
        import traceback
        error_trace = traceback.format_exc()
        
        print(f"❌ [Task {self.request.id}] FAILED: {str(exc)}")
        print(error_trace)
        
        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            print(f"   🔄 Retrying ({self.request.retries + 1}/{self.max_retries})...")
            raise self.retry(exc=exc, countdown=5 * (2 ** self.request.retries))
        
        return {
            "success": False, 
            "error": str(exc),
            "error_trace": error_trace[:500],
            "tool_logs": ToolLogger.get_logs()[-5:]
        }


# ==================== BATCH PROCESSING FOR LARGE UPLOADS ====================

@shared_task(bind=True)
def process_large_upload_batch(self, upload_id, batch_start, batch_end, batch_size=50):
    """Optimized batch processing for very large PDFs."""
    try:
        from .pdf_utils import extract_text_from_pdf, chunk_text, sanitize_text
        from .llm_config import embeddings, INDEX_NAME
        from langchain_pinecone import PineconeVectorStore
        import os

        upload = Upload.objects.get(id=upload_id)
        print(f"📄 Processing batch {batch_start}-{batch_end} of {upload.filename}")

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

        print(f"✅ Batch complete: {len(vector_texts)} chunks")

        return {
            "success": True,
            "chunks_processed": len(vector_texts),
            "batch_range": f"{batch_start}-{batch_end}"
        }

    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return {"success": False, "error": str(e)}


# ==================== CLEANUP TASKS ====================

@shared_task
def cleanup_old_logs():
    """Clean up old logs (run daily)"""
    ToolLogger.clear_logs()
    print("🧹 Agent logs cleaned")


@shared_task
def cleanup_failed_uploads():
    """Clean up failed uploads older than 24 hours"""
    from django.utils import timezone
    from datetime import timedelta
    
    cutoff = timezone.now() - timedelta(hours=24)
    failed = Upload.objects.filter(
        status='failed',
        created_at__lt=cutoff
    )
    
    count = failed.count()
    failed.delete()
    
    print(f"🧹 Cleaned {count} failed uploads")
    return {"deleted": count}