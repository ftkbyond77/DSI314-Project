# core/tasks_agentic_optimized.py - Optimized Async Tasks

from celery import shared_task
from .models import Upload, Chunk, Plan
from .agent_tools_advanced import (
    EnhancedTaskAnalysisTool,
    FlexibleSchedulingTool,
    ToolLogger
)
from .llm_config import llm
from langchain_core.messages import HumanMessage, SystemMessage
import json
import time
from typing import Dict, List

# ==================== OPTIMIZED PLANNING TASK ====================

def convert_time_to_hours(time_dict: dict) -> float:
    total = 0.0
    total += int(time_dict.get("years") or 0) * 365 * 24
    total += int(time_dict.get("months") or 0) * 30 * 24
    total += int(time_dict.get("weeks") or 0) * 7 * 24
    total += int(time_dict.get("days") or 0) * 24
    total += int(time_dict.get("hours") or 0)
    return total

def generate_reasoning_fast(tasks: List[Dict], schedule_data: Dict, 
                            user_goal: str, sort_method: str) -> Dict:
    """
    Generate human-readable reasoning with single LLM call.
    Fast and efficient - only for explanation, not decision-making.
    """
    try:
        # Build concise prompt
        task_summary = []
        for task in tasks[:5]:  # Top 5 only
            task_summary.append(
                f"{task['priority']}. {task['task']} - "
                f"{task['analysis']['category']}, "
                f"{task['analysis']['pages']} pages, "
                f"complexity {task['analysis']['complexity']}/10"
            )
        
        prompt = f"""You are an AI study planner. Explain the prioritization decisions briefly.

    User Goal: {user_goal}
    Sort Method: {sort_method}
    Top Tasks:
    {chr(10).join(task_summary)}

    Provide:
    1. Brief explanation (2-3 sentences) of why these priorities make sense
    2. One-line reasoning for each of the top 3 tasks

    Be concise, clear, and focus on the logical connections."""
        
        messages = [
            SystemMessage(content="You are a concise study planning assistant."),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        explanation = response.content.strip()
        
        # Parse task-specific reasoning
        task_reasoning = {}
        for task in tasks[:3]:
            task_reasoning[task["task"]] = f"Priority {task['priority']}: {sort_method.title()} method - {task['analysis']['category']} content with {task['analysis']['urgency_score']}/10 urgency"
        
        return {
            "full_explanation": explanation,
            "schedule": schedule_data.get("reasoning", "Optimized weekly schedule"),
            "tasks": task_reasoning,
            "method": sort_method
        }
        
    except Exception as e:
        print(f"⚠️ Reasoning generation failed: {e}")
        # Fallback to rule-based reasoning
        return {
            "full_explanation": f"Tasks prioritized using {sort_method} method based on your goal: {user_goal}",
            "schedule": schedule_data.get("reasoning", "Weekly schedule created"),
            "tasks": {
                task["task"]: f"Priority {task['priority']} - {task['analysis']['category']}"
                for task in tasks[:3]
            },
            "method": sort_method
        }

@shared_task(bind=True, max_retries=2, time_limit=600)
def generate_optimized_plan_async(self, user_id, upload_ids, user_goal, 
                                  time_input, constraints, sort_method):
    """
    Ultra-optimized agentic planning with minimal LLM calls.
    Uses tool-based analysis for speed, LLM only for final reasoning.
    """
    try:
        print(f"🚀 [Task {self.request.id}] Starting optimized planning")
        start_time = time.time()
        
        # Clear logs
        ToolLogger.clear_logs()
        
        # Update state
        self.update_state(
            state='PROCESSING',
            meta={'current': 1, 'total': 4, 'status': 'Gathering documents...'}
        )
        
        # Get uploads
        from django.contrib.auth import get_user_model
        User = get_user_model()
        
        uploads = Upload.objects.filter(
            id__in=upload_ids,
            user_id=user_id,
            status='processed'
        ).select_related('user')
        
        if not uploads.exists():
            return {"success": False, "error": "No processed uploads found"}
        
        print(f"   Found {uploads.count()} processed uploads")
        
        # Phase 1: Fast Task Analysis (Tool-based, no LLM)
        self.update_state(
            state='PROCESSING',
            meta={'current': 2, 'total': 4, 'status': 'Analyzing tasks...'}
        )
        
        task_tool = EnhancedTaskAnalysisTool()
        analyzed_tasks = []
        
        for upload in uploads:
            # Get representative sample (fast)
            chunks = list(Chunk.objects.filter(upload=upload).order_by('start_page')[:2])
            summary = " ".join([chunk.text[:200] for chunk in chunks])
            
            metadata = {
                "pages": upload.pages or 0,
                "chunk_count": Chunk.objects.filter(upload=upload).count(),
                "deadline": None  # Can be extended
            }
            
            # Run tool analysis
            analysis_result = task_tool._run(
                task_name=upload.filename,
                content_summary=summary,
                metadata=metadata,
                sort_preference=sort_method
            )
            
            analyzed_tasks.append(json.loads(analysis_result))
        
        print(f"   Analyzed {len(analyzed_tasks)} tasks in {time.time() - start_time:.2f}s")
        
        # Phase 2: Sort by preferred method (Fast, no LLM)
        self.update_state(
            state='PROCESSING',
            meta={'current': 3, 'total': 4, 'status': 'Prioritizing tasks...'}
        )
        
        # Sort by preferred score
        analyzed_tasks.sort(
            key=lambda x: x["analysis"]["preferred_score"],
            reverse=True
        )
        
        # Assign priorities
        for idx, task in enumerate(analyzed_tasks, 1):
            task["priority"] = idx
        
        print(f"   Prioritized tasks using '{sort_method}' method")
        
        # Phase 3: Generate Schedule (Tool-based, no LLM)
        self.update_state(
            state='PROCESSING',
            meta={'current': 4, 'total': 4, 'status': 'Creating schedule...'}
        )
        
        # Check if user provided any time
        total_hours_available = convert_time_to_hours(time_input)
        
        if total_hours_available > 0:
            scheduling_tool = FlexibleSchedulingTool()
            schedule_result = scheduling_tool._run(
                prioritized_tasks=json.dumps(analyzed_tasks),
                available_time=time_input,
                constraints=constraints,
                sort_method=sort_method
            )
            
            schedule_data = json.loads(schedule_result)
            schedule = schedule_data.get("schedule", [])
            
            print(f"   Created schedule with {len(schedule)} items")
        else:
            # No time provided - skip scheduling
            schedule_data = {
                "schedule": [],
                "reasoning": "No time availability provided. Tasks are prioritized but not scheduled.",
                "total_allocated_hours": 0,
                "utilization_percent": 0
            }
            schedule = []
            print(f"   No time provided - skipping schedule creation")
        
        # Phase 4: Generate Reasoning (Single LLM call for explanations)
        reasoning = generate_reasoning_fast(
            analyzed_tasks,
            schedule_data,
            user_goal,
            sort_method
        )
        
        # Build final plan
        plan_json = []
        
        # Add schedule
        if schedule:
            plan_json.append({
                "file": "📅 WEEKLY SCHEDULE",
                "priority": 0,
                "reason": reasoning.get("schedule", "AI-generated schedule"),
                "schedule": schedule,
                "metadata": {
                    "type": "schedule",
                    "total_hours": schedule_data.get("total_allocated_hours", 0),
                    "utilization_percent": schedule_data.get("utilization_percent", 0),
                    "sort_method": sort_method
                }
            })
        
        # Add tasks
        for task in analyzed_tasks:
            analysis = task["analysis"]
            plan_json.append({
                "file": task["task"],
                "priority": task["priority"],
                "reason": reasoning.get("tasks", {}).get(task["task"], "AI analysis"),
                "chunk_count": analysis.get("chunks", 0),
                "pages": analysis.get("pages", 0),
                "estimated_hours": analysis.get("estimated_hours", 0),
                "category": analysis.get("category", "general"),
                "complexity": analysis.get("complexity", 5),
                "urgency": analysis.get("urgency_score", 5)
            })
        
        # Save to database
        Plan.objects.filter(user_id=user_id, upload__id__in=upload_ids).delete()
        
        plan = Plan.objects.create(
            user_id=user_id,
            upload=uploads.first(),
            plan_json=plan_json
        )
        
        # Get metrics
        tool_logs = ToolLogger.get_logs()
        tool_summary = ToolLogger.get_summary()
        total_time = time.time() - start_time
        
        print(f"✅ [Task {self.request.id}] Completed in {total_time:.2f}s")
        print(f"   Tool calls: {tool_summary.get('total_calls', 0)}")
        print(f"   Avg tool time: {tool_summary.get('avg_time_per_call', 0):.2f}ms")
        
        return {
            "success": True,
            "plan_id": plan.id,
            "total_tasks": len(analyzed_tasks),
            "schedule_items": len(schedule),
            "execution_time": round(total_time, 2),
            "tool_logs": tool_logs[-10:],  # Last 10 logs
            "tool_summary": tool_summary,
            "reasoning": reasoning.get("full_explanation", "")
        }
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        
        print(f"❌ [Task {self.request.id}] Failed: {str(e)}")
        print(error_trace)
        
        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            print(f"   Retrying ({self.request.retries + 1}/{self.max_retries})...")
            raise self.retry(exc=e, countdown=5 * (2 ** self.request.retries))
        
        return {
            "success": False,
            "error": str(e),
            "error_trace": error_trace[:500],
            "tool_logs": ToolLogger.get_logs()[-5:]
        }
    
    # def _convert_time_to_hours(self, time_dict: Dict) -> float:
    #     """Convert flexible time to total hours"""
    #     total = 0.0
    #     total += time_dict.get("years", 0) * 365 * 24
    #     total += time_dict.get("months", 0) * 30 * 24
    #     total += time_dict.get("weeks", 0) * 7 * 24
    #     total += time_dict.get("days", 0) * 24
    #     total += time_dict.get("hours", 0)
    #     return total


# ==================== BATCH PROCESSING FOR LARGE UPLOADS ====================

@shared_task(bind=True)
def process_large_upload_batch(self, upload_id, batch_start, batch_end):
    """
    Process a batch of pages for very large PDFs.
    Enables chunked processing to avoid memory issues.
    """
    try:
        from .pdf_utils import extract_text_from_pdf, chunk_text, sanitize_text
        from .llm_config import embeddings, INDEX_NAME
        from langchain_pinecone import PineconeVectorStore
        import os
        
        upload = Upload.objects.get(id=upload_id)
        
        print(f"📄 Processing batch {batch_start}-{batch_end} of {upload.filename}")
        
        # Extract text for this batch only
        text_data = extract_text_from_pdf(upload.file.path)
        pages_batch = text_data["pages"][batch_start:batch_end]
        
        # Chunk this batch
        batch_text_data = {
            "pages": pages_batch,
            "total_pages": len(pages_batch)
        }
        chunks = chunk_text(batch_text_data)
        
        # Initialize vector store
        vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        
        # Process chunks
        batch_texts = []
        batch_ids = []
        batch_metadatas = []
        
        for i, chunk_data in enumerate(chunks):
            chunk_text = sanitize_text(chunk_data["text"])
            
            if not chunk_text or len(chunk_text) < 10:
                continue
            
            chunk_id = f"{upload_id}_{batch_start}_{i}"
            
            # Save to database
            Chunk.objects.create(
                upload=upload,
                chunk_id=chunk_id,
                text=chunk_text,
                start_page=chunk_data["start_page"],
                end_page=chunk_data["end_page"]
            )
            
            batch_texts.append(chunk_text)
            batch_ids.append(chunk_id)
            batch_metadatas.append({
                "upload_id": upload_id,
                "file": upload.filename,
                "start_page": chunk_data["start_page"],
                "end_page": chunk_data["end_page"]
            })
        
        # Add to vector store
        if batch_texts:
            vector_store.add_texts(
                texts=batch_texts,
                ids=batch_ids,
                metadatas=batch_metadatas
            )
        
        print(f"✅ Batch complete: {len(batch_texts)} chunks")
        
        return {
            "success": True,
            "chunks_processed": len(batch_texts),
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