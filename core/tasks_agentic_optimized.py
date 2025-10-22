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

def _clean_task_name(task_name: str) -> str:
    clean_name = re.sub(r'[^\w\s]', '', task_name)
    clean_name = clean_name.strip()
    return clean_name

# def _generate_reasoning_fast(tasks: List[Dict], schedule_data: Dict, 
#                              user_goal: str, sort_method: str) -> Dict:
#     """
#     Generate comprehensive, detailed reasoning with 10-12 lines per task.
#     Provides structured reasoning with logical flow and professional presentation.
#     """
#     try:
#         # Build enriched task information
#         task_details = []
#         top_5_tasks = tasks[:5]
#         max_urgency = max((t['analysis'].get('urgency_score', 5) for t in top_5_tasks), default=5)
#         max_complexity = max((t['analysis'].get('complexity', 5) for t in top_5_tasks), default=5)
        
#         for idx, task in enumerate(top_5_tasks, 1):
#             analysis = task['analysis']
#             urgency = analysis.get('urgency_score', 5)
#             complexity = analysis.get('complexity', 5)
#             urgency_ratio = (urgency / max_urgency) if max_urgency > 0 else 0
#             complexity_ratio = (complexity / max_complexity) if max_complexity > 0 else 0
            
#             clean_task_name = _clean_task_name(task['task'])
            
#             task_details.append({
#                 'name': clean_task_name,
#                 'category': analysis['category'],
#                 'pages': analysis['pages'],
#                 'complexity': complexity,
#                 'urgency': urgency,
#                 'urgency_ratio': urgency_ratio,
#                 'complexity_ratio': complexity_ratio,
#                 'is_foundational': analysis.get('is_foundational', False),
#                 'estimated_hours': analysis.get('estimated_hours', 0),
#                 'priority': task['priority']
#             })
        
#         # Enhanced system prompt for professional, detailed reasoning
#         system_prompt = """You are an expert academic strategist providing comprehensive, logical study guidance.

# **CRITICAL REQUIREMENTS FOR REASONING:**
# - Generate EXACTLY 10-12 lines of reasoning per task (mix of sentences and bullet points)
# - Use professional, academic language with specific metrics and data
# - Structure reasoning with clear logical flow: Context → Analysis → Impact → Strategy → Execution
# - Include quantitative justifications (percentages, scores, time estimates)
# - Make explicit comparisons between tasks to justify ranking
# - Use bullet points for specific action items and key insights
# - Maintain consistent professional tone throughout

# **REASONING STRUCTURE:**
# 1. Opening Statement (1-2 lines): Why this task ranks at this position
# 2. Quantitative Analysis (2-3 bullet points): Metrics, scores, comparative data
# 3. Strategic Importance (2-3 lines): How it fits the overall study plan
# 4. Learning Dependencies (1-2 lines): Prerequisites and what it enables
# 5. Execution Recommendations (2-3 bullet points): Specific study approaches
# 6. Expected Outcomes (1 line): What mastery looks like

# Be specific, data-driven, and actionable. No generic statements."""
        
#         top_3_tasks = tasks[:3]
        
#         # Format task details for prompt
#         task_comparison = "\n".join([
#             f"Task {i+1}: {td['name']} | Pages: {td['pages']} | Complexity: {td['complexity']}/10 | "
#             f"Urgency: {td['urgency']}/10 | Category: {td['category']} | Est. Time: {td['estimated_hours']}h"
#             for i, td in enumerate(task_details[:3])
#         ])
        
#         user_prompt = f"""**STUDENT'S GOAL:** {user_goal}
# **PRIORITIZATION METHOD:** {sort_method}
# **AVAILABLE DATA:**

# {task_comparison}

# ---

# **GENERATE COMPREHENSIVE REASONING FOR EACH TASK:**

# For each of the 3 tasks above, provide EXACTLY 10-12 lines of detailed reasoning following this structure:

# ### Task 1: {task_details[0]['name']}

# [Opening: 1-2 lines explaining why this is ranked #1, referencing the {sort_method} method and specific metrics]

# **Quantitative Justification:**
# • Urgency Score: {task_details[0]['urgency']}/10 ({"%.0f" % (task_details[0]['urgency_ratio']*100)}% of maximum) - [explain urgency factors]
# • Complexity Level: {task_details[0]['complexity']}/10 - [explain difficulty and prerequisites]
# • Time Investment: {task_details[0]['estimated_hours']} hours required for comprehensive mastery

# **Strategic Positioning:**
# [2-3 lines explaining how this task serves as a cornerstone for achieving "{user_goal}". Include specific connections to other materials and learning objectives. Reference the {task_details[0]['pages']} pages of content and how they build critical knowledge.]

# **Learning Architecture:**
# {"This material serves as a foundation for subsequent topics." if task_details[0]['is_foundational'] else "This builds upon previously established concepts."} [1-2 lines on dependencies]

# **Implementation Strategy:**
# • Begin with a 30-minute overview scan to map key concepts across all {task_details[0]['pages']} pages
# • Allocate {"intensive 2-hour blocks" if task_details[0]['complexity'] >= 7 else "focused 90-minute sessions"} for deep engagement
# • Create comprehensive notes and practice problems for retention

# **Success Metrics:** Complete understanding demonstrated through ability to solve complex problems and explain concepts to others.

# ---

# ### Task 2: {task_details[1]['name']}

# [Similar structure with 10-12 lines, explaining why it's #2, comparing with Task 1]

# ---

# ### Task 3: {task_details[2]['name']}

# [Similar structure with 10-12 lines, explaining why it's #3, comparing with Tasks 1 & 2]"""
        
#         messages = [
#             SystemMessage(content=system_prompt),
#             HumanMessage(content=user_prompt)
#         ]
        
#         response = llm.invoke(messages)
#         full_explanation = response.content.strip()
        
#         # Parse and structure the reasoning
#         task_reasoning = {}
        
#         for task_idx, task in enumerate(top_3_tasks, 1):
#             task_name = task["task"]
#             clean_name = _clean_task_name(task_name)
            
#             # Extract the detailed reasoning for this task
#             explanation = _extract_enhanced_reasoning(full_explanation, clean_name, task_idx)
            
#             if not explanation:
#                 # Generate detailed fallback reasoning
#                 explanation = _generate_enhanced_fallback(
#                     clean_name, 
#                     task_details[task_idx - 1],
#                     sort_method,
#                     user_goal,
#                     task_idx,
#                     task_details
#                 )
            
#             task_reasoning[task_name] = explanation
        
#         # Enhanced reasoning for remaining tasks (4+)
#         for idx, task in enumerate(tasks[3:], 4):
#             task_name = task["task"]
#             clean_name = _clean_task_name(task_name)
#             analysis = task["analysis"]
            
#             task_reasoning[task_name] = f"""## {clean_name}

# **Priority Ranking:** Position #{idx} in study sequence
# **Document Metrics:** {analysis.get('pages', 0)} pages | {analysis.get('estimated_hours', 0)} hours estimated study time

# **Quantitative Assessment:**
# • Urgency Level: {analysis.get('urgency_score', 5)}/10 - {"High priority for upcoming assessments" if analysis.get('urgency_score', 5) >= 7 else "Moderate timeline flexibility"}
# • Complexity Rating: {analysis.get('complexity', 5)}/10 - {"Advanced material requiring prerequisite knowledge" if analysis.get('complexity', 5) >= 7 else "Accessible with current knowledge base"}
# • Category: {analysis.get('category', 'General')} - Core component of curriculum

# **Strategic Relevance:**
# This material {"forms a foundational component" if analysis.get('is_foundational') else "builds upon established foundations"} supporting the goal: "{user_goal}".
# Recommended to study after completing higher-priority materials for optimal knowledge integration.

# **Study Approach:** Allocate dedicated focus blocks with regular review sessions for retention."""
        
#         # Enhanced schedule reasoning
#         schedule_reasoning = f"""**Optimized Weekly Schedule Analysis:**

# Total study time allocated: {schedule_data.get('total_allocated_hours', 0)} hours
# Utilization efficiency: {schedule_data.get('utilization_percent', 0)}% of available time
# Schedule optimization method: {sort_method} prioritization

# The schedule has been algorithmically optimized to:
# • Balance high-complexity materials with lighter review sessions
# • Align with your stated preferences and constraints
# • Maximize retention through spaced repetition principles
# • Ensure adequate preparation time for assessments"""
        
#         return {
#             "full_explanation": full_explanation,
#             "schedule": schedule_reasoning,
#             "tasks": task_reasoning,
#             "method": sort_method
#         }
    
#     except Exception as e:
#         print(f"⚠️ Reasoning generation failed: {e}")
#         # Generate fallback reasoning for all tasks
#         task_reasoning = {}
#         for idx, task in enumerate(tasks, 1):
#             task_name = task["task"]
#             clean_name = _clean_task_name(task_name)
#             analysis = task["analysis"]
            
#             task_reasoning[task_name] = _generate_enhanced_fallback(
#                 clean_name,
#                 {
#                     'name': clean_name,
#                     'category': analysis.get('category', 'General'),
#                     'pages': analysis.get('pages', 0),
#                     'complexity': analysis.get('complexity', 5),
#                     'urgency': analysis.get('urgency_score', 5),
#                     'is_foundational': analysis.get('is_foundational', False),
#                     'estimated_hours': analysis.get('estimated_hours', 0),
#                     'priority': idx
#                 },
#                 sort_method,
#                 user_goal,
#                 min(idx, 3),
#                 []
#             )
        
#         return {
#             "full_explanation": f"Study plan prioritized using {sort_method} methodology to achieve: {user_goal}",
#             "schedule": schedule_data.get("reasoning", "Optimized weekly schedule based on available time"),
#             "tasks": task_reasoning,
#             "method": sort_method
#         }



def _extract_enhanced_reasoning(full_text: str, task_name: str, task_idx: int) -> str:
    """
    Extract enhanced structured reasoning from LLM response.
    """
    # Try multiple patterns to find the task section
    patterns = [
        f"### Task {task_idx}: {re.escape(task_name)}(.*?)(?=###|$)",
        f"### {re.escape(task_name)}(.*?)(?=###|$)",
        f"Task {task_idx}.*?{re.escape(task_name)}(.*?)(?=------|###|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
        if match:
            section = match.group(0) if "###" in pattern else match.group(1)
            # Clean up the section
            section = section.strip()
            # Ensure it's substantial (at least 500 characters for detailed reasoning)
            if len(section) > 500:
                return section
    
    return None


def _generate_enhanced_fallback(clean_name: str, task_details: Dict, sort_method: str, 
                                user_goal: str, task_idx: int, all_tasks: List) -> str:
    """
    Generate comprehensive fallback reasoning with 10-12 lines of professional content.
    """
    urgency = task_details.get('urgency', 5)
    complexity = task_details.get('complexity', 5)
    is_foundational = task_details.get('is_foundational', False)
    est_hours = task_details.get('estimated_hours', 0)
    category = task_details.get('category', 'General')
    pages = task_details.get('pages', 0)
    
    # Position-specific messaging
    position_rationale = {
        1: f"This document achieves the highest priority ranking through the {sort_method} methodology, "
           f"demonstrating critical importance for your academic success.",
        2: f"Ranked second in the optimized sequence, this material provides essential complementary knowledge "
           f"that bridges foundational concepts with advanced applications.",
        3: f"Positioned third in the priority hierarchy, this content consolidates and extends the concepts "
           f"introduced in higher-priority materials."
    }
    
    # Complexity-based recommendations
    if complexity >= 8:
        study_approach = "Requires intensive study with multiple review cycles and practice problem sets"
    elif complexity >= 5:
        study_approach = "Moderate difficulty requiring focused study sessions with regular comprehension checks"
    else:
        study_approach = "Accessible material suitable for self-paced learning with standard review intervals"
    
    # Build the comprehensive reasoning
    reasoning = f"""## {clean_name}

**Priority Position #{task_details['priority']}** - {position_rationale.get(task_idx, f"Strategically positioned at rank {task_details['priority']} for optimal learning progression.")}

**Quantitative Analysis:**
• Urgency Score: {urgency}/10 - {"Immediate attention required" if urgency >= 8 else "High priority" if urgency >= 6 else "Standard timeline"} based on curriculum deadlines and dependencies
• Complexity Level: {complexity}/10 - {study_approach}
• Document Scope: {pages} pages requiring approximately {est_hours} hours for comprehensive mastery
• Content Category: {category} - {"Foundational prerequisite material" if is_foundational else "Advanced topic building on prerequisites"}

**Strategic Importance:**
This material directly supports your objective to "{user_goal}" by {"establishing critical foundational knowledge" if is_foundational else "advancing specialized understanding"}.
The {pages}-page document contains {"essential theoretical frameworks" if category in ['Theory', 'Concepts'] else "practical applications and case studies"} that form 
{"the backbone of subsequent learning" if task_idx == 1 else "important connections to previously studied concepts"}.

**Learning Dependencies:**
{"⚡ No prerequisites - start immediately" if is_foundational and task_idx == 1 else f"📚 Builds upon concepts from higher-priority materials"}.
{"This knowledge enables understanding of ALL subsequent topics in the curriculum." if task_idx == 1 and is_foundational else "Successful completion unlocks advanced topics in the learning pathway."}

**Recommended Study Strategy:**
• Initial Assessment: Conduct a 20-minute preview to identify key concepts and learning objectives
• Deep Engagement: Allocate {f"{est_hours} hours across multiple sessions" if est_hours > 0 else "appropriate study blocks"} for thorough understanding
• Active Learning: Create concept maps, solve practice problems, and generate summary notes
• Retention Protocol: Implement spaced repetition with reviews at 24 hours, 3 days, and 1 week

**Expected Outcomes:** 
Mastery demonstrated through {"ability to solve complex problems independently" if complexity >= 7 else "confident application of concepts"} and 
capacity to explain key principles to others. Achievement directly contributes to "{user_goal}"."""
    
    return reasoning


# Also update the FlexibleSchedulingTool to add task types
# Add this helper function to determine task type
def _determine_task_type(task_name: str, task_analysis: Dict) -> str:
    """
    Determine the type of study activity based on task characteristics.
    Returns: Theory, Practical, Exam, Assignment, Review, or Workshop
    """
    name_lower = task_name.lower()
    category = task_analysis.get('category', '').lower()
    complexity = task_analysis.get('complexity', 5)
    
    # Check for specific keywords
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
        return "Theory"  # Default to Theory for standard materials


# Update the schedule generation in FlexibleSchedulingTool (in agent_tools_advanced.py)
# When creating schedule items, add the type field:
# Example modification for the schedule creation:
"""
schedule_item = {
    "day": day_name,
    "time": time_slot,
    "task": task_name,
    "hours": duration,
    "type": _determine_task_type(task_name, task_analysis)  # Add this line
}
"""

# ==================== OPTIMIZED PLANNING TASK ====================

@shared_task(bind=True, max_retries=2, time_limit=600)
def generate_optimized_plan_async(self, user_id, upload_ids, user_goal, 
                                  time_input, constraints, sort_method):
    """
    Ultra-optimized agentic planning with knowledge-grounded reasoning.
    """
    try:
        print(f"🚀 [Task {self.request.id}] Starting knowledge-grounded planning")
        start_time = time.time()
        ToolLogger.clear_logs()

        from django.contrib.auth import get_user_model
        User = get_user_model()
        
        # Get uploads
        uploads = Upload.objects.filter(
            id__in=upload_ids,
            user_id=user_id,
            status='processed'
        ).select_related('user')
        
        if not uploads.exists():
            return {"success": False, "error": "No processed uploads found"}

        print(f"   Found {uploads.count()} processed uploads")

        # Phase 1: Fast Task Analysis with KB Grounding
        self.update_state(
            state='PROCESSING',
            meta={'current': 1, 'total': 4, 'status': 'Analyzing tasks with knowledge base...'}
        )
        
        task_tool = EnhancedTaskAnalysisTool()
        analyzed_tasks = []
        
        for upload in uploads:
            # Get content for KB comparison
            chunks = list(Chunk.objects.filter(upload=upload).order_by('start_page')[:3])
            content = " ".join([chunk.text[:500] for chunk in chunks])  # More content for KB
            
            metadata = {
                "pages": upload.pages or 0, 
                "chunk_count": Chunk.objects.filter(upload=upload).count(), 
                "deadline": None,
                "source_type": "textbook"  # Can be enhanced with actual detection
            }
            
            # Analysis with KB grounding enabled
            analysis_result = task_tool._run(
                task_name=upload.filename, 
                content_summary=content, 
                metadata=metadata, 
                sort_preference=sort_method,
                use_knowledge_grounding=True  # Enable KB comparison
            )
            
            analyzed_tasks.append(json.loads(analysis_result))

        print(f"   Analyzed {len(analyzed_tasks)} tasks with KB grounding")

        # Phase 2: Sort and assign priority
        self.update_state(
            state='PROCESSING',
            meta={'current': 2, 'total': 4, 'status': 'Prioritizing with KB context...'}
        )
        
        # Sort by knowledge-adjusted score if available
        analyzed_tasks.sort(
            key=lambda x: x["analysis"].get("knowledge_adjusted_score", 
                                           x["analysis"].get("preferred_score", 0)), 
            reverse=True
        )
        
        for idx, task in enumerate(analyzed_tasks, 1):
            task["priority"] = idx

        print(f"   Prioritized tasks using '{sort_method}' method with KB weighting")

        # Phase 3: Generate schedule
        self.update_state(
            state='PROCESSING',
            meta={'current': 3, 'total': 4, 'status': 'Creating optimized schedule...'}
        )
        
        total_hours_available = _convert_time_to_hours(time_input)
        
        if total_hours_available > 0:
            scheduling_tool = FlexibleSchedulingTool()
            schedule_result = scheduling_tool._run(
                prioritized_tasks=json.dumps(analyzed_tasks),
                available_time=time_input,
                constraints=constraints,
                sort_method=sort_method
            )
            schedule_data = json.loads(schedule_result)
            schedule_data['available_hours'] = total_hours_available
            schedule_data['constraints'] = constraints
            schedule = schedule_data.get("schedule", [])
            print(f"   Created schedule with {len(schedule)} items")
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
            print(f"   No time provided - skipping schedule")

        # Phase 4: Generate Knowledge-Grounded Reasoning
        self.update_state(
            state='PROCESSING',
            meta={'current': 4, 'total': 4, 'status': 'Generating KB-grounded reasoning...'}
        )
        
        # USE NEW REASONING ENGINE
        reasoning = generate_knowledge_grounded_reasoning(
            tasks=analyzed_tasks,
            schedule_data=schedule_data,
            user_goal=user_goal,
            sort_method=sort_method
        )
        
        total_time = time.time() - start_time

        # Build plan_json - SCHEDULE FIRST, THEN TASKS
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
                    "kb_grounded": reasoning.get("kb_grounded", False)
                }
            })
        
        # Add prioritized tasks with KB-grounded reasoning
        for task in analyzed_tasks:
            analysis = task["analysis"]
            kb_grounding = analysis.get('knowledge_grounding', {})
            
            plan_json.append({
                "file": task["task"],
                "priority": task["priority"],
                "reason": reasoning.get("tasks", {}).get(task["task"], "AI analysis"),
                "chunk_count": analysis.get("chunks", 0),
                "pages": analysis.get("pages", 0),
                "estimated_hours": analysis.get("estimated_hours", 0),
                "category": analysis.get("category", "general"),
                "complexity": analysis.get("complexity", 5),
                "urgency": analysis.get("urgency_score", 5),
                # NEW: Add KB metrics for transparency
                "kb_relevance": kb_grounding.get("knowledge_relevance_score", 0.5),
                "kb_confidence": kb_grounding.get("confidence", 0),
                "kb_depth": kb_grounding.get("knowledge_depth", "unknown"),
                "knowledge_adjusted": analysis.get("knowledge_adjusted_score") is not None
            })

        # Save Plan
        plan = Plan.objects.create(
            user_id=user_id,
            upload=uploads.first(),  
            version=Plan.objects.filter(user_id=user_id).count() + 1,
            plan_json=plan_json
        )
        
        # Save StudyPlanHistory
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
        
        # Calculate KB statistics
        kb_stats = {
            "avg_confidence": sum(
                t.get('analysis', {}).get('knowledge_grounding', {}).get('confidence', 0)
                for t in analyzed_tasks
            ) / len(analyzed_tasks) if analyzed_tasks else 0,
            "tasks_with_kb": sum(
                1 for t in analyzed_tasks 
                if t.get('analysis', {}).get('knowledge_grounding', {}).get('confidence', 0) > 0.3
            ),
            "avg_kb_relevance": sum(
                t.get('analysis', {}).get('knowledge_grounding', {}).get('knowledge_relevance_score', 0)
                for t in analyzed_tasks
            ) / len(analyzed_tasks) if analyzed_tasks else 0,
            "kb_grounded_reasoning": reasoning.get("kb_grounded", False)
        }
        
        print(f"✅ [Task {self.request.id}] Completed in {total_time:.2f}s")
        print(f"   Schedule items: {len(schedule)}")
        print(f"   Total tasks: {len(analyzed_tasks)}")
        print(f"   Tool calls: {tool_summary.get('total_calls', 0)}")
        print(f"   KB-grounded tasks: {kb_stats['tasks_with_kb']}/{len(analyzed_tasks)}")
        print(f"   Avg KB confidence: {kb_stats['avg_confidence']:.3f}")
        
        return {
            "success": True, 
            "plan_id": plan.id, 
            "history_id": history.id, 
            "total_time": round(total_time, 2),
            "total_tasks": len(analyzed_tasks),
            "schedule_items": len(schedule),
            "execution_time": round(total_time, 2),
            "tool_logs": tool_logs[-10:],
            "tool_summary": tool_summary,
            "kb_statistics": kb_stats,
            "reasoning": reasoning.get("full_explanation", "")
        }

    except Exception as exc:
        import traceback
        error_trace = traceback.format_exc()
        
        print(f"❌ [Task {self.request.id}] Failed: {str(exc)}")
        print(error_trace)
        
        # Retry with exponential backoff
        if self.request.retries < self.max_retries:
            print(f"   Retrying ({self.request.retries + 1}/{self.max_retries})...")
            raise self.retry(exc=exc, countdown=5 * (2 ** self.request.retries))
        
        return {
            "success": False, 
            "error": str(exc),
            "error_trace": error_trace[:500],
            "tool_logs": ToolLogger.get_logs()[-5:]
        }

# ==================== OPTIONAL BATCH PDF CHUNK INSERT ====================


# ==================== HELPER METHODS ====================

def _convert_time_to_hours(time_dict: Dict) -> float:
    """Convert flexible time to total hours"""
    total = 0.0
    total += time_dict.get("years", 0) * 365 * 24
    total += time_dict.get("months", 0) * 30 * 24
    total += time_dict.get("weeks", 0) * 7 * 24
    total += time_dict.get("days", 0) * 24
    total += time_dict.get("hours", 0)
    return total

# ==================== BATCH PROCESSING FOR LARGE UPLOADS ====================

@shared_task(bind=True)
def process_large_upload_batch(self, upload_id, batch_start, batch_end, batch_size=50):
    """
    Optimized batch processing for very large PDFs.
    - Extracts text in chunks
    - Sanitizes
    - Inserts into DB via bulk_create
    - Supports flexible batch sizes
    """
    try:
        from .pdf_utils import extract_text_from_pdf, chunk_text, sanitize_text
        from .llm_config import embeddings, INDEX_NAME
        from langchain_pinecone import PineconeVectorStore
        import os

        upload = Upload.objects.get(id=upload_id)
        print(f"📄 Processing batch {batch_start}-{batch_end} of {upload.filename}")

        # Extract text for the batch
        text_data = extract_text_from_pdf(upload.file.path)
        pages_batch = text_data["pages"][batch_start:batch_end]

        # Chunk the batch
        chunks = chunk_text({"pages": pages_batch, "total_pages": len(pages_batch)})

        all_chunks_data = []
        vector_texts = []
        vector_ids = []
        vector_metadatas = []

        # Prepare data for DB bulk insert + vector store
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

            # Bulk insert every `batch_size` chunks to reduce DB load
            if len(all_chunks_data) >= batch_size:
                Chunk.objects.bulk_create(all_chunks_data, batch_size=batch_size)
                all_chunks_data = []

        # Insert remaining chunks
        if all_chunks_data:
            Chunk.objects.bulk_create(all_chunks_data, batch_size=batch_size)

        # Add chunks to vector store
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

        print(f"✅ Batch complete: {len(vector_texts)} chunks added to DB & vector store")

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