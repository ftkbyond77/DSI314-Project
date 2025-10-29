# core/llm_reasoning_integration.py - LLM Integration for Knowledge-Grounded Reasoning

from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from .llm_config import llm
import json
import re

class KnowledgeGroundedReasoningEngine:
    """
    Integrates knowledge base context into LLM reasoning for explainable prioritization.
    
    Uses industry-standard prompting techniques:
    - Chain-of-Thought (CoT) reasoning
    - Few-shot examples
    - Structured output format
    - Contextual grounding
    """
    
    REASONING_SYSTEM_PROMPT = """
    ROLE: You are an expert academic strategist and reasoning engine.

MISSION: Your purpose is to analyze a list of study materials and prioritize them for a student. You must provide comprehensive, knowledge-grounded reasoning for your prioritization. Your analysis MUST integrate two distinct data sources:

1. Intrinsic Properties: The material's own characteristics (e.g., complexity, page count, urgency/deadline, source type).
2. Knowledge Base (KB) Context: How this material relates to the student's existing knowledge (e.g., relevance scores, coverage depth, identified gaps).

Core Prioritization Logic (Internal Monologue)
Before generating any output, you must apply the following logical hierarchy. This is your internal reasoning process.

Rule 1: Urgency is Paramount.
IF a material has an imminent deadline (e.g., "Due Today," "Due Tomorrow") or is a hard prerequisite for another urgent task, it MUST be the highest priority (e.g., Priority 1 or 2).
Rationale: Deadlines are non-negotiable and override all other factors, including KB coverage.

Rule 2: Knowledge Gaps are Foundational.
IF a material fills a critical, documented "Knowledge Gap" (i.e., KB coverage is "Minimal" or "None") AND it is foundational (i.e., enables future learning), it MUST receive high priority, just below any urgent deadlines.
Rationale: Building a foundation is more important than reviewing familiar topics.

Rule 3: Redundancy is Inefficient.
IF a material has "Extensive" or "Substantial" KB coverage (i.e., it is redundant), it MUST be deprioritized.
Rationale: The student's time is better spent on new information.
Exception: This rule is overridden by Rule 1 (Urgency). An assignment on a familiar topic is still a high priority if it's due soon.

Rule 4: Complexity is a Modifier, Not a Driver.
IF a material is "High Complexity," its priority is modified by other factors:
- High Complexity + Low KB Coverage → Increase Priority (This is a difficult, new topic).
- High Complexity + High KB Coverage → Decrease Priority (The student should review existing, simpler KB materials first).

Rule 5: Handle Uncertainty.
IF KB confidence is "Low," you MUST state this and explicitly note that your prioritization relies more heavily on Intrinsic Properties (Urgency, Type, Complexity) as the KB data is unreliable.

Required Output Structure (User-Facing Response)
You will generate a response by first stating the full priority list, followed by a detailed reasoning block for each material, presented in priority order.

You MUST use the following four-part Markdown structure for every item:

[Priority #]: [Material Title]

1. Material Analysis (WHAT):
Summary: [Concisely describe the material's content and scope.]
Source: [Identify source type, e.g., Textbook Chapter, Research Paper, Assignment, Lecture.]
Properties: [List key intrinsic data, e.g., Complexity: High, Pages: 45, Urgency: Low.]

2. Strategic Importance (WHY):
Rationale: [Explain why this material matters for the student's goals. State its dependencies (e.g., "This is foundational for Task B") and its urgency (e.g., "Imminent deadline").]

3. Knowledge Base Context (HOW):
Coverage: [State the KB coverage level (e.g., Extensive, Moderate, Minimal, Gap) and the assessment Confidence (High, Medium, Low).]
Gap Analysis: [Explicitly state if this material fills a known knowledge gap or if it is redundant (e.g., "This material is redundant; similar content is already covered by [X, Y].") Cite KB relevance scores if provided.]

4. Prioritization & Rationale (PRIORITY):
Justification: [Synthesize all points above into a final verdict. This section MUST include a comparative statement explaining why this item is ranked where it is relative to others (e.g., "This is Priority 1 due to its imminent deadline, which takes precedence over the high KB coverage of Task 2.")]
Trade-off (if any): [Acknowledge any conflicting factors, e.g., "Although this topic is a knowledge gap, its 'Low' urgency places it behind the time-sensitive assignment."]

Critical Constraints
- Data-Driven: All reasoning must be explicitly grounded in the provided intrinsic data and KB context. Cite the data.
- Comparative Analysis: The "Prioritization & Rationale" section is the most critical. Always justify a rank by comparing it to the items ranked above or below it.
- Verbosity: High-priority items (e.g., Priority 1-3) require 8-12 lines of detailed, multi-faceted reasoning. Lower-priority items can be more concise (4-6 lines).
- Tone: Maintain a professional, academic, and authoritative tone. You are the expert strategist.

    
    """

    @classmethod
    def generate_prioritization_reasoning(
        cls,
        tasks: List[Dict],
        user_goal: str,
        sort_method: str,
        max_tasks_detailed: int = None  # FIXED: Changed from 5 to None to process all tasks
    ) -> Dict:
        """
        Generate comprehensive reasoning for task prioritization using KB context.
        
        Args:
            tasks: List of analyzed tasks with KB grounding
            user_goal: Student's learning objective
            sort_method: Prioritization method used
            max_tasks_detailed: Number of tasks to provide detailed reasoning for (None = all tasks)
        
        Returns:
            Dict with structured reasoning for each task
        """
        
        # FIXED: If max_tasks_detailed is None, process all tasks
        if max_tasks_detailed is None:
            max_tasks_detailed = len(tasks)
        
        # Prepare task summaries with KB context
        task_summaries = []
        for idx, task in enumerate(tasks[:max_tasks_detailed], 1):
            analysis = task.get('analysis', {})
            kb_grounding = analysis.get('knowledge_grounding', {})
            kb_reasoning = analysis.get('kb_reasoning_context', {})
            
            summary = f"""
    **Task {idx}: {task['task']}**

    Intrinsic Properties:
    - Category: {analysis.get('category', 'unknown')}
    - Complexity: {analysis.get('complexity', 5)}/10
    - Urgency: {analysis.get('urgency_score', 5)}/10
    - Pages: {analysis.get('pages', 0)}
    - Estimated Hours: {analysis.get('estimated_hours', 0)}
    - Foundational: {'Yes' if analysis.get('is_foundational') else 'No'}
    - Source Type: {analysis.get('source_type', 'unknown')}

    Knowledge Base Context:
    - KB Relevance Score: {kb_grounding.get('knowledge_relevance_score', 0.5):.3f}
    - Confidence: {kb_grounding.get('confidence', 0):.3f}
    - KB Depth: {kb_grounding.get('knowledge_depth', 'unknown')}
    - Documents Found: {kb_grounding.get('documents_found', 0)}
    - Dominant KB Category: {kb_grounding.get('dominant_kb_category', 'unknown')}
    - KB Interpretation: {analysis.get('kb_interpretation', 'N/A')}

    KB Reasoning Context:
    {json.dumps(kb_reasoning.get('knowledge_coverage', {}), indent=2) if kb_reasoning.get('knowledge_coverage') else 'No KB reasoning context available'}

    Priority Rationale: {kb_reasoning.get('priority_rationale', 'Standard prioritization')}
    """
            task_summaries.append(summary)
        
        # Build prompt - FIXED: Use len(task_summaries) instead of hardcoded value
        user_prompt = f"""**STUDENT'S GOAL:** {user_goal}
    **PRIORITIZATION METHOD:** {sort_method}

    **TASKS TO ANALYZE (in priority order):**

    {chr(10).join(task_summaries)}

    ---

    **GENERATE KNOWLEDGE-GROUNDED REASONING:**

    For each of the {len(task_summaries)} tasks above, provide comprehensive reasoning following the "What, Why, How, Priority" framework. Each task should receive 8-12 lines of detailed analysis that:

    1. Describes WHAT the material covers and its characteristics
    2. Explains WHY it matters for the student's goal: "{user_goal}"
    3. Discusses HOW the knowledge base context influences the decision (KB coverage, gaps, domain context)
    4. Justifies the PRIORITY ranking with explicit comparisons to other tasks

    **Critical**: Use the KB metrics (relevance scores, confidence, depth) to support your reasoning. Explain how KB context either reinforces or modifies the priority ranking.

    Format each task as:

    ### Task [number]: [Task Name]

    **Material Overview (WHAT):**
    [2-3 lines describing content, scope, and characteristics]

    **Strategic Importance (WHY):**
    [3-4 lines explaining relevance to goal, dependencies, and urgency factors]

    **Knowledge Base Analysis (HOW):**
    [3-4 lines discussing KB coverage, confidence, gaps, and domain context]
    - KB Coverage: [depth] with [confidence level] confidence
    - [Specific KB insights from the data above]

    **Priority Justification (PRIORITY):**
    [2-3 lines synthesizing all factors and explicitly comparing to other tasks]
    - Ranked at position [number] because [specific reasons incorporating both intrinsic and KB factors]

    ---

    Begin your analysis:"""

        try:
            messages = [
                SystemMessage(content=cls.REASONING_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            full_reasoning = response.content.strip()
            
            # Parse reasoning per task
            task_reasoning = cls._parse_task_reasoning(full_reasoning, tasks[:max_tasks_detailed])
            
            # Generate summary reasoning for remaining tasks (if any)
            for idx, task in enumerate(tasks[max_tasks_detailed:], max_tasks_detailed + 1):
                task_reasoning[task['task']] = cls._generate_summary_reasoning(
                    task=task,
                    rank=idx,
                    user_goal=user_goal
                )
            
            return {
                "full_explanation": full_reasoning,
                "tasks": task_reasoning,
                "method": sort_method,
                "kb_grounded": True
            }
            
        except Exception as e:
            print(f"⚠️ LLM reasoning failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to template-based reasoning
            return cls._generate_fallback_reasoning(tasks, user_goal, sort_method)
    
    @classmethod
    def _parse_task_reasoning(cls, full_text: str, tasks: List[Dict]) -> Dict[str, str]:
        """
        Parse LLM output to extract reasoning for each task.
        """
        task_reasoning = {}
        
        for idx, task in enumerate(tasks, 1):
            task_name = task['task']
            
            # Try to extract section for this task
            patterns = [
                f"### Task {idx}:.*?(?=###|$)",
                f"Task {idx}:.*?(?=Task {idx+1}:|$)",
                f"#{'{1,3}'} {re.escape(task_name)}.*?(?=###|$)"
            ]
            
            extracted = None
            for pattern in patterns:
                match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted = match.group(0).strip()
                    if len(extracted) > 200:  # Substantial content
                        break
            
            if extracted:
                task_reasoning[task_name] = extracted
            else:
                # Use fallback
                task_reasoning[task_name] = cls._generate_summary_reasoning(
                    task=task,
                    rank=idx,
                    user_goal="achieve learning objectives"
                )
        
        return task_reasoning
    
    @classmethod
    def _generate_summary_reasoning(cls, task: Dict, rank: int, user_goal: str) -> str:
        """
        Generate concise reasoning for lower-priority tasks.
        """
        analysis = task.get('analysis', {})
        kb_grounding = analysis.get('knowledge_grounding', {})
        
        task_name = task['task']
        category = analysis.get('category', 'general')
        complexity = analysis.get('complexity', 5)
        urgency = analysis.get('urgency_score', 5)
        kb_score = kb_grounding.get('knowledge_relevance_score', 0.5)
        kb_depth = kb_grounding.get('knowledge_depth', 'unknown')
        confidence = kb_grounding.get('confidence', 0)
        
        return f"""## {task_name}

**Priority Position #{rank}** - {category.replace('_', ' ').title()} material

**Material Profile:**
This {analysis.get('source_type', 'document')} covers {category} topics with complexity {complexity}/10 and urgency {urgency}/10.

**Knowledge Base Context:**
KB analysis shows {kb_depth} coverage (relevance: {kb_score:.2f}, confidence: {confidence:.2f}). {cls._kb_interpretation_snippet(kb_depth, kb_score, confidence)}

**Priority Rationale:**
Ranked #{rank} based on {analysis.get('sort_method', 'hybrid')} prioritization. {cls._priority_snippet(rank, urgency, kb_depth)} Supports goal: "{user_goal}".

**Recommended Approach:** {"Focus after higher-priority items" if rank > 3 else "Address promptly"} with {"intensive study blocks" if complexity >= 7 else "standard study sessions"}."""
    
    @classmethod
    def _kb_interpretation_snippet(cls, kb_depth: str, kb_score: float, confidence: float) -> str:
        """Generate interpretation snippet based on KB metrics."""
        if confidence < 0.3:
            return "Limited KB data available for assessment."
        
        if kb_depth.startswith('extensive'):
            return "Extensive KB coverage suggests abundant reference materials available."
        elif kb_depth.startswith('substantial'):
            return "Good KB coverage provides solid foundation resources."
        elif kb_depth in ['moderate', 'limited']:
            return "Moderate KB coverage - opportunity to expand knowledge base."
        else:
            return "Minimal KB coverage indicates knowledge gap worth addressing."
    
    @classmethod
    def _priority_snippet(cls, rank: int, urgency: int, kb_depth: str) -> str:
        """Generate priority explanation snippet."""
        if rank <= 3:
            if urgency >= 8:
                return "High urgency demands immediate attention regardless of KB coverage."
            elif kb_depth in ['minimal', 'none', 'limited']:
                return "Priority elevated due to knowledge gap identification."
            else:
                return "Strategic importance and timing justify top-tier priority."
        else:
            if kb_depth.startswith('extensive'):
                return "Lower priority due to abundant existing KB resources."
            else:
                return "Standard priority for systematic learning progression."
    
    @classmethod
    def _generate_fallback_reasoning(cls, tasks: List[Dict], user_goal: str, sort_method: str) -> Dict:
        """
        Fallback reasoning when LLM fails.
        """
        task_reasoning = {}
        
        for idx, task in enumerate(tasks, 1):
            task_reasoning[task['task']] = cls._generate_summary_reasoning(
                task=task,
                rank=idx,
                user_goal=user_goal
            )
        
        return {
            "full_explanation": f"Tasks prioritized using {sort_method} methodology to achieve: {user_goal}",
            "tasks": task_reasoning,
            "method": sort_method,
            "kb_grounded": False
        }


# ==================== COMPARISON REASONING ====================

class ComparisonReasoningEngine:
    """
    Generate reasoning for explicit task comparisons.
    Useful for explaining why Task A ranks above Task B.
    """
    
    @classmethod
    def explain_ranking_difference(
        cls,
        task_higher: Dict,
        task_lower: Dict,
        rank_higher: int,
        rank_lower: int
    ) -> str:
        """
        Explain why one task ranks above another.
        """
        
        analysis_h = task_higher.get('analysis', {})
        analysis_l = task_lower.get('analysis', {})
        
        kb_h = analysis_h.get('knowledge_grounding', {})
        kb_l = analysis_l.get('knowledge_grounding', {})
        
        # Collect factors
        factors = []
        
        # Urgency comparison
        urgency_diff = analysis_h.get('urgency_score', 5) - analysis_l.get('urgency_score', 5)
        if abs(urgency_diff) >= 2:
            if urgency_diff > 0:
                factors.append(f"**Urgency advantage** (+{urgency_diff} points)")
            else:
                factors.append(f"**Lower urgency** ({urgency_diff} points)")
        
        # Foundational comparison
        if analysis_h.get('is_foundational') and not analysis_l.get('is_foundational'):
            factors.append("**Foundational material** (prerequisite for other topics)")
        
        # KB comparison
        kb_score_diff = kb_h.get('knowledge_relevance_score', 0.5) - kb_l.get('knowledge_relevance_score', 0.5)
        kb_depth_h = kb_h.get('knowledge_depth', 'unknown')
        kb_depth_l = kb_l.get('knowledge_depth', 'unknown')
        
        if kb_depth_h in ['minimal', 'none'] and kb_depth_l.startswith('extensive'):
            factors.append(f"**Knowledge gap priority** (KB depth: {kb_depth_h} vs {kb_depth_l})")
        elif kb_depth_l in ['minimal', 'none'] and kb_depth_h.startswith('extensive'):
            factors.append(f"**Higher KB coverage** (may reduce relative priority)")
        
        # Complexity comparison
        complexity_diff = analysis_h.get('complexity', 5) - analysis_l.get('complexity', 5)
        if analysis_h.get('is_foundational') and complexity_diff < 0:
            factors.append("**Foundational + accessible** (ideal starting point)")
        
        # Build explanation
        if not factors:
            return f"Tasks #{rank_higher} and #{rank_lower} have similar priority factors. Ranking reflects marginal differences in {analysis_h.get('sort_method', 'hybrid')} scoring."
        
        explanation = f"**Why Task #{rank_higher} ranks above Task #{rank_lower}:**\n\n"
        explanation += "\n".join(f"• {factor}" for factor in factors)
        explanation += f"\n\nDespite these differences, both tasks are important for achieving your learning goals."
        
        return explanation


# ==================== SCHEDULE REASONING ====================

class ScheduleReasoningEngine:
    """
    Generate reasoning for schedule decisions.
    """
    
    @classmethod
    def explain_schedule(
        cls,
        schedule: List[Dict],
        total_hours: float,
        tasks: List[Dict],
        constraints: str
    ) -> str:
        """
        Generate comprehensive reasoning for schedule.
        """
        
        if not schedule:
            return "No schedule generated due to insufficient time availability."
        
        # Analyze schedule
        days_used = set(item['day'] for item in schedule)
        time_slots_used = set(item.get('time', 'unknown') for item in schedule)
        total_allocated = sum(item.get('hours', 0) for item in schedule)
        utilization = (total_allocated / total_hours * 100) if total_hours > 0 else 0
        
        # Task type distribution
        type_dist = {}
        for item in schedule:
            task_type = item.get('type', 'Study')
            type_dist[task_type] = type_dist.get(task_type, 0) + 1
        
        reasoning = f"""**Optimized Weekly Schedule Analysis:**

**Time Allocation:**
- Total Available: {total_hours:.1f} hours
- Total Allocated: {total_allocated:.1f} hours
- Utilization Rate: {utilization:.1f}%
- Days Utilized: {len(days_used)} days ({', '.join(sorted(days_used))})

**Activity Distribution:**
{chr(10).join(f'• {type_name}: {count} session(s)' for type_name, count in sorted(type_dist.items()))}

**Scheduling Strategy:**
The schedule has been optimized using evidence-based learning principles:

1. **Cognitive Load Management:** High-complexity materials (Theory, Exam Prep) scheduled during peak cognitive hours (mornings) when mental clarity is highest.

2. **Spaced Repetition:** Sessions distributed across multiple days to enhance retention through spaced learning intervals.

3. **Task Type Optimization:** 
   - Theory sessions: Concentrated in mornings for deep conceptual understanding
   - Practical work: Afternoons when hands-on engagement is optimal
   - Review sessions: Evenings for consolidation and reinforcement

4. **Priority Alignment:** High-priority tasks (ranks 1-3) receive earlier scheduling and longer session durations to ensure adequate preparation time.

"""
        
        # Add constraint acknowledgment
        if constraints:
            reasoning += f"""5. **Constraint Compliance:** Schedule respects specified preferences: "{constraints}"

"""
        
        # Knowledge base context
        has_kb_context = any(
            task.get('analysis', {}).get('knowledge_grounding', {}).get('confidence', 0) > 0.3
            for task in tasks
        )
        
        if has_kb_context:
            reasoning += """6. **Knowledge-Grounded Scheduling:** Tasks with limited KB coverage receive proportionally more study time to build foundational understanding, while extensively covered topics are allocated standard durations with emphasis on active practice.

"""
        
        reasoning += f"""**Expected Outcomes:**
Following this schedule systematically will result in comprehensive coverage of all materials while maintaining sustainable learning pace. The {utilization:.0f}% utilization rate allows for flexibility and prevents burnout."""
        
        return reasoning


# ==================== INTEGRATION FUNCTION ====================

def generate_knowledge_grounded_reasoning(
    tasks: List[Dict],
    schedule_data: Dict,
    user_goal: str,
    sort_method: str
) -> Dict:
    """
    Main integration function: Generate comprehensive knowledge-grounded reasoning.
    
    This replaces the existing _generate_reasoning_fast in tasks_agentic_optimized.py
    
    Args:
        tasks: Analyzed tasks with KB grounding
        schedule_data: Schedule information
        user_goal: Student's learning objective
        sort_method: Prioritization method
    
    Returns:
        Comprehensive reasoning dict for all tasks and schedule
    """
    
    # FIXED: Generate task prioritization reasoning for ALL tasks (removed max_tasks_detailed limit)
    task_reasoning_result = KnowledgeGroundedReasoningEngine.generate_prioritization_reasoning(
        tasks=tasks,
        user_goal=user_goal,
        sort_method=sort_method,
        max_tasks_detailed=None  # Changed from 5 to None to process all tasks
    )
    
    # Generate schedule reasoning
    schedule = schedule_data.get('schedule', [])
    total_hours = schedule_data.get('available_hours', 0)
    constraints = schedule_data.get('constraints', '')
    
    schedule_reasoning = ScheduleReasoningEngine.explain_schedule(
        schedule=schedule,
        total_hours=total_hours,
        tasks=tasks,
        constraints=constraints
    )
    
    # Generate comparison reasoning for top tasks
    comparisons = {}
    if len(tasks) >= 2:
        comparisons['1_vs_2'] = ComparisonReasoningEngine.explain_ranking_difference(
            task_higher=tasks[0],
            task_lower=tasks[1],
            rank_higher=1,
            rank_lower=2
        )
    
    if len(tasks) >= 3:
        comparisons['2_vs_3'] = ComparisonReasoningEngine.explain_ranking_difference(
            task_higher=tasks[1],
            task_lower=tasks[2],
            rank_higher=2,
            rank_lower=3
        )
    
    return {
        "full_explanation": task_reasoning_result.get('full_explanation', ''),
        "schedule": schedule_reasoning,
        "tasks": task_reasoning_result.get('tasks', {}),
        "comparisons": comparisons,
        "method": sort_method,
        "kb_grounded": task_reasoning_result.get('kb_grounded', False),
        "metadata": {
            "total_tasks": len(tasks),
            "detailed_reasoning_count": len(tasks),  # FIXED: Changed from min(5, len(tasks)) to len(tasks)
            "kb_confidence_avg": sum(
                t.get('analysis', {}).get('knowledge_grounding', {}).get('confidence', 0)
                for t in tasks
            ) / len(tasks) if tasks else 0
        }
    }