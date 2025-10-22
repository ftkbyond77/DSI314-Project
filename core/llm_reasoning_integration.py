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
    
    REASONING_SYSTEM_PROMPT = """You are an expert academic strategist providing comprehensive, knowledge-grounded study plan reasoning.

**YOUR ROLE:**
You analyze study materials using both intrinsic properties (complexity, urgency, pages) AND knowledge base context (how similar content is already covered) to provide logical, defensible prioritization with clear reasoning.

**REASONING FRAMEWORK - Use "What, Why, How, Priority" Structure:**

1. **WHAT** - Material Description
   - Content summary and scope
   - Category, complexity, and characteristics
   - Source type (textbook, research paper, assignment, etc.)

2. **WHY** - Strategic Importance
   - Knowledge base context: Is this well-covered, partially covered, or a knowledge gap?
   - How does it relate to the student's goal?
   - Dependencies: What does it enable? What does it require?
   - Urgency factors (deadlines, prerequisites for other work)

3. **HOW** - Knowledge Base Evidence
   - KB coverage depth (extensive/substantial/moderate/limited/minimal)
   - Confidence in KB assessment (high/medium/low)
   - Similar materials available in KB
   - Knowledge gaps this material fills (or doesn't fill)
   - Domain-specific context (STEM vs Business vs etc.)

4. **PRIORITY** - Justified Ranking
   - Synthesize intrinsic factors + KB context
   - Explain why this ranks #X relative to others
   - Compare explicitly to other materials when relevant
   - Address trade-offs (e.g., "Although KB coverage is high, urgency demands priority")

**CRITICAL REQUIREMENTS:**
- Be specific and data-driven: cite actual metrics (KB relevance scores, complexity ratings, etc.)
- Make explicit comparisons between tasks to justify relative ranking
- Acknowledge uncertainty when KB confidence is low
- Explain how KB context influenced the decision
- Use professional, academic language
- Provide 8-12 lines of reasoning per high-priority task
- Be consistent: similar materials should get similar reasoning

**KNOWLEDGE BASE INTERPRETATION:**
- Extensive KB coverage → Lower priority unless urgency is high (materials already available)
- Minimal KB coverage → Higher priority (fills knowledge gap, foundational opportunity)
- High KB relevance + High confidence → Strong signal for standard priority
- Low confidence → Rely more on intrinsic factors (complexity, urgency)

Your reasoning should be logical, defensible, and actionable."""

    @classmethod
    def generate_prioritization_reasoning(
        cls,
        tasks: List[Dict],
        user_goal: str,
        sort_method: str,
        max_tasks_detailed: int = 5
    ) -> Dict:
        """
        Generate comprehensive reasoning for task prioritization using KB context.
        
        Args:
            tasks: List of analyzed tasks with KB grounding
            user_goal: Student's learning objective
            sort_method: Prioritization method used
            max_tasks_detailed: Number of tasks to provide detailed reasoning for
        
        Returns:
            Dict with structured reasoning for each task
        """
        
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
        
        # Build prompt - FIXED f-string escaping
        user_prompt = f"""**STUDENT'S GOAL:** {user_goal}
    **PRIORITIZATION METHOD:** {sort_method}

    **TASKS TO ANALYZE (in priority order):**

    {chr(10).join(task_summaries)}

    ---

    **GENERATE KNOWLEDGE-GROUNDED REASONING:**

    For each of the {max_tasks_detailed} tasks above, provide comprehensive reasoning following the "What, Why, How, Priority" framework. Each task should receive 8-12 lines of detailed analysis that:

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
            
            # Generate summary reasoning for remaining tasks
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
    
    # Generate task prioritization reasoning
    task_reasoning_result = KnowledgeGroundedReasoningEngine.generate_prioritization_reasoning(
        tasks=tasks,
        user_goal=user_goal,
        sort_method=sort_method,
        max_tasks_detailed=5
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
            "detailed_reasoning_count": min(5, len(tasks)),
            "kb_confidence_avg": sum(
                t.get('analysis', {}).get('knowledge_grounding', {}).get('confidence', 0)
                for t in tasks
            ) / len(tasks) if tasks else 0
        }
    }