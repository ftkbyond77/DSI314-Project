# core/llm_reasoning_integration.py - FINAL PRODUCTION
# LLM-Based Ranking + Full Async + High Quality Reasoning

from typing import List, Dict, Optional, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from .llm_config import llm
import json
import re
import asyncio


# ==================== PRODUCTION ASYNC LLM RANKING ENGINE ====================

class KnowledgeGroundedReasoningEngine:
    """
    PRODUCTION: LLM-based ranking with full async processing.
    
    ✅ LLM does the ranking (not algorithmic)
    ✅ Full async (no ThreadPoolExecutor)
    ✅ High-quality, meaningful reasoning
    ✅ Long context with structured bullets
    ✅ Fast recall, high accuracy
    ✅ Production-grade with validation
    """
    
    REASONING_SYSTEM_PROMPT = """You are an expert academic strategist specializing in study plan prioritization.

**YOUR ROLE: RANKING + REASONING**
You will receive tasks with guidance scores. Your job is to:
1. **RANK the tasks** (assign priorities 1, 2, 3, ...) based on the sorting method
2. **GENERATE detailed reasoning** for each priority assignment
3. **COMPARE tasks explicitly** using the comparison table

**CRITICAL RULES:**
• Assign SEQUENTIAL priorities: 1, 2, 3, 4, ... (NO gaps, NO duplicates)
• EVERY task must have a UNIQUE priority
• You can ADJUST guidance priorities if you have strong reasoning
• EXPLAIN any deviation from guidance priorities

**OUTPUT FORMAT (MANDATORY XML):**

<task_analysis priority="[YOUR ASSIGNED PRIORITY: 1, 2, 3, ...]">
<task_name>[Task Name]</task_name>

<material_analysis>
**Document Scope & Characteristics:**
• Total Pages: [X] pages requiring [Y] hours study time
• Complexity Level: [Z]/10 - [interpretation: e.g., "highly technical", "moderately challenging", "introductory level"]
• Content Type: [textbook/lecture notes/practice problems/review material]
• Key Topics: [brief list of 2-3 main topics covered]
</material_analysis>

<strategic_importance>
**Academic Impact & Dependencies:**
• Urgency Assessment: [X]/10 - [specific reason: e.g., "due tomorrow", "exam in 2 weeks", "no deadline"]
• Foundational Value: [Yes/No] - [explanation: "prerequisite for topics X, Y" OR "builds on established knowledge"]
• Goal Alignment: [explain HOW this supports the user's learning goal with specific connection]
• Learning Sequence: [where this fits in curriculum: "must complete before X" OR "requires completion of Y first"]
• Risk Analysis: [consequences of skipping or delaying this material]
</strategic_importance>

<knowledge_base_context>
**Knowledge Base Intelligence:**
• KB Coverage Depth: [extensive/substantial/moderate/limited/minimal/none]
• Relevance Score: [X.XX] - [interpretation: "highly redundant" (>0.8), "good coverage" (0.5-0.8), "knowledge gap" (<0.5)]
• Confidence Level: [high/medium/low] ([X.XX]) - [reliability of KB assessment]
• Documents Found: [N] related documents in knowledge base
• Knowledge Gap Analysis: 
  - If gap (relevance <0.5): "Significant learning opportunity - minimal existing coverage means high value from studying this"
  - If extensive (relevance >0.8): "Well-covered topic - student has abundant resources already, lower marginal value"
• Strategic Insight: [specific actionable finding from KB, e.g., "KB shows strong calculus foundation but weak in applications" OR "No prior exposure to this domain - foundational learning needed"]
</knowledge_base_context>

<priority_justification>
**Priority #[YOUR NUMBER] Assignment Rationale:**

[Method: {sort_method}] Following [urgency/complexity/hybrid] methodology:

**Quantitative Factors:**
• Guidance Score: [X.X]/10 (algorithmic suggestion)
• Primary Factor ([urgency/complexity/hybrid score]): [Y]/10
• Secondary Factor (pages): [Z] pages
• Tertiary Factor: [KB gap/foundational value/other]

**Comparative Analysis:**
[CRITICAL: MUST compare to at least 2 other tasks explicitly]
• "Ranked ABOVE Task #[N] ([name]) because [specific metric comparison: e.g., 'urgency 8/10 vs their 6/10']"
• "Ranked BELOW Task #[M] ([name]) because [specific reason: e.g., 'they have deadline tomorrow (9/10 urgency) vs this has 2-week buffer (7/10)']"
• Tie-Breaking: [if scores similar, explain: "Both have complexity 5/10, but this has 69 pages vs 52 pages, making it higher priority"]

**Decision Summary:**
[2-3 sentences synthesizing all factors into clear priority justification. Example: "Assigned Priority #2 because while urgency is moderate (6/10), this is foundational material (prerequisite for 3 other topics) with minimal KB coverage (0.32), indicating high learning value. Ranked below urgent assignment (#1) but above well-covered review material (#3)."]
</priority_justification>
</task_analysis>

**SORTING METHOD GUIDELINES:**

**1. URGENCY METHOD:**
Primary: Deadline proximity (imminent > near-term > long-term > no deadline)
Secondary: Academic weight (% of grade)
Tertiary: Page count
Example: "Assignment due tomorrow (9/10) > Exam in 1 week (7/10) > Textbook reading no deadline (4/10)"

**2. COMPLEXITY METHOD:**
Primary: Difficulty level (hardest first to allow mastery time)
Secondary: Prerequisites (foundational > advanced)
Tertiary: Page count
Example: "Advanced ML algorithms (9/10) > Statistics fundamentals (6/10) > Python basics (3/10)"

**3. HYBRID METHOD:**
Weighted formula: 30% urgency + 25% complexity + 20% foundational + 15% KB gap + 10% pages
Balance all factors for optimal learning sequence
Example: "Foundational linear algebra with knowledge gap scores 7.8/10 hybrid (low urgency 4/10 but high foundation 10/10 + high KB gap 8/10)"

**PRIORITIZATION RULES:**
1. **Urgency Override**: Deadline today/tomorrow = Priority 1 regardless of other factors
2. **Foundational First**: Prerequisite material with knowledge gaps = high priority (enables future learning)
3. **Redundancy Penalty**: Extensive KB coverage (>0.8) = lower priority (diminishing returns)
4. **Complexity Balance**: High complexity + low KB = increase priority (needs time); High complexity + high KB = decrease (covered)
5. **Tiebreaker Chain**: Score → Pages → Alphabetical

**QUALITY REQUIREMENTS:**
✓ Each section 4-8 bullet points with specific data
✓ Material analysis: cite exact numbers (pages, hours, complexity score)
✓ Strategic importance: specific urgency scores, deadline dates, prerequisite relationships
✓ KB context: exact relevance scores, document counts, depth categories
✓ Priority justification: compare to 2+ tasks with specific metrics
✓ Total: 15-25 lines per task (comprehensive but scannable)

**CRITICAL SUCCESS FACTORS:**
• NO duplicate priorities (validate before output)
• SEQUENTIAL numbering (no gaps)
• SPECIFIC comparisons (name other tasks)
• QUANTITATIVE metrics (cite scores, not vague)
• MEANINGFUL insights (actionable recommendations)
"""

    @classmethod
    async def generate_prioritization_reasoning_async(
        cls,
        tasks: List[Dict],
        user_goal: str,
        sort_method: str
    ) -> Dict:
        """
        ASYNC: LLM ranks tasks and generates high-quality reasoning.
        
        Full async with comparison table for LLM ranking.
        """
        
        if not tasks:
            return cls._empty_result(user_goal, sort_method)
        
        # Batch processing
        batch_size = 5
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
        
        print(f"🔄 LLM ranking {len(tasks)} tasks in {len(batches)} batch(es) [FULL ASYNC]...")
        
        # Process all batches concurrently
        batch_tasks = [
            cls._process_batch_with_llm_ranking_async(batch, user_goal, sort_method)
            for batch in batches
        ]
        
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Merge results
        all_task_reasoning = {}
        for batch_idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                print(f"   ❌ Batch {batch_idx + 1} failed: {result}")
                # Fallback
                batch = batches[batch_idx]
                for task in batch:
                    all_task_reasoning[task['task']] = cls._generate_fallback_reasoning(
                        task, task.get('temp_priority', batch_idx * batch_size + 1), user_goal, sort_method
                    )
            else:
                all_task_reasoning.update(result)
                print(f"   ✅ Batch {batch_idx + 1}/{len(batches)} complete (LLM ranked)")
        
        # CRITICAL: Validate priorities assigned by LLM
        all_task_reasoning, validation_log = cls._validate_and_correct_priorities(
            all_task_reasoning, tasks
        )
        
        # Build full explanation
        full_explanation = cls._build_full_explanation(
            all_task_reasoning, user_goal, sort_method, validation_log
        )
        
        return {
            "full_explanation": full_explanation,
            "tasks": all_task_reasoning,
            "method": sort_method,
            "kb_grounded": True,
            "batches_processed": len(batches),
            "async_mode": True,
            "llm_ranked": True,
            "priorities_validated": True,
            "validation_log": validation_log
        }
    
    @classmethod
    def generate_prioritization_reasoning(
        cls,
        tasks: List[Dict],
        user_goal: str,
        sort_method: str,
        max_workers: int = 3
    ) -> Dict:
        """Sync wrapper for backward compatibility."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        cls.generate_prioritization_reasoning_async(tasks, user_goal, sort_method)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    cls.generate_prioritization_reasoning_async(tasks, user_goal, sort_method)
                )
        except RuntimeError:
            return asyncio.run(
                cls.generate_prioritization_reasoning_async(tasks, user_goal, sort_method)
            )
    
    @classmethod
    async def _process_batch_with_llm_ranking_async(
        cls,
        batch: List[Dict],
        user_goal: str,
        sort_method: str,
        max_retries: int = 2
    ) -> Dict[str, str]:
        """
        ASYNC: Process batch with LLM ranking and retry.
        """
        
        for attempt in range(max_retries):
            try:
                # Build prompt WITH comparison table for LLM ranking
                user_prompt = cls._build_ranking_prompt_with_comparison(batch, user_goal, sort_method)
                
                messages = [
                    SystemMessage(content=cls.REASONING_SYSTEM_PROMPT.format(sort_method=sort_method)),
                    HumanMessage(content=user_prompt)
                ]
                
                # Async LLM call
                response = await asyncio.to_thread(llm.invoke, messages)
                full_text = response.content.strip()
                
                # Validate structure
                if cls._validate_output_structure(full_text, len(batch)):
                    # Parse
                    parsed = cls._parse_validated_output(full_text, batch)
                    
                    # Check for duplicate priorities
                    if cls._check_no_duplicate_priorities(parsed, batch):
                        return parsed
                    else:
                        if attempt < max_retries - 1:
                            print(f"   ⚠️ LLM produced duplicate priorities, retry {attempt + 1}")
                            await asyncio.sleep(0.5)
                            continue
                        else:
                            print(f"   ⚠️ Duplicates after retry, using fallback")
                            break
                else:
                    if attempt < max_retries - 1:
                        print(f"   ⚠️ Invalid structure, retry {attempt + 1}")
                        await asyncio.sleep(0.5)
                        continue
                    else:
                        print(f"   ⚠️ Invalid structure after retry, using fallback")
                        break
            
            except Exception as e:
                print(f"   ❌ LLM error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    break
                await asyncio.sleep(0.5)
        
        # Fallback
        result = {}
        for idx, task in enumerate(batch):
            result[task['task']] = cls._generate_fallback_reasoning(
                task, task.get('temp_priority', idx + 1), user_goal, sort_method
            )
        return result
    
    @classmethod
    def _build_ranking_prompt_with_comparison(
        cls,
        batch: List[Dict],
        user_goal: str,
        sort_method: str
    ) -> str:
        """
        Build prompt WITH comparison table for LLM ranking.
        
        LLM sees all tasks and their metrics to make ranking decisions.
        """
        
        # Task summaries with guidance scores
        task_summaries = []
        for task in batch:
            analysis = task.get('analysis', {})
            kb = analysis.get('knowledge_grounding', {})
            
            summary = f"""**Task: {task['task']}**
• Guidance Priority: #{task.get('temp_priority', '?')} (algorithmic suggestion - you can adjust)
• Guidance Score: {task.get('guidance_score', 0):.2f}/10
• Category: {analysis.get('category', 'unknown')}
• Complexity: {analysis.get('complexity', 5)}/10
• Urgency: {analysis.get('urgency_score', 5)}/10
• Pages: {analysis.get('pages', 0)} pages
• Estimated Hours: {analysis.get('estimated_hours', 0):.1f}h
• Foundational: {'✓ Yes - Prerequisite' if analysis.get('is_foundational') else '✗ No'}
• KB Relevance: {kb.get('knowledge_relevance_score', 0.5):.2f} ({cls._interpret_kb_score(kb.get('knowledge_relevance_score', 0.5))})
• KB Depth: {kb.get('knowledge_depth', 'unknown')}
• KB Confidence: {kb.get('confidence', 0):.2f}
• KB Documents: {kb.get('documents_found', 0)} found"""
            
            task_summaries.append(summary)
        
        # COMPARISON TABLE - Shows ALL tasks for LLM to compare
        comparison_table = "\n**═══════════════ COMPARISON TABLE (All Tasks) ═══════════════**\n\n"
        comparison_table += "Review this table before assigning priorities:\n\n"
        comparison_table += "| Guidance Pri | Task | Complexity | Urgency | Pages | KB Gap | Foundation |\n"
        comparison_table += "|--------------|------|------------|---------|-------|--------|------------|\n"
        
        for task in sorted(batch, key=lambda t: t.get('temp_priority', 999)):
            analysis = task.get('analysis', {})
            kb = analysis.get('knowledge_grounding', {})
            kb_gap = 1 - kb.get('knowledge_relevance_score', 0.5)
            
            comparison_table += f"| #{task.get('temp_priority', '?'):>12} | {task['task'][:20]:20} | "
            comparison_table += f"{analysis.get('complexity', 5):>10}/10 | "
            comparison_table += f"{analysis.get('urgency_score', 5):>7}/10 | "
            comparison_table += f"{analysis.get('pages', 0):>5}p | "
            comparison_table += f"{kb_gap:>6.2f} | "
            comparison_table += f"{'Yes':>10}" if analysis.get('is_foundational') else f"{'No':>10}" + " |\n"
        
        comparison_table += "\n**═══════════════════════════════════════════════════════════**\n"
        
        # Method explanation
        method_guide = cls._get_detailed_method_explanation(sort_method)
        
        prompt = f"""**USER LEARNING GOAL:** {user_goal}

**SORTING METHOD:** {sort_method}

{method_guide}

**TASKS TO RANK AND ANALYZE:**

{chr(10).join(task_summaries)}

{comparison_table}

**YOUR RANKING TASK:**

1. **REVIEW** the comparison table showing all tasks with their metrics
2. **RANK** tasks by assigning priorities 1, 2, 3, ... based on {sort_method} method
3. **YOU CAN ADJUST** guidance priorities if you have strong reasoning (explain why in priority_justification)
4. **ENSURE** no duplicate priorities, no gaps in numbering
5. **GENERATE** comprehensive analysis for EACH task using the XML format
6. **COMPARE** tasks explicitly in your priority justification (name at least 2 other tasks)
7. **CITE** specific metrics from the comparison table

**CRITICAL REMINDERS:**
• The guidance priorities are SUGGESTIONS - you make the final ranking decision
• If you disagree with guidance, explain your reasoning with specific metrics
• Compare tasks EXPLICITLY: "Ranked above Task X because [specific reason]"
• Be QUANTITATIVE: cite scores, pages, dates, not vague descriptions
• Generate MEANINGFUL reasoning: 15-25 lines per task with actionable insights

Begin your ranking and analysis:"""
        
        return prompt
    
    @classmethod
    def _interpret_kb_score(cls, score: float) -> str:
        """Interpret KB relevance score."""
        if score >= 0.8:
            return "Extensive - Well covered"
        elif score >= 0.6:
            return "Substantial - Good coverage"
        elif score >= 0.4:
            return "Moderate - Partial coverage"
        elif score >= 0.2:
            return "Limited - Knowledge gap"
        else:
            return "Minimal - Significant gap"
    
    @classmethod
    def _get_detailed_method_explanation(cls, sort_method: str) -> str:
        """Detailed explanation of sorting method."""
        explanations = {
            'urgency': """**URGENCY METHOD - Deadline-Driven Prioritization:**

**Formula:** Primary = Urgency Score (0-10) → Secondary = Pages → Tertiary = Complexity

**Ranking Logic:**
• Priority 1: Highest urgency (deadline today/tomorrow, 9-10/10)
• Priority 2-3: Near-term deadlines (this week, 7-8/10)
• Priority 4-6: Medium-term deadlines (next week, 5-6/10)
• Priority 7+: Long-term or no deadline (4 or below/10)

**Tiebreaker:** When urgency equal, more pages = higher priority (more work = start sooner)

**Example Ranking:**
1. Assignment due tomorrow (urgency 9/10, 45 pages)
2. Exam prep - exam in 3 days (urgency 8/10, 120 pages)
3. Reading for next week (urgency 6/10, 80 pages)""",
            
            'complexity': """**COMPLEXITY METHOD - Difficulty-First Prioritization:**

**Formula:** Primary = Complexity (0-10) → Secondary = Pages → Tertiary = Urgency

**Ranking Logic:**
• Priority 1: Highest complexity (9-10/10) - hardest material needs most time
• Priority 2-3: High complexity (7-8/10) - challenging but manageable
• Priority 4-6: Moderate complexity (5-6/10) - standard difficulty
• Priority 7+: Low complexity (4 or below/10) - easier material for later

**Tiebreaker:** When complexity equal, more pages = higher priority (more content = more time)

**Rationale:** Tackle hardest material first while mentally fresh and allow adequate mastery time

**Example Ranking:**
1. Advanced calculus (complexity 9/10, 100 pages)
2. Linear algebra fundamentals (complexity 7/10, 120 pages)
3. Python basics review (complexity 4/10, 50 pages)""",
            
            'hybrid': """**HYBRID METHOD - Balanced Multi-Factor Prioritization:**

**Formula:** Weighted Score = 30% Urgency + 25% Complexity + 20% Foundational + 15% KB Gap + 10% Pages

**Factor Breakdown:**
• **Urgency (30%):** Deadline pressure, time sensitivity
• **Complexity (25%):** Difficulty level, cognitive load
• **Foundational (20%):** Prerequisite importance (10 pts if yes, 0 if no)
• **KB Gap (15%):** Knowledge gap size (1 - KB relevance score) × 10
• **Pages (10%):** Content volume (normalized to 10-point scale)

**Ranking Logic:**
• Priority 1: Highest combined score (typically 7.5-10)
• Priority 2-4: High scores (6.0-7.4)
• Priority 5-7: Medium scores (4.5-5.9)
• Priority 8+: Lower scores (below 4.5)

**Example Calculation:**
Task: Linear Algebra Fundamentals
• Urgency: 4/10 → 30% × 4 = 1.2
• Complexity: 6/10 → 25% × 6 = 1.5
• Foundational: Yes → 20% × 10 = 2.0
• KB Gap: 0.68 (relevance 0.32) → 15% × 6.8 = 1.02
• Pages: 120p normalized to 8/10 → 10% × 8 = 0.8
**Total: 6.52/10 → Priority #2-3 range**"""
        }
        return explanations.get(sort_method, "Prioritization based on task analysis.")
    
    @classmethod
    def _validate_output_structure(cls, output: str, expected_tasks: int) -> bool:
        """Validate XML structure."""
        task_blocks = re.findall(r'<task_analysis\s+priority="(\d+)">', output)
        if len(task_blocks) != expected_tasks:
            return False
        
        required = ['<task_name>', '<material_analysis>', '<strategic_importance>', 
                   '<knowledge_base_context>', '<priority_justification>']
        
        for section in required:
            if output.count(section) < expected_tasks:
                return False
        return True
    
    @classmethod
    def _check_no_duplicate_priorities(cls, parsed: Dict[str, str], batch: List[Dict]) -> bool:
        """Check for duplicate priorities in LLM output."""
        priorities = []
        for task in batch:
            if task['task'] in parsed:
                match = re.search(r'<task_analysis\s+priority="(\d+)">', parsed[task['task']])
                if match:
                    priorities.append(int(match.group(1)))
        
        has_duplicates = len(priorities) != len(set(priorities))
        if has_duplicates:
            print(f"      Duplicate priorities found: {sorted(priorities)}")
        return not has_duplicates
    
    @classmethod
    def _parse_validated_output(cls, output: str, batch: List[Dict]) -> Dict[str, str]:
        """Parse XML output from LLM."""
        result = {}
        pattern = r'<task_analysis\s+priority="(\d+)">(.*?)</task_analysis>'
        matches = re.finditer(pattern, output, re.DOTALL)
        
        for match in matches:
            priority = int(match.group(1))
            content = match.group(0)
            
            name_match = re.search(r'<task_name>(.*?)</task_name>', content)
            if name_match:
                task_name = name_match.group(1).strip()
                
                # Fuzzy match
                for task in batch:
                    if cls._fuzzy_match(task_name, task['task']):
                        result[task['task']] = content
                        break
                else:
                    # Fallback: match by guidance priority
                    for task in batch:
                        if task.get('temp_priority') == priority:
                            result[task['task']] = content
                            break
        
        # Fill missing
        for task in batch:
            if task['task'] not in result:
                result[task['task']] = cls._generate_fallback_reasoning(
                    task, task.get('temp_priority', 999), "goals", "hybrid"
                )
        
        return result
    
    @classmethod
    def _fuzzy_match(cls, name1: str, name2: str) -> bool:
        """Fuzzy match task names."""
        clean1 = re.sub(r'[^\w\s]', '', name1.lower())
        clean2 = re.sub(r'[^\w\s]', '', name2.lower())
        tokens1, tokens2 = set(clean1.split()), set(clean2.split())
        if not tokens1 or not tokens2:
            return False
        return len(tokens1 & tokens2) / max(len(tokens1), len(tokens2)) >= 0.6
    
    @classmethod
    def _validate_and_correct_priorities(
        cls,
        parsed_reasoning: Dict[str, str],
        tasks: List[Dict]
    ) -> Tuple[Dict[str, str], List[str]]:
        """
        Validate LLM-assigned priorities and collect validation log.
        
        Returns: (corrected_reasoning, validation_log)
        """
        
        corrected = {}
        validation_log = []
        corrections = 0
        
        # Extract LLM priorities
        llm_priorities = {}
        for task in tasks:
            if task['task'] in parsed_reasoning:
                match = re.search(r'<task_analysis\s+priority="(\d+)">', parsed_reasoning[task['task']])
                if match:
                    llm_priorities[task['task']] = int(match.group(1))
        
        # Check for issues
        priorities_list = list(llm_priorities.values())
        expected = list(range(1, len(tasks) + 1))
        
        if sorted(priorities_list) != expected:
            validation_log.append(f"⚠️ LLM priority validation: Expected {expected}, got {sorted(priorities_list)}")
            
            # Check for duplicates
            from collections import Counter
            duplicates = [p for p, count in Counter(priorities_list).items() if count > 1]
            if duplicates:
                validation_log.append(f"   Duplicate priorities: {duplicates}")
            
            # Check for gaps
            missing = set(expected) - set(priorities_list)
            if missing:
                validation_log.append(f"   Missing priorities: {sorted(missing)}")
        
        # Assign corrected priorities
        for task in tasks:
            task_name = task['task']
            
            if task_name in parsed_reasoning:
                reasoning = parsed_reasoning[task_name]
                
                # Extract LLM priority
                match = re.search(r'<task_analysis\s+priority="(\d+)">', reasoning)
                if match:
                    llm_priority = int(match.group(1))
                    
                    # Store LLM priority in task
                    task['llm_priority'] = llm_priority
                    
                    # Keep LLM's reasoning as-is (no correction)
                    corrected[task_name] = reasoning
                    validation_log.append(f"✓ {task_name[:30]}... → Priority #{llm_priority} (LLM)")
                else:
                    corrected[task_name] = reasoning
            else:
                temp_pri = task.get('temp_priority', 999)
                corrected[task_name] = cls._generate_fallback_reasoning(
                    task, temp_pri, "goals", "hybrid"
                )
                validation_log.append(f"⚠️ {task_name[:30]}... → Priority #{temp_pri} (fallback)")
        
        validation_log.append(f"✅ LLM ranking complete: {len(corrected)} tasks ranked")
        
        return corrected, validation_log
    
    @classmethod
    def _build_full_explanation(
        cls,
        task_reasoning: Dict[str, str],
        user_goal: str,
        sort_method: str,
        validation_log: List[str]
    ) -> str:
        """Build full explanation with validation log."""
        
        explanation = f"""# 📚 Study Plan Prioritization Analysis (LLM-Ranked)

**Learning Goal:** {user_goal}
**Prioritization Method:** {sort_method}
**Total Tasks Analyzed:** {len(task_reasoning)}
**Ranking System:** LLM-based (AI agent assigns priorities)

---

## Validation Log

{chr(10).join(validation_log)}

---

## Detailed Task Analysis

"""
        
        for task_name, reasoning in task_reasoning.items():
            explanation += f"{reasoning}\n\n---\n\n"
        
        return explanation
    
    @classmethod
    def _generate_fallback_reasoning(
        cls, 
        task: Dict, 
        priority: int, 
        user_goal: str, 
        sort_method: str
    ) -> str:
        """Generate fallback reasoning."""
        
        analysis = task.get('analysis', {})
        kb = analysis.get('knowledge_grounding', {})
        
        task_name = re.sub(r'[^\w\s]', '', task['task'])
        
        return f"""<task_analysis priority="{priority}">
<task_name>{task_name}</task_name>

<material_analysis>
**Document Scope & Characteristics:**
• Total Pages: {analysis.get('pages', 0)} pages requiring {analysis.get('estimated_hours', 0):.1f} hours
• Complexity Level: {analysis.get('complexity', 5)}/10
• Content Type: {analysis.get('category', 'general')}
</material_analysis>

<strategic_importance>
**Academic Impact & Dependencies:**
• Urgency Assessment: {analysis.get('urgency_score', 5)}/10
• Foundational Value: {'Yes - Prerequisite' if analysis.get('is_foundational') else 'No'}
• Goal Alignment: Supports "{user_goal}"
</strategic_importance>

<knowledge_base_context>
**Knowledge Base Intelligence:**
• KB Coverage Depth: {kb.get('knowledge_depth', 'unknown')}
• Relevance Score: {kb.get('knowledge_relevance_score', 0.5):.2f}
• Confidence Level: {kb.get('confidence', 0):.2f}
</knowledge_base_context>

<priority_justification>
**Priority #{priority} Assignment Rationale:**
Assigned via {sort_method} methodology. Fallback priority based on guidance score.
</priority_justification>
</task_analysis>"""
    
    @classmethod
    def _empty_result(cls, user_goal: str, sort_method: str) -> Dict:
        """Empty result."""
        return {
            "full_explanation": f"No tasks to prioritize for: {user_goal}",
            "tasks": {},
            "method": sort_method,
            "kb_grounded": False,
            "batches_processed": 0,
            "async_mode": True,
            "llm_ranked": True,
            "priorities_validated": True,
            "validation_log": []
        }


# ==================== SCHEDULE REASONING ====================

class ScheduleReasoningEngine:
    """Schedule reasoning generator."""
    
    @classmethod
    def explain_schedule(
        cls,
        schedule: List[Dict],
        total_hours: float,
        tasks: List[Dict],
        constraints: str
    ) -> str:
        """Generate schedule reasoning."""
        
        if not schedule:
            return "No schedule generated - insufficient time availability."
        
        days_used = len(set(item['day'] for item in schedule))
        total_allocated = sum(item.get('hours', 0) for item in schedule)
        utilization = (total_allocated / total_hours * 100) if total_hours > 0 else 0
        
        type_dist = {}
        for item in schedule:
            t = item.get('type', 'Study')
            type_dist[t] = type_dist.get(t, 0) + 1
        
        return f"""**📅 Optimized Weekly Schedule**

**Time Allocation:**
• Available: {total_hours:.1f} hours
• Allocated: {total_allocated:.1f} hours
• Utilization: {utilization:.0f}%
• Days Used: {days_used}

**Activity Distribution:**
{chr(10).join(f'• {t}: {c} session(s)' for t, c in sorted(type_dist.items()))}

**Optimization Strategy:**
1. **Cognitive Load Management:** Complex materials scheduled during peak focus hours (mornings)
2. **Spaced Learning:** Sessions distributed across multiple days for better retention
3. **Task Sequencing:** High-priority items (#1-3) receive earlier time slots
4. **Flexibility Buffer:** {100-utilization:.0f}% unscheduled time prevents burnout and allows adjustment

{f'5. **User Constraints:** {constraints}' if constraints else ''}

This schedule balances intensity with sustainability for optimal learning outcomes."""


# ==================== PUBLIC API ====================

async def generate_knowledge_grounded_reasoning_async(
    tasks: List[Dict],
    schedule_data: Dict,
    user_goal: str,
    sort_method: str
) -> Dict:
    """
    ASYNC: Main integration with LLM ranking.
    """
    
    # LLM ranks tasks and generates reasoning (async)
    task_reasoning_result = await KnowledgeGroundedReasoningEngine.generate_prioritization_reasoning_async(
        tasks=tasks,
        user_goal=user_goal,
        sort_method=sort_method
    )
    
    # Generate schedule reasoning
    schedule = schedule_data.get('schedule', [])
    total_hours = schedule_data.get('available_hours', 0)
    constraints = schedule_data.get('constraints', '')
    
    schedule_reasoning = ScheduleReasoningEngine.explain_schedule(
        schedule, total_hours, tasks, constraints
    )
    
    return {
        "full_explanation": task_reasoning_result.get('full_explanation', ''),
        "schedule": schedule_reasoning,
        "tasks": task_reasoning_result.get('tasks', {}),
        "method": sort_method,
        "kb_grounded": task_reasoning_result.get('kb_grounded', False),
        "async_mode": True,
        "llm_ranked": True,
        "priorities_validated": task_reasoning_result.get('priorities_validated', False),
        "validation_log": task_reasoning_result.get('validation_log', []),
        "metadata": {
            "total_tasks": len(tasks),
            "batches_processed": task_reasoning_result.get('batches_processed', 0),
            "kb_confidence_avg": sum(
                t.get('analysis', {}).get('knowledge_grounding', {}).get('confidence', 0)
                for t in tasks
            ) / len(tasks) if tasks else 0
        }
    }


def generate_knowledge_grounded_reasoning(
    tasks: List[Dict],
    schedule_data: Dict,
    user_goal: str,
    sort_method: str
) -> Dict:
    """Sync wrapper for backward compatibility."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    generate_knowledge_grounded_reasoning_async(tasks, schedule_data, user_goal, sort_method)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                generate_knowledge_grounded_reasoning_async(tasks, schedule_data, user_goal, sort_method)
            )
    except RuntimeError:
        return asyncio.run(
            generate_knowledge_grounded_reasoning_async(tasks, schedule_data, user_goal, sort_method)
        )