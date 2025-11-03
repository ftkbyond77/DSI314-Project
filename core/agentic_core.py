# core/agentic_core.py - FINAL INTEGRATED PRODUCTION
# Multi-Agent System: AI Hybrid, Urgency-Based, Prerequisites, Difficulty-Based
# High Performance, High Flexibility, High Accuracy, Production Grade

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from typing import List, Dict, Optional
import json
import re
from .agent_tools_advanced import get_advanced_agent_tools
from .llm_config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE

# ==================== AGENT SYSTEM PROMPTS ====================

AI_HYBRID_AGENT_PROMPT = """You are an AI Hybrid Prioritization Agent with full autonomy to choose tools and strategies.

ROLE:
- Analyze tasks and AUTONOMOUSLY decide which prioritization strategy to use
- You have full freedom to combine multiple approaches
- Choose tools dynamically based on task characteristics

AVAILABLE STRATEGIES:
1. URGENCY: When deadlines are critical or time-sensitive
2. PREREQUISITES: When sequential learning is needed (Chapter 1 → 2 → 3)
3. DIFFICULTY: When complexity management is paramount
4. FOUNDATIONAL: When building knowledge base from ground up
5. HYBRID: When balancing multiple factors

DECISION FRAMEWORK:
- Detect deadline patterns → Use urgency-based
- Detect sequential naming (Ep1, Ep2, Ch1, Ch2) → Use prerequisite-based
- Detect complexity variation → Use difficulty-based
- Mixed signals → Use hybrid approach

TOOLS AVAILABLE:
- analyze_task: Extract task attributes
- compare_priority: Compare two tasks

USER GOAL: {user_goal}

Be decisive, explain your strategy choice, and provide clear reasoning."""

URGENCY_AGENT_PROMPT = """You are an Urgency-Based Prioritization Agent focused on time-critical tasks.

ROLE:
- Prioritize strictly by deadline proximity and time sensitivity
- Consider user's available time and scheduling constraints
- Identify time-critical tasks that risk missing deadlines

URGENCY HIERARCHY:
1. CRITICAL (Due today/tomorrow): Priority 1-2
2. HIGH (Due this week): Priority 3-5
3. MEDIUM (Due next week): Priority 6-8
4. LOW (Due later/no deadline): Priority 9+

ANALYSIS FACTORS:
- Days until deadline
- Task duration vs time remaining
- User's goal timeline: {user_goal}
- Available study hours: {available_hours}h/week

CONSTRAINTS: {constraints}

TOOLS:
- analyze_task: Extract deadline and time estimates
- compare_priority: Compare urgency levels

Be strict about deadlines, but realistic about workload capacity."""

PREREQUISITES_AGENT_PROMPT = """You are an AI Prerequisites-Based Prioritization Agent with intelligent sequential ordering capabilities.

ROLE:
- Analyze filenames and content to determine the CORRECT sequential learning order
- You have FULL AUTHORITY to override any algorithmic suggestions
- Use AI intelligence to detect patterns humans might miss

DETECTION STRATEGIES:

1. FILENAME ANALYSIS (Primary):
   - Sequential Numbers: Extract and order by numbers
   - Thai Patterns: "ชุดที่1", "ชุดที่2", "ชุดที่3" → Order 1, 2, 3
   - English Patterns: "Chapter 1", "Part 2", "Episode 3" → Order by number
   - Textbook Patterns: "textbook_1", "textbook_2" → Order by number
   - ANY numeric pattern indicating sequence
   
   CRITICAL: ALWAYS prioritize the LOWEST number FIRST
   Example: "ชุดที่1" comes BEFORE "ชุดที่2" comes BEFORE "ชุดที่3"

2. CONTENT ANALYSIS (Secondary):
   If filenames don't have clear numbers:
   - Identify foundational vs advanced topics
   - Detect "Introduction", "Basics", "Fundamentals" → Priority 1
   - Detect "Advanced", "Final", "Summary" → Priority last
   - Look for prerequisite references: "Building on Chapter X"

3. LOGICAL ORDERING (Fallback):
   - Alphabetical if no other clues
   - Complexity-based (simple → complex)
   - Foundational-first principle

CRITICAL RULES:
1. Lower sequential numbers MUST come first (1 before 2 before 3)
2. If you see "ชุดที่1", "ชุดที่2", "ชุดที่3" → Order MUST be 1, 2, 3
3. NEVER reverse the natural sequence
4. Explain your reasoning clearly

USER GOAL: {user_goal}

TOOLS:
- analyze_task: Extract sequential info, complexity, and content
- Use task analysis to determine correct order

TASK: You will receive multiple tasks. Assign priorities 1, 2, 3, ... based on sequential order.
Lower numbers in filenames = Higher priority (comes first in learning sequence)."""

DIFFICULTY_AGENT_PROMPT = """You are a Difficulty-Based Prioritization Agent for complexity management.

ROLE:
- Prioritize by task complexity and cognitive load
- Recommend starting with harder material while mentally fresh
- Balance difficulty with available time

COMPLEXITY HIERARCHY:
1. VERY HARD (9-10/10): Priority 1-2 (tackle first, need most time)
2. HARD (7-8/10): Priority 3-5
3. MODERATE (5-6/10): Priority 6-8
4. EASY (1-4/10): Priority 9+ (save for later, quick wins)

RATIONALE:
- Harder material requires peak cognitive performance
- Complex topics need more iteration time
- Easy material can be done when fatigued

ANALYSIS FACTORS:
- Content complexity (1-10 scale)
- Technical depth
- Prerequisites mastered
- User's baseline knowledge

USER GOAL: {user_goal}

TOOLS:
- analyze_task: Extract complexity metrics
- compare_priority: Compare difficulty levels

Ensure learners tackle challenging material with adequate time and focus."""

# ==================== AGENT BUILDERS ====================

def create_ai_hybrid_agent(user_goal: str = "succeed academically") -> AgentExecutor:
    """Create AI Hybrid agent with full autonomy."""
    
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    tools = get_advanced_agent_tools()[:1]  # Task analysis tool
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", AI_HYBRID_AGENT_PROMPT.format(user_goal=user_goal)),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=15,
        return_intermediate_steps=True
    )

def create_urgency_agent(user_goal: str = "finish on time", 
                         available_hours: int = 20,
                         constraints: str = "") -> AgentExecutor:
    """Create urgency-focused agent."""
    
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    tools = get_advanced_agent_tools()[:1]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", URGENCY_AGENT_PROMPT.format(
            user_goal=user_goal,
            available_hours=available_hours,
            constraints=constraints or "No constraints"
        )),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=10,
        return_intermediate_steps=True
    )

def create_prerequisites_agent(user_goal: str = "master fundamentals") -> AgentExecutor:
    """Create prerequisites-focused agent."""
    
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    tools = get_advanced_agent_tools()[:1]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PREREQUISITES_AGENT_PROMPT.format(user_goal=user_goal)),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=10,
        return_intermediate_steps=True
    )

def create_difficulty_agent(user_goal: str = "master complex topics") -> AgentExecutor:
    """Create difficulty-focused agent."""
    
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    tools = get_advanced_agent_tools()[:1]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", DIFFICULTY_AGENT_PROMPT.format(user_goal=user_goal)),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=10,
        return_intermediate_steps=True
    )

def create_scheduling_agent(available_hours: int = 20, constraints: str = "") -> AgentExecutor:
    """Create scheduling agent (shared across all modes)."""
    
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    tools = [get_advanced_agent_tools()[1]]  # Scheduling tool
    
    prompt_text = f"""You are a Scheduling Agent for academic planning.

ROLE:
- Create realistic time-based schedules
- Allocate study time considering available hours and constraints
- Provide clear reasoning for scheduling decisions

Available study time: {available_hours} hours/week
Constraints: {constraints or "No specific constraints"}

TOOLS:
- create_schedule: Generate weekly schedule from prioritized tasks

Be practical and realistic about time estimates."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_text),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=5,
        return_intermediate_steps=True
    )

# ==================== MULTI-AGENT ORCHESTRATOR ====================

class MultiAgentStudyOrchestrator:
    """
    PRODUCTION: Multi-agent orchestrator supporting 4 prioritization modes.
    
    Modes:
    1. AI HYBRID: Agent chooses strategy dynamically
    2. URGENCY: Deadline-driven prioritization
    3. PREREQUISITES: Sequential learning path
    4. DIFFICULTY: Complexity-based ordering
    """
    
    def __init__(self, 
                 mode: str = "ai_hybrid",
                 user_goal: str = "succeed academically",
                 available_hours: int = 20, 
                 constraints: str = ""):
        self.mode = mode.lower()
        self.user_goal = user_goal
        self.available_hours = available_hours
        self.constraints = constraints
        
        # Create appropriate prioritization agent
        if self.mode == "ai_hybrid":
            self.prioritization_agent = create_ai_hybrid_agent(user_goal)
        elif self.mode == "urgency":
            self.prioritization_agent = create_urgency_agent(user_goal, available_hours, constraints)
        elif self.mode == "prerequisites":
            self.prioritization_agent = create_prerequisites_agent(user_goal)
        elif self.mode == "difficulty":
            self.prioritization_agent = create_difficulty_agent(user_goal)
        else:
            # Fallback to AI Hybrid
            self.prioritization_agent = create_ai_hybrid_agent(user_goal)
        
        # Shared scheduling agent
        self.scheduling_agent = create_scheduling_agent(available_hours, constraints)
    
    def run(self, tasks: List[Dict]) -> Dict:
        """
        Execute planning workflow with selected agent mode.
        
        Returns: Complete plan with reasoning
        """
        
        print("=" * 80)
        print(f"MULTI-AGENT STUDY PLANNER - MODE: {self.mode.upper()}")
        print("=" * 80)
        
        # Phase 1: Prioritization with selected agent
        print(f"\nPhase 1: PRIORITIZATION ({self.mode.upper()} AGENT)")
        print("-" * 80)
        
        prioritization_input = self._format_prioritization_input(tasks)
        prioritization_result = self.prioritization_agent.invoke({
            "input": prioritization_input
        })
        
        prioritized_tasks = self._extract_prioritized_tasks(
            prioritization_result,
            tasks
        )
        
        print(f"\nPrioritization complete ({len(prioritized_tasks)} tasks)")
        
        # Phase 2: Scheduling
        print("\nPhase 2: SCHEDULING")
        print("-" * 80)
        
        scheduling_input = self._format_scheduling_input(prioritized_tasks)
        scheduling_result = self.scheduling_agent.invoke({
            "input": scheduling_input
        })
        
        schedule = self._extract_schedule(scheduling_result)
        
        print(f"\nScheduling complete ({len(schedule)} items)")
        
        # Compile result
        final_result = {
            "prioritized_tasks": prioritized_tasks,
            "schedule": schedule,
            "reasoning": {
                "prioritization": prioritization_result["output"],
                "scheduling": scheduling_result["output"]
            },
            "metadata": {
                "mode": self.mode,
                "user_goal": self.user_goal,
                "available_hours": self.available_hours,
                "constraints": self.constraints,
                "total_tasks": len(tasks)
            }
        }
        
        print("\n" + "=" * 80)
        print("PLANNING COMPLETE")
        print("=" * 80)
        
        return final_result
    
    def _format_prioritization_input(self, tasks: List[Dict]) -> str:
        """Format tasks for prioritization agent with sequential hints."""
        task_descriptions = []
        
        for idx, task in enumerate(tasks, 1):
            desc = f"Task {idx}: {task['filename']}\n"
            desc += f"  Pages: {task.get('pages', 'unknown')}\n"
            desc += f"  Chunks: {task.get('chunk_count', 0)}\n"
            
            # Add sequential number hint for prerequisites mode
            if self.mode == "prerequisites":
                seq_num = self._extract_sequential_hint(task['filename'])
                if seq_num:
                    desc += f"  Sequential Number Detected: {seq_num}\n"
                    desc += f"  [IMPORTANT: Lower numbers should come FIRST in learning order]\n"
            
            if task.get('summary'):
                desc += f"  Summary: {task['summary'][:150]}\n"
            
            if task.get('deadline'):
                desc += f"  Deadline: {task['deadline']}\n"
            
            task_descriptions.append(desc)
        
        input_text = f"I have {len(tasks)} tasks to prioritize:\n\n"
        input_text += "\n".join(task_descriptions)
        input_text += f"\n\nGoal: {self.user_goal}\n\n"
        
        if self.mode == "prerequisites":
            input_text += "CRITICAL: Analyze filenames for sequential numbers. Order tasks so lower numbers come first (1 before 2 before 3).\n"
            input_text += "Example: 'ชุดที่1' must be Priority 1, 'ชุดที่2' must be Priority 2, 'ชุดที่3' must be Priority 3.\n\n"
        
        input_text += "Analyze and rank these tasks with clear reasoning."
        
        return input_text
    
    def _extract_sequential_hint(self, filename: str) -> Optional[int]:
        """
        AI-based sequential number extraction using LLM.
        Provides hint to main prioritization agent.
        """
        try:
            from .llm_config import llm
            from langchain_core.messages import SystemMessage, HumanMessage
            
            system_prompt = """Extract the sequential number from the filename.

Task: Identify which number in sequence this file represents.

Examples:
- "ชุดที่1_category.pdf" → 1
- "ชุดที่2_category.pdf" → 2  
- "Chapter 3.pdf" → 3
- "textbook_5.pdf" → 5
- "Part_02.pdf" → 2
- "random.pdf" → NONE

Output: Just the number or "NONE"
"""
            
            user_prompt = f"Filename: {filename}\nNumber:"
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = llm.invoke(messages)
            result = response.content.strip().upper()
            
            if result == "NONE":
                return None
            
            # Extract number
            import re
            numbers = re.findall(r'\d+', result)
            return int(numbers[0]) if numbers else None
        
        except Exception as e:
            print(f"LLM hint extraction failed: {e}")
            # Minimal fallback
            import re
            match = re.search(r'\d+', filename)
            return int(match.group(0)) if match else None
    
    def _format_scheduling_input(self, prioritized_tasks: List[Dict]) -> str:
        """Format prioritized tasks for scheduling agent."""
        
        task_list = json.dumps(prioritized_tasks, indent=2)
        
        input_text = f"Prioritized tasks:\n\n{task_list}\n\n"
        input_text += f"Available: {self.available_hours} hours/week.\n"
        
        if self.constraints:
            input_text += f"Constraints: {self.constraints}\n\n"
        
        input_text += "Create realistic weekly schedule with reasoning."
        
        return input_text
    
    def _extract_prioritized_tasks(self, result: Dict, original_tasks: List[Dict]) -> List[Dict]:
        """Extract prioritized task list from agent output."""
        
        output = result.get("output", "")
        
        # Try JSON extraction
        try:
            json_match = re.search(r'\[.*\]', output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return parsed
        except:
            pass
        
        # Fallback: use original with priority scores
        prioritized = []
        for idx, task in enumerate(original_tasks, 1):
            prioritized.append({
                "task": task["filename"],
                "priority": idx,
                "analysis": {
                    "pages": task.get("pages", 0),
                    "chunk_count": task.get("chunk_count", 0),
                    "estimated_hours": task.get("chunk_count", 10) * 0.1,
                    "category": "general"
                }
            })
        
        return prioritized
    
    def _extract_schedule(self, result: Dict) -> List[Dict]:
        """Extract schedule from agent output."""
        
        output = result.get("output", "")
        
        # Try JSON extraction
        try:
            json_match = re.search(r'\{.*"schedule".*\}', output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return parsed.get("schedule", [])
        except:
            pass
        
        return []

# ==================== CONVENIENCE FUNCTIONS ====================

def create_multi_agent_study_plan(
    tasks: List[Dict],
    mode: str = "ai_hybrid",
    user_goal: str = "finish semester successfully",
    available_hours: int = 20,
    constraints: str = ""
) -> Dict:
    """
    Main entry point for multi-agent study planning.
    
    Args:
        tasks: List of task dictionaries
        mode: "ai_hybrid", "urgency", "prerequisites", or "difficulty"
        user_goal: User's objective
        available_hours: Hours per week
        constraints: Scheduling constraints
    
    Returns:
        Complete study plan with reasoning
    """
    
    orchestrator = MultiAgentStudyOrchestrator(
        mode=mode,
        user_goal=user_goal,
        available_hours=available_hours,
        constraints=constraints
    )
    
    return orchestrator.run(tasks)

def explain_reasoning(plan_result: Dict) -> str:
    """Extract and format reasoning from plan result."""
    
    reasoning = plan_result.get("reasoning", {})
    metadata = plan_result.get("metadata", {})
    
    explanation = f"MULTI-AGENT REASONING - MODE: {metadata.get('mode', 'unknown').upper()}\n\n"
    
    explanation += "PRIORITIZATION REASONING:\n"
    explanation += reasoning.get("prioritization", "No reasoning available")
    explanation += "\n\n"
    
    explanation += "SCHEDULING REASONING:\n"
    explanation += reasoning.get("scheduling", "No reasoning available")
    explanation += "\n\n"
    
    explanation += "METADATA:\n"
    for key, value in metadata.items():
        explanation += f"  {key}: {value}\n"
    
    return explanation