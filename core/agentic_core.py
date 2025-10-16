# core/agentic_core.py - Reasoning-Driven Agentic AI System

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage
from typing import List, Dict
import json
from .agent_tools import get_agent_tools
from .llm_config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE

# ==================== AGENT PROMPTS ====================

PRIORITIZATION_AGENT_PROMPT = """You are an expert Prioritization Agent for academic study planning.

Your role:
- Analyze multiple tasks/documents and rank them by importance
- Provide clear, logical reasoning for each priority decision
- Consider: urgency, foundational importance, topic dependencies, user goals

Your reasoning should follow this structure:
1. **Analysis**: What are the key attributes of each task?
2. **Comparison**: How do tasks compare against each other?
3. **Decision**: Which task should be prioritized and why?
4. **Justification**: Explain the logical reasoning behind the decision

Always think step-by-step and make your reasoning transparent.

Available tools:
- analyze_task: Extract attributes from a single task
- compare_priority: Compare two tasks and determine priority

User's main goal: {user_goal}

Be decisive but explain your reasoning clearly."""

SCHEDULING_AGENT_PROMPT = """You are an expert Scheduling Agent for academic planning.

Your role:
- Take prioritized tasks and create a realistic time-based schedule
- Allocate study time considering available hours and constraints
- Provide clear reasoning for scheduling decisions

Your reasoning should explain:
1. **Why** you scheduled high-priority items at specific times
2. **How** you balanced workload across the week
3. **What** constraints you considered (deadlines, preferences, capacity)

Be practical and realistic about time estimates.

Available tools:
- create_schedule: Generate a weekly schedule from prioritized tasks

User's available study time: {available_hours} hours/week
Additional constraints: {constraints}

Think about the most efficient path to achieve the user's goal."""


# ==================== AGENT BUILDERS ====================

def create_prioritization_agent(user_goal: str = "succeed academically") -> AgentExecutor:
    """Create the prioritization agent with reasoning capabilities"""
    
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    tools = get_agent_tools()[:2]  # Only task analysis and priority comparison
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", PRIORITIZATION_AGENT_PROMPT.format(user_goal=user_goal)),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=10,
        return_intermediate_steps=True
    )
    
    return agent_executor


def create_scheduling_agent(available_hours: int = 20, constraints: str = "") -> AgentExecutor:
    """Create the scheduling agent with reasoning capabilities"""
    
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    tools = [get_agent_tools()[2]]  # Only scheduling tool
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SCHEDULING_AGENT_PROMPT.format(
            available_hours=available_hours,
            constraints=constraints or "No specific constraints"
        )),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        return_intermediate_steps=True
    )
    
    return agent_executor


# ==================== MULTI-AGENT ORCHESTRATOR ====================

class StudyPlanningOrchestrator:
    """
    Orchestrates prioritization and scheduling agents to create
    a complete study plan with transparent reasoning.
    """
    
    def __init__(self, user_goal: str = "succeed academically",
                 available_hours: int = 20, constraints: str = ""):
        self.user_goal = user_goal
        self.available_hours = available_hours
        self.constraints = constraints
        
        self.prioritization_agent = create_prioritization_agent(user_goal)
        self.scheduling_agent = create_scheduling_agent(available_hours, constraints)
    
    def run(self, tasks: List[Dict]) -> Dict:
        """
        Execute complete planning workflow:
        1. Analyze all tasks
        2. Prioritize with reasoning
        3. Schedule with reasoning
        
        Returns complete plan with reasoning trace
        """
        
        print("=" * 80)
        print("🤖 AGENTIC AI STUDY PLANNER")
        print("=" * 80)
        
        # Phase 1: Task Analysis & Prioritization
        print("\n📊 PHASE 1: PRIORITIZATION")
        print("-" * 80)
        
        prioritization_input = self._format_prioritization_input(tasks)
        prioritization_result = self.prioritization_agent.invoke({
            "input": prioritization_input
        })
        
        prioritized_tasks = self._extract_prioritized_tasks(
            prioritization_result,
            tasks
        )
        
        print("\n✅ Prioritization complete")
        print(f"Reasoning: {prioritization_result['output'][:200]}...")
        
        # Phase 2: Scheduling
        print("\n📅 PHASE 2: SCHEDULING")
        print("-" * 80)
        
        scheduling_input = self._format_scheduling_input(prioritized_tasks)
        scheduling_result = self.scheduling_agent.invoke({
            "input": scheduling_input
        })
        
        schedule = self._extract_schedule(scheduling_result)
        
        print("\n✅ Scheduling complete")
        print(f"Reasoning: {scheduling_result['output'][:200]}...")
        
        # Compile final result
        final_result = {
            "prioritized_tasks": prioritized_tasks,
            "schedule": schedule,
            "reasoning": {
                "prioritization": prioritization_result["output"],
                "scheduling": scheduling_result["output"]
            },
            "metadata": {
                "user_goal": self.user_goal,
                "available_hours": self.available_hours,
                "constraints": self.constraints,
                "total_tasks": len(tasks)
            }
        }
        
        print("\n" + "=" * 80)
        print("✨ PLANNING COMPLETE")
        print("=" * 80)
        
        return final_result
    
    def _format_prioritization_input(self, tasks: List[Dict]) -> str:
        """Format tasks for prioritization agent"""
        task_descriptions = []
        
        for idx, task in enumerate(tasks, 1):
            desc = (
                f"Task {idx}: {task['filename']}\n"
                f"  - Pages: {task.get('pages', 'unknown')}\n"
                f"  - Chunks: {task.get('chunk_count', 0)}\n"
                f"  - Summary: {task.get('summary', 'No summary available')[:200]}\n"
            )
            
            if task.get('deadline'):
                desc += f"  - Deadline: {task['deadline']}\n"
            
            task_descriptions.append(desc)
        
        input_text = (
            f"I have {len(tasks)} tasks to prioritize:\n\n"
            + "\n".join(task_descriptions) +
            f"\n\nMy goal: {self.user_goal}\n\n"
            "Please analyze each task and create a prioritized ranking with clear reasoning."
        )
        
        return input_text
    
    def _format_scheduling_input(self, prioritized_tasks: List[Dict]) -> str:
        """Format prioritized tasks for scheduling agent"""
        
        task_list = json.dumps(prioritized_tasks, indent=2)
        
        input_text = (
            f"Here are my prioritized tasks:\n\n{task_list}\n\n"
            f"I have {self.available_hours} hours available per week.\n"
        )
        
        if self.constraints:
            input_text += f"Constraints: {self.constraints}\n\n"
        
        input_text += "Please create a realistic weekly schedule with clear reasoning."
        
        return input_text
    
    def _extract_prioritized_tasks(self, result: Dict, original_tasks: List[Dict]) -> List[Dict]:
        """Extract prioritized task list from agent output"""
        
        # Try to extract structured data from agent output
        output = result.get("output", "")
        
        # Parse any JSON in the output
        try:
            # Look for JSON structure in output
            import re
            json_match = re.search(r'\[.*\]', output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return parsed
        except:
            pass
        
        # Fallback: use intermediate steps
        intermediate = result.get("intermediate_steps", [])
        
        # Build prioritized list from original tasks with priority scores
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
        """Extract schedule from agent output"""
        
        output = result.get("output", "")
        
        # Try to parse JSON schedule
        try:
            import re
            json_match = re.search(r'\{.*"schedule".*\}', output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                return parsed.get("schedule", [])
        except:
            pass
        
        # Fallback: return empty schedule
        return []


# ==================== CONVENIENCE FUNCTIONS ====================

def create_agentic_study_plan(
    tasks: List[Dict],
    user_goal: str = "finish semester with good grades",
    available_hours: int = 20,
    constraints: str = ""
) -> Dict:
    """
    Main entry point for agentic study planning.
    
    Args:
        tasks: List of task dictionaries with filename, pages, chunks, etc.
        user_goal: User's main objective
        available_hours: Hours available per week
        constraints: Additional scheduling constraints
    
    Returns:
        Complete study plan with prioritization, schedule, and reasoning
    """
    
    orchestrator = StudyPlanningOrchestrator(
        user_goal=user_goal,
        available_hours=available_hours,
        constraints=constraints
    )
    
    return orchestrator.run(tasks)


def explain_reasoning(plan_result: Dict) -> str:
    """
    Extract and format reasoning from plan result for display.
    """
    
    reasoning = plan_result.get("reasoning", {})
    
    explanation = "🧠 AI REASONING TRACE\n\n"
    
    explanation += "📊 PRIORITIZATION REASONING:\n"
    explanation += reasoning.get("prioritization", "No reasoning available")
    explanation += "\n\n"
    
    explanation += "📅 SCHEDULING REASONING:\n"
    explanation += reasoning.get("scheduling", "No reasoning available")
    explanation += "\n\n"
    
    explanation += "📈 METADATA:\n"
    metadata = plan_result.get("metadata", {})
    for key, value in metadata.items():
        explanation += f"  - {key}: {value}\n"
    
    return explanation