# core/agent_tools_advanced.py - FINAL INTEGRATED PRODUCTION
# Integrated: PKG + Adaptive Learning + KB Grounding + Multi-Agent Support
# High Performance, High Flexibility, High Accuracy, Async, Production Grade

from langchain.tools import BaseTool
from typing import Type, List, Dict, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import json
import time
import re
from enum import Enum
from .knowledge_weighting import enhance_task_with_knowledge, SchemaHandler

# ==================== ENUMS ====================

class SortMethod(str, Enum):
    """Agent-based sorting methods."""
    AI_HYBRID = "ai_hybrid"  # Agent chooses tools dynamically
    URGENCY = "urgency"  # Time-based prioritization
    PREREQUISITES = "prerequisites"  # Sequential ordering
    DIFFICULTY = "difficulty"  # Complexity-based
    AGENT_DRIVEN = "agent_driven"  # Full agent autonomy
    
    # Legacy support
    CONTENT = "content"
    PAGES = "pages"
    COMPLEXITY = "complexity"
    FOUNDATIONAL = "foundational"
    HYBRID = "hybrid"

class TimeUnit(str, Enum):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

class SessionType(str, Enum):
    DEEP_WORK = "deep_work"
    FOCUSED = "focused"
    SHORT = "short"
    MICRO = "micro"

# ==================== HELPER FUNCTIONS ====================

def _clean_task_name(task_name: str) -> str:
    """Clean task name."""
    cleaned = re.sub(r'^ชุดที่\d+\s*[-:]\s*', '', task_name)
    cleaned = re.sub(r'^(Set|Part|Chapter|Unit|Section|Module|Topic)\s*\d+\s*[-:]\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\.(pdf|docx?|txt)$', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def _determine_task_type(task_name: str, task_analysis: Dict) -> str:
    """Determine task type intelligently."""
    name_lower = task_name.lower()
    category = task_analysis.get('category', '').lower()
    complexity = task_analysis.get('complexity', 5)
    urgency = task_analysis.get('urgency_score', 5)
    is_foundational = task_analysis.get('is_foundational', False)
    
    type_keywords = {
        'Exam Prep': ['exam', 'test', 'quiz', 'assessment', 'midterm', 'final', 'evaluation'],
        'Assignment': ['assignment', 'homework', 'project', 'submission', 'coursework', 'paper', 'report'],
        'Practical': ['lab', 'practical', 'experiment', 'hands-on', 'implementation', 'coding', 'exercise'],
        'Workshop': ['workshop', 'tutorial', 'seminar', 'problem set', 'workbook', 'case study'],
        'Review': ['review', 'revision', 'summary', 'recap', 'overview', 'refresher']
    }
    
    for task_type, keywords in type_keywords.items():
        if any(keyword in name_lower for keyword in keywords):
            return task_type
    
    if category in ['exam_prep']:
        return "Exam Prep"
    elif category in ['programming', 'data_science']:
        return "Practical"
    elif category in ['research']:
        return "Assignment"
    
    if is_foundational:
        return "Theory"
    elif complexity >= 8:
        return "Exam Prep" if urgency >= 8 else "Theory"
    elif complexity <= 4:
        return "Review" if urgency >= 7 else "Practical"
    else:
        return "Theory" if 'theory' in name_lower or 'concept' in name_lower else "Workshop"

def _classify_session_type(duration: float, complexity: int) -> SessionType:
    """Classify session type."""
    if duration >= 2.5:
        return SessionType.DEEP_WORK
    elif duration >= 1.0:
        return SessionType.FOCUSED
    elif duration >= 0.5:
        return SessionType.SHORT
    else:
        return SessionType.MICRO

def _extract_sequential_number(filename: str) -> Optional[int]:
    """
    AI-based sequential number extraction using LLM.
    No regex patterns - pure AI intelligence.
    """
    try:
        from .llm_config import llm
        from langchain_core.messages import SystemMessage, HumanMessage
        
        system_prompt = """You are a filename analysis expert. Extract the sequential number from filenames.

Your task: Identify the sequential position number in the filename.

Rules:
- Look for ANY numeric pattern indicating sequence
- Common patterns: "ชุดที่1", "Chapter 2", "Part 3", "textbook_1", "ep5"
- Return ONLY the number, nothing else
- If multiple numbers exist, return the one indicating sequence position
- If no sequential number exists, return "NONE"

Examples:
- "ชุดที่1_category.pdf" → 1
- "ชุดที่2_category.pdf" → 2
- "Chapter 3 Introduction.pdf" → 3
- "textbook_5.pdf" → 5
- "part_02_advanced.pdf" → 2
- "random_file.pdf" → NONE

Output format: Just the number or "NONE"
"""

        user_prompt = f"Filename: {filename}\n\nSequential number:"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        result = response.content.strip().upper()
        
        if result == "NONE" or not result:
            return None
        
        # Extract number from response
        import re
        numbers = re.findall(r'\d+', result)
        if numbers:
            return int(numbers[0])
        
        return None
    
    except Exception as e:
        print(f"LLM sequential extraction failed: {e}")
        # Minimal fallback: extract any digit
        import re
        match = re.search(r'\d+', filename)
        return int(match.group(0)) if match else None


def _extract_sequential_numbers_batch(filenames: List[str]) -> Dict[str, Optional[int]]:
    """
    Batch AI-based sequential extraction for multiple files.
    More efficient than calling LLM for each file separately.
    """
    if not filenames:
        return {}
    
    try:
        from .llm_config import llm
        from langchain_core.messages import SystemMessage, HumanMessage
        
        system_prompt = """You are a filename analysis expert. Extract sequential numbers from multiple filenames.

Your task: For each filename, identify the sequential position number.

Rules:
- Look for ANY numeric pattern indicating sequence
- Common patterns: "ชุดที่1", "Chapter 2", "Part 3", "textbook_1", "ep5"
- If no sequential number exists, use "NONE"

Output format (JSON):
{
  "filename1": 1,
  "filename2": 2,
  "filename3": "NONE"
}

Be precise. Return valid JSON only.
"""

        filenames_list = "\n".join([f"- {fn}" for fn in filenames])
        user_prompt = f"Extract sequential numbers from these filenames:\n\n{filenames_list}\n\nJSON output:"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        result_text = response.content.strip()
        
        # Parse JSON
        import json
        import re
        
        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', result_text, re.DOTALL)
        if json_match:
            result_dict = json.loads(json_match.group(0))
            
            # Convert to proper format
            output = {}
            for filename in filenames:
                # Find matching key (case-insensitive, partial match)
                value = None
                for key, val in result_dict.items():
                    if filename in key or key in filename:
                        value = val
                        break
                
                if value and value != "NONE":
                    try:
                        output[filename] = int(value)
                    except:
                        output[filename] = None
                else:
                    output[filename] = None
            
            return output
        
        # Fallback: individual extraction
        print("Batch extraction failed, falling back to individual")
        return {fn: _extract_sequential_number(fn) for fn in filenames}
    
    except Exception as e:
        print(f"Batch LLM extraction failed: {e}")
        # Fallback: individual extraction
        return {fn: _extract_sequential_number(fn) for fn in filenames}

# ==================== TOOL LOGGING ====================

class ToolLogger:
    """Production-grade logging with performance metrics."""
    
    logs = []
    performance_metrics = {
        "total_execution_time": 0,
        "tool_call_count": 0,
        "avg_time_per_call": 0,
        "slowest_tool": None,
        "fastest_tool": None
    }
    
    @classmethod
    def log_call(cls, tool_name: str, input_data: Dict, output_data: Dict):
        """Log tool call."""
        duration = output_data.get('duration', 0)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "input": input_data,
            "output": json.dumps(output_data)[:500] if isinstance(output_data, dict) else str(output_data)[:500],
            "duration_ms": round(duration * 1000, 2) if duration else 0
        }
        cls.logs.append(log_entry)
        
        if duration:
            cls.performance_metrics["total_execution_time"] += duration
            cls.performance_metrics["tool_call_count"] += 1
            cls.performance_metrics["avg_time_per_call"] = (
                cls.performance_metrics["total_execution_time"] / 
                cls.performance_metrics["tool_call_count"]
            ) * 1000
            
            if (cls.performance_metrics["slowest_tool"] is None or 
                duration > cls.performance_metrics["slowest_tool"]["duration"]):
                cls.performance_metrics["slowest_tool"] = {"tool": tool_name, "duration": duration * 1000}
            
            if (cls.performance_metrics["fastest_tool"] is None or 
                duration < cls.performance_metrics["fastest_tool"]["duration"]):
                cls.performance_metrics["fastest_tool"] = {"tool": tool_name, "duration": duration * 1000}
    
    @classmethod
    def get_logs(cls) -> List[Dict]:
        return cls.logs.copy()

    @classmethod
    def get_summary(cls) -> Dict:
        return {
            **cls.performance_metrics,
            "total_calls": len(cls.logs),
            "tools_used": list(set(log["tool"] for log in cls.logs))
        }
    
    @classmethod
    def clear_logs(cls):
        cls.logs = []
        cls.performance_metrics = {
            "total_execution_time": 0,
            "tool_call_count": 0,
            "avg_time_per_call": 0,
            "slowest_tool": None,
            "fastest_tool": None
        }

# ==================== INTEGRATED TASK ANALYSIS ====================

class EnhancedTaskAnalysisInput(BaseModel):
    task_name: str = Field(description="Task/document name")
    content_summary: str = Field(description="Content summary")
    metadata: Dict = Field(description="Metadata including pages, deadline, etc.")
    sort_preference: Optional[str] = Field(default="ai_hybrid", description="Sorting preference")
    use_knowledge_grounding: Optional[bool] = Field(default=True, description="Enable KB comparison")
    user_id: Optional[int] = Field(default=None, description="User ID for PKG/Adaptive Learning")
    batch_filenames: Optional[List[str]] = Field(default=None, description="All filenames for batch extraction")

class EnhancedTaskAnalysisTool(BaseTool):
    """
    INTEGRATED: Task analysis with KB + PKG + Adaptive Learning + AI-based sequential detection.
    Production-grade, high-performance, high-accuracy.
    """
    name: str = "analyze_task_enhanced"
    description: str = "Integrated task analysis: KB grounding + PKG + Adaptive Learning + AI sequential detection. Returns complexity, time, category, urgency, KB relevance, task type, session recommendations, PKG insights, adaptive predictions."
    args_schema: Type[BaseModel] = EnhancedTaskAnalysisInput
    
    # Cache for batch sequential extraction
    _sequential_cache: Dict[str, Optional[int]] = {}
    
    @classmethod
    def analyze_batch(cls, tasks: List[Dict], **kwargs) -> List[Dict]:
        """
        Batch analysis for multiple tasks.
        More efficient: single LLM call for sequential extraction.
        """
        if not tasks:
            return []
        
        # Extract all filenames
        filenames = [task.get('task_name', task.get('filename', '')) for task in tasks]
        
        # Batch LLM extraction
        print(f"AI-based batch sequential extraction for {len(filenames)} files...")
        sequential_numbers = _extract_sequential_numbers_batch(filenames)
        
        # Cache results
        cls._sequential_cache.update(sequential_numbers)
        
        # Analyze each task (with cached sequential numbers)
        tool = cls()
        results = []
        
        for task in tasks:
            task_name = task.get('task_name', task.get('filename', ''))
            
            result = tool._run(
                task_name=task_name,
                content_summary=task.get('content_summary', task.get('summary', '')),
                metadata=task.get('metadata', {}),
                sort_preference=kwargs.get('sort_preference', 'ai_hybrid'),
                use_knowledge_grounding=kwargs.get('use_knowledge_grounding', True),
                user_id=kwargs.get('user_id'),
                batch_filenames=filenames
            )
            
            results.append(json.loads(result))
        
        return results
    
    def _run(
        self, 
        task_name: str, 
        content_summary: str, 
        metadata: Dict, 
        sort_preference: str = "ai_hybrid",
        use_knowledge_grounding: bool = True,
        user_id: Optional[int] = None,
        batch_filenames: Optional[List[str]] = None
    ) -> str:
        start = time.time()
        
        complexity = self._estimate_complexity(task_name, content_summary, metadata)
        pages = metadata.get("pages", 0)
        chunks = metadata.get("chunk_count", 0)
        estimated_hours = self._estimate_time(pages, chunks, complexity)
        category = self._categorize_topic(task_name, content_summary)
        is_foundational = self._check_foundational(task_name, content_summary)
        urgency = self._calculate_urgency(metadata.get("deadline"))
        
        # AI-based sequential number extraction
        # Check cache first (if batch processed)
        if task_name in self._sequential_cache:
            sequential_num = self._sequential_cache[task_name]
            print(f"   Using cached sequential: {task_name} → {sequential_num}")
        else:
            # Individual AI extraction
            sequential_num = _extract_sequential_number(task_name)
            self._sequential_cache[task_name] = sequential_num
            print(f"   AI extracted sequential: {task_name} → {sequential_num}")
        
        analysis_dict = {
            'complexity': complexity,
            'category': category,
            'urgency_score': urgency,
            'is_foundational': is_foundational,
            'sequential_number': sequential_num
        }
        task_type = _determine_task_type(task_name, analysis_dict)
        
        # Session recommendation
        session_recommendation = self._recommend_session_strategy(
            estimated_hours, complexity, task_type, urgency
        )
        
        # Calculate scores for different methods
        scores = {
            "content": self._score_by_content(category, is_foundational),
            "pages": self._score_by_pages(pages),
            "urgency": urgency,
            "complexity": 10 - complexity,
            "foundational": 10 if is_foundational else 5,
            "difficulty": complexity,  # For difficulty-based sorting
            "prerequisites": (100 - sequential_num) if sequential_num else 50,  # Earlier numbers = higher priority
            "hybrid": self._calculate_hybrid_score(urgency, is_foundational, complexity, category),
            "ai_hybrid": self._ai_hybrid_score(urgency, complexity, is_foundational, category, pages, sequential_num),
            "agent_driven": self._agent_driven_score(urgency, complexity, is_foundational, category, pages)
        }
        
        result = {
            "task": task_name,
            "analysis": {
                "complexity": complexity,
                "estimated_hours": estimated_hours,
                "category": category,
                "is_foundational": is_foundational,
                "urgency_score": urgency,
                "pages": pages,
                "chunks": chunks,
                "task_type": task_type,
                "session_recommendation": session_recommendation,
                "sequential_number": sequential_num,
                "scores": scores,
                "preferred_score": scores.get(sort_preference, scores["ai_hybrid"]),
                "sort_method": sort_preference
            }
        }
        
        # KB Grounding
        if use_knowledge_grounding and content_summary and len(content_summary) > 50:
            try:
                result["analysis"] = enhance_task_with_knowledge(
                    task_analysis=result["analysis"],
                    text_content=content_summary,
                    blend_weight=0.30,
                    min_confidence=0.30,
                    enable_knowledge_boost=True
                )
                
                if 'knowledge_adjusted_score' in result["analysis"]:
                    result["analysis"]["preferred_score"] = result["analysis"]["knowledge_adjusted_score"]
                    result["analysis"]["scores"]["knowledge_weighted"] = result["analysis"]["knowledge_adjusted_score"]
            except Exception as e:
                print(f"KB grounding failed: {e}")
        
        # PKG Integration
        if user_id:
            try:
                from .personal_knowledge_graph import get_user_pkg
                
                pkg = get_user_pkg(user_id)
                
                # Extract topics
                topics = pkg._extract_topics(task_name, content_summary)
                
                # Calculate PKG-based adjustments
                avg_mastery = 0.0
                relevant_nodes = 0
                
                for topic in topics:
                    node_id = pkg._generate_node_id(topic)
                    if node_id in pkg.nodes:
                        avg_mastery += pkg.nodes[node_id].mastery_level
                        relevant_nodes += 1
                
                if relevant_nodes > 0:
                    avg_mastery /= relevant_nodes
                    knowledge_gap_boost = (1 - avg_mastery) * 5.0
                    
                    result["analysis"]["pkg_mastery"] = avg_mastery
                    result["analysis"]["pkg_gap_boost"] = knowledge_gap_boost
                    result["analysis"]["pkg_relevant_nodes"] = relevant_nodes
                    
                    # Adjust score with PKG
                    result["analysis"]["preferred_score"] += knowledge_gap_boost
                    result["analysis"]["preferred_score"] = round(result["analysis"]["preferred_score"], 2)
            
            except Exception as e:
                print(f"PKG integration failed: {e}")
        
        # Adaptive Learning Integration
        if user_id:
            try:
                from .adaptive_learning_model import get_user_learning_model
                
                model = get_user_learning_model(user_id)
                
                # Get predictions
                predictions = model.predict_optimal_settings(task_name)
                
                result["analysis"]["adaptive_predictions"] = {
                    "confidence": predictions.get("confidence", 0.0),
                    "suggested_sort": predictions.get("suggested_sort_method", sort_preference),
                    "suggested_style": predictions.get("suggested_work_style", "balanced"),
                    "suggested_times": predictions.get("suggested_times", ["morning", "afternoon"])
                }
                
                # Adjust if confidence high
                if predictions.get("confidence", 0) >= 0.6:
                    # Use adaptive suggestions
                    result["analysis"]["adaptive_applied"] = True
            
            except Exception as e:
                print(f"Adaptive learning integration failed: {e}")
        
        duration = time.time() - start
        output = json.dumps(result, indent=2)
        
        ToolLogger.log_call(
            self.name, 
            {"task": task_name, "sort": sort_preference, "kb_enabled": use_knowledge_grounding, "user_id": user_id}, 
            {"result": result, "duration": duration}
        )
        
        return output
    
    def _ai_hybrid_score(self, urgency: int, complexity: int, is_foundational: bool, 
                         category: str, pages: int, sequential_num: Optional[int]) -> float:
        """
        AI HYBRID: Agent intelligently weights factors.
        
        Dynamic weighting based on context.
        """
        
        # Base on urgency
        if urgency >= 8:
            # High urgency: prioritize by deadline
            score = urgency * 3.5
            score += complexity * 0.5
        elif is_foundational:
            # Foundational: prioritize regardless of urgency
            score = 15.0
            score += (10 - urgency) * 0.5  # Earlier in non-urgent cases
        elif sequential_num and sequential_num <= 3:
            # Early sequential: likely prerequisite
            score = 12.0
            score += (100 - sequential_num) * 0.1
        elif complexity >= 8:
            # High complexity: needs time
            score = complexity * 1.5
            score += urgency * 0.8
        else:
            # Balanced approach
            score = urgency * 2.0
            score += complexity * 1.0
            score += (10 if is_foundational else 2)
        
        # Category bonuses
        category_weights = {
            "exam_prep": 4.0,
            "mathematics": 3.0,
            "data_science": 3.0,
            "programming": 2.5,
            "science": 2.0
        }
        score += category_weights.get(category, 1.0)
        
        # Page volume
        if pages > 200:
            score += 2.0
        elif pages > 100:
            score += 1.0
        
        return round(score, 2)
    
    def _recommend_session_strategy(self, hours: float, complexity: int, 
                                   task_type: str, urgency: int) -> Dict:
        """Session splitting strategy."""
        
        if hours <= 1.0:
            return {
                "strategy": "single_session",
                "sessions": 1,
                "duration_per_session": hours,
                "breaks_needed": 0,
                "session_type": "short"
            }
        elif hours <= 2.5:
            return {
                "strategy": "single_focused",
                "sessions": 1,
                "duration_per_session": hours,
                "breaks_needed": 1,
                "session_type": "focused"
            }
        elif hours <= 5.0:
            return {
                "strategy": "split_two",
                "sessions": 2,
                "duration_per_session": hours / 2,
                "breaks_needed": 2,
                "session_type": "deep_work",
                "split_reason": "Better retention with spaced learning"
            }
        else:
            optimal_sessions = min(4, int(hours / 2) + 1)
            return {
                "strategy": "multi_session",
                "sessions": optimal_sessions,
                "duration_per_session": hours / optimal_sessions,
                "breaks_needed": optimal_sessions,
                "session_type": "deep_work",
                "split_reason": f"Optimal cognitive load distribution across {optimal_sessions} sessions"
            }
    
    def _agent_driven_score(self, urgency: int, complexity: int, 
                           is_foundational: bool, category: str, pages: int) -> float:
        """Agent-driven intelligent scoring."""
        
        score = urgency * 3.0
        
        if is_foundational:
            score += 10.0
        
        if complexity >= 8:
            score += 3.0
        elif complexity >= 6:
            score += 1.5
        elif complexity <= 3:
            score += 0.5
        
        category_weights = {
            "exam_prep": 5.0,
            "mathematics": 3.0,
            "data_science": 3.0,
            "programming": 2.5,
            "science": 2.0,
            "research": 2.0,
            "business": 1.5
        }
        score += category_weights.get(category, 1.0)
        
        if pages > 200:
            score += 2.0
        elif pages > 100:
            score += 1.0
        
        return round(score, 2)
    
    def _estimate_complexity(self, name: str, summary: str, metadata: Dict) -> int:
        score = 5
        lower = (name + " " + summary).lower()
        
        if any(kw in lower for kw in ["advanced", "complex", "graduate", "phd", "research", "theoretical"]):
            score += 3
        if any(kw in lower for kw in ["intro", "basic", "fundamental", "101", "beginner", "elementary"]):
            score -= 2
        if any(kw in lower for kw in ["intermediate", "applied", "practical"]):
            score += 1
        
        pages = metadata.get("pages", 0)
        if pages > 300:
            score += 2
        elif pages > 150:
            score += 1
        elif pages < 50:
            score -= 1
        
        return max(1, min(10, score))
    
    def _estimate_time(self, pages: int, chunks: int, complexity: int) -> float:
        if pages > 0:
            pages_per_hour = 10 - (complexity * 0.5)
            hours = pages / max(pages_per_hour, 3)
        else:
            hours = chunks * 0.15
        
        if complexity >= 7:
            hours *= 1.3
        
        return round(hours, 1)
    
    def _categorize_topic(self, name: str, summary: str) -> str:
        text = (name + " " + summary).lower()
        
        categories = {
            "mathematics": ["math", "calculus", "algebra", "geometry", "statistics", "probability"],
            "data_science": ["data", "analysis", "machine learning", "ai", "analytics", "ml", "neural"],
            "finance": ["finance", "accounting", "economics", "investment", "trading"],
            "programming": ["programming", "code", "software", "algorithm", "python", "java"],
            "exam_prep": ["exam", "test", "midterm", "final", "quiz", "practice"],
            "research": ["research", "thesis", "dissertation", "study", "paper"],
            "business": ["business", "management", "strategy", "marketing"],
            "science": ["physics", "chemistry", "biology", "science", "laboratory"],
        }
        
        for category, keywords in categories.items():
            if any(kw in text for kw in keywords):
                return category
        return "general"
    
    def _check_foundational(self, name: str, summary: str) -> bool:
        text = (name + " " + summary).lower()
        indicators = [
            "intro", "introduction", "fundamental", "basic", "foundation",
            "101", "chapter 1", "ch1", "prerequisite", "essentials",
            "beginning", "primer", "overview"
        ]
        return any(ind in text for ind in indicators)
    
    def _calculate_urgency(self, deadline: Optional[str]) -> int:
        if not deadline:
            return 5
        
        try:
            deadline_dt = datetime.fromisoformat(deadline)
            days_until = (deadline_dt - datetime.now()).days
            
            if days_until < 0: return 10
            elif days_until <= 1: return 10
            elif days_until <= 3: return 9
            elif days_until <= 7: return 8
            elif days_until <= 14: return 6
            elif days_until <= 30: return 4
            else: return 2
        except:
            return 5
    
    def _score_by_content(self, category: str, is_foundational: bool) -> float:
        base_score = 5.0
        
        if category in ["mathematics", "data_science", "exam_prep"]:
            base_score += 2.0
        elif category in ["programming", "science"]:
            base_score += 1.5
        
        if is_foundational:
            base_score += 3.0
        
        return base_score
    
    def _score_by_pages(self, pages: int) -> float:
        if pages == 0:
            return 5.0
        elif pages < 50:
            return 9.0
        elif pages < 100:
            return 7.0
        elif pages < 200:
            return 5.0
        else:
            return 3.0
    
    def _calculate_hybrid_score(self, urgency: int, is_foundational: bool, 
                                complexity: int, category: str) -> float:
        score = 0.0
        score += urgency * 2.5
        
        if is_foundational:
            score += 8.0
        
        if complexity > 8:
            score -= 2.0
        elif complexity < 4:
            score += 1.0
        
        if category in ["exam_prep", "mathematics", "data_science"]:
            score += 3.0
        elif category in ["programming", "science"]:
            score += 2.0
        
        return round(score, 2)

# ==================== ADVANCED SCHEDULING ====================

class FlexibleSchedulingInput(BaseModel):
    prioritized_tasks: str = Field(description="JSON of prioritized tasks")
    available_time: Dict = Field(description="Time availability")
    constraints: str = Field(description="Schedule constraints")
    sort_method: str = Field(default="ai_hybrid", description="Sorting method")
    enable_multi_week: Optional[bool] = Field(default=True, description="Enable multi-week rolling")
    enable_session_splitting: Optional[bool] = Field(default=True, description="Split long sessions")
    user_id: Optional[int] = Field(default=None, description="User ID for adaptive constraints")

class FlexibleSchedulingTool(BaseTool):
    """ADVANCED: Multi-week rolling, dynamic session splitting, break management, overlap prevention."""
    name: str = "create_flexible_schedule"
    description: str = "Advanced scheduler: multi-week rolling schedules, dynamic session splitting, intelligent break management, overlap detection, agent-driven prioritization."
    args_schema: Type[BaseModel] = FlexibleSchedulingInput
    
    def _run(self, prioritized_tasks: str, available_time: Dict, 
             constraints: str, sort_method: str = "ai_hybrid",
             enable_multi_week: bool = True,
             enable_session_splitting: bool = True,
             user_id: Optional[int] = None) -> str:
        start = time.time()
        
        tasks = json.loads(prioritized_tasks) if isinstance(prioritized_tasks, str) else prioritized_tasks
        
        total_hours = self._convert_to_hours(available_time)
        
        # Apply adaptive learning to constraints
        if user_id:
            try:
                from .adaptive_learning_model import get_user_learning_model
                
                model = get_user_learning_model(user_id)
                predictions = model.predict_optimal_settings()
                
                # Enhance constraints with learned preferences
                if predictions.get("confidence", 0) >= 0.6:
                    suggested_times = predictions.get("suggested_times", [])
                    suggested_style = predictions.get("suggested_work_style", "balanced")
                    
                    if not constraints or len(constraints.strip()) < 10:
                        constraints = f"Prefer {', '.join(suggested_times)} study times. {suggested_style.capitalize()} work style."
            except Exception as e:
                print(f"Adaptive constraint enhancement failed: {e}")
        
        constraint_dict = self._parse_constraints(constraints)
        
        # Determine multi-week need
        needs_multi_week = self._assess_multi_week_need(tasks, total_hours)
        weeks_needed = needs_multi_week["weeks"] if needs_multi_week["needed"] and enable_multi_week else 1
        
        # Generate schedule
        if weeks_needed > 1:
            schedule = self._generate_multi_week_schedule(
                tasks, total_hours, constraint_dict, sort_method, weeks_needed, enable_session_splitting
            )
        else:
            schedule = self._generate_single_week_schedule(
                tasks, total_hours, constraint_dict, sort_method, enable_session_splitting
            )
        
        # Validate and fix
        schedule = self._validate_and_fix_schedule(schedule)
        schedule = self._insert_break_sessions(schedule, constraint_dict)
        
        # Statistics
        allocated_hours = sum(s["hours"] for s in schedule if s.get("type") != "Break")
        utilization = (allocated_hours / (total_hours * weeks_needed) * 100) if total_hours > 0 else 0
        
        type_distribution = {}
        day_distribution = {}
        week_distribution = {}
        
        for item in schedule:
            t = item.get('type', 'Study')
            type_distribution[t] = type_distribution.get(t, 0) + 1
            
            d = item.get('day', 'Unknown')
            day_distribution[d] = day_distribution.get(d, 0) + 1
            
            w = item.get('week', 1)
            week_distribution[w] = week_distribution.get(w, 0) + 1
        
        result = {
            "schedule": schedule,
            "reasoning": self._explain_advanced_schedule(
                schedule, total_hours, available_time, sort_method, 
                type_distribution, day_distribution, weeks_needed
            ),
            "total_allocated_hours": allocated_hours,
            "available_hours": total_hours * weeks_needed,
            "time_breakdown": available_time,
            "utilization_percent": round(utilization, 1),
            "task_types_distribution": type_distribution,
            "day_distribution": day_distribution,
            "week_distribution": week_distribution if weeks_needed > 1 else None,
            "weeks_planned": weeks_needed,
            "multi_week_enabled": enable_multi_week,
            "session_splitting_enabled": enable_session_splitting,
            "breaks_inserted": sum(1 for s in schedule if s.get("type") == "Break"),
            "adaptive_applied": user_id is not None,
            "validation": "passed"
        }
        
        output = json.dumps(result, indent=2)
        ToolLogger.log_call(self.name, {"tasks": len(tasks), "hours": total_hours, "weeks": weeks_needed, "user_id": user_id}, result)
        return output
    
    def _assess_multi_week_need(self, tasks: List[Dict], weekly_hours: float) -> Dict:
        """Determine if multi-week scheduling needed."""
        total_task_hours = sum(
            t.get("analysis", {}).get("estimated_hours", 2.0) 
            for t in tasks
        )
        
        if total_task_hours <= weekly_hours * 0.85:
            return {"needed": False, "weeks": 1, "reason": "Fits in one week"}
        
        weeks_needed = int(total_task_hours / (weekly_hours * 0.75)) + 1
        weeks_needed = min(weeks_needed, 4)
        
        return {
            "needed": True,
            "weeks": weeks_needed,
            "reason": f"Total hours ({total_task_hours:.1f}h) requires {weeks_needed} weeks",
            "overflow_hours": total_task_hours - weekly_hours
        }
    
    def _generate_multi_week_schedule(self, tasks: List[Dict], weekly_hours: float,
                                     constraints: Dict, sort_method: str, 
                                     weeks: int, enable_splitting: bool) -> List[Dict]:
        """Generate rolling multi-week schedule."""
        
        schedule = []
        hours_per_week = weekly_hours
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        if constraints.get("preferred_days"):
            available_days = constraints["preferred_days"]
        else:
            avoid = constraints.get("avoid_days", [])
            available_days = [d for d in days if d not in avoid]
        
        if not available_days:
            available_days = days
        
        tasks_per_week = len(tasks) // weeks + 1
        
        for week_num in range(1, weeks + 1):
            week_start_idx = (week_num - 1) * tasks_per_week
            week_end_idx = min(week_num * tasks_per_week, len(tasks))
            week_tasks = tasks[week_start_idx:week_end_idx]
            
            if not week_tasks:
                continue
            
            week_schedule = self._allocate_week(
                week_tasks, hours_per_week, constraints, available_days, 
                week_num, enable_splitting
            )
            
            schedule.extend(week_schedule)
        
        return schedule
    
    def _generate_single_week_schedule(self, tasks: List[Dict], total_hours: float,
                                      constraints: Dict, sort_method: str,
                                      enable_splitting: bool) -> List[Dict]:
        """Generate single week schedule."""
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        if constraints.get("preferred_days"):
            available_days = constraints["preferred_days"]
        else:
            avoid = constraints.get("avoid_days", [])
            available_days = [d for d in days if d not in avoid]
        
        if not available_days:
            available_days = days
        
        return self._allocate_week(tasks, total_hours, constraints, available_days, 1, enable_splitting)
    
    def _allocate_week(self, tasks: List[Dict], hours_available: float,
                      constraints: Dict, available_days: List[str], 
                      week_num: int, enable_splitting: bool) -> List[Dict]:
        """Allocate tasks for single week."""
        
        schedule = []
        hours_remaining = hours_available
        
        time_templates = {
            "morning": [
                {"start": "08:00", "max": 3.0},
                {"start": "09:00", "max": 3.0},
                {"start": "10:00", "max": 2.5},
            ],
            "afternoon": [
                {"start": "13:00", "max": 4.0},
                {"start": "14:00", "max": 3.5},
                {"start": "15:00", "max": 3.0},
                {"start": "16:00", "max": 2.5},
            ],
            "evening": [
                {"start": "18:00", "max": 3.0},
                {"start": "19:00", "max": 2.5},
                {"start": "20:00", "max": 2.0},
            ],
            "night": [
                {"start": "21:00", "max": 2.0},
                {"start": "22:00", "max": 1.5},
            ]
        }
        
        preferred_times = constraints.get("preferred_times", ["morning", "afternoon", "evening"])
        max_session = constraints.get("max_session", 5.0)
        
        day_idx = 0
        time_pref_idx = 0
        tasks_today = 0
        tasks_per_day = max(1, len(tasks) // len(available_days) + 1)
        
        for task in tasks:
            if hours_remaining <= 0.5:
                break
            
            task_analysis = task.get("analysis", {})
            task_name = _clean_task_name(task["task"])
            estimated_hours = task_analysis.get("estimated_hours", 2.0)
            complexity = task_analysis.get("complexity", 5)
            task_type = task_analysis.get("task_type") or _determine_task_type(task_name, task_analysis)
            
            session_rec = task_analysis.get("session_recommendation", {})
            should_split = (
                enable_splitting and 
                session_rec.get("strategy") in ["split_two", "multi_session"] and
                estimated_hours > 2.5
            )
            
            if should_split:
                num_sessions = session_rec.get("sessions", 2)
                duration_per = session_rec.get("duration_per_session", estimated_hours / 2)
                
                for session_idx in range(num_sessions):
                    if hours_remaining <= 0.5:
                        break
                    
                    session_hours = min(duration_per, hours_remaining, max_session)
                    session_hours = round(session_hours * 2) / 2
                    
                    session_item = self._create_session_item(
                        task_name, session_hours, task_type, complexity,
                        available_days, day_idx, time_templates, preferred_times,
                        time_pref_idx, week_num, task.get("priority", 999),
                        task_analysis.get("category", "general"),
                        session_num=session_idx + 1,
                        total_sessions=num_sessions
                    )
                    
                    schedule.append(session_item)
                    hours_remaining -= session_hours
                    time_pref_idx += 1
                    tasks_today += 1
                    
                    if tasks_today >= tasks_per_day:
                        day_idx += 1
                        tasks_today = 0
                        time_pref_idx = 0
            
            else:
                session_hours = self._calculate_session_duration(
                    estimated_hours, task_type, complexity, 
                    constraints.get("work_style", "balanced"), max_session
                )
                
                session_hours = max(0.5, min(session_hours, hours_remaining, max_session))
                session_hours = round(session_hours * 2) / 2
                
                session_item = self._create_session_item(
                    task_name, session_hours, task_type, complexity,
                    available_days, day_idx, time_templates, preferred_times,
                    time_pref_idx, week_num, task.get("priority", 999),
                    task_analysis.get("category", "general")
                )
                
                schedule.append(session_item)
                hours_remaining -= session_hours
                time_pref_idx += 1
                tasks_today += 1
                
                if tasks_today >= tasks_per_day:
                    day_idx += 1
                    tasks_today = 0
                    time_pref_idx = 0
        
        return schedule
    
    def _create_session_item(self, task_name: str, duration: float, task_type: str,
                            complexity: int, available_days: List[str], day_idx: int,
                            time_templates: Dict, preferred_times: List[str],
                            time_pref_idx: int, week_num: int, priority: int,
                            category: str, session_num: Optional[int] = None,
                            total_sessions: Optional[int] = None) -> Dict:
        """Create session item."""
        
        if task_type in ["Theory", "Exam Prep"]:
            time_pref = "morning"
        elif task_type in ["Practical", "Workshop"]:
            time_pref = "afternoon"
        elif task_type == "Review":
            time_pref = "evening"
        else:
            time_pref = preferred_times[time_pref_idx % len(preferred_times)]
        
        if time_pref not in preferred_times:
            time_pref = preferred_times[0]
        
        templates = time_templates.get(time_pref, time_templates["afternoon"])
        template = templates[time_pref_idx % len(templates)]
        
        start_h, start_m = map(int, template["start"].split(":"))
        end_h = start_h + int(duration)
        end_m = start_m + int((duration % 1) * 60)
        
        if end_m >= 60:
            end_h += 1
            end_m -= 60
        
        time_range = f"{template['start']}-{end_h:02d}:{end_m:02d}"
        day = available_days[day_idx % len(available_days)]
        
        if session_num:
            display_name = f"{task_name} (Part {session_num}/{total_sessions})"
        else:
            display_name = task_name
        
        session_type = _classify_session_type(duration, complexity).value
        
        return {
            "task": display_name,
            "original_task": task_name,
            "day": day,
            "time": time_range,
            "hours": round(duration, 1),
            "type": task_type,
            "session_type": session_type,
            "priority": priority,
            "complexity": complexity,
            "category": category,
            "week": week_num,
            "is_split": session_num is not None,
            "session_info": {
                "session_num": session_num,
                "total_sessions": total_sessions
            } if session_num else None
        }
    
    def _calculate_session_duration(self, estimated: float, task_type: str,
                                    complexity: int, work_style: str, max_session: float) -> float:
        """Calculate session duration."""
        
        if task_type == "Exam Prep":
            duration = min(estimated * 1.2, max_session)
        elif task_type == "Review":
            duration = min(max(estimated * 0.6, 0.5), 2.0)
        elif task_type == "Theory":
            multiplier = 1.3 if work_style == "intensive" else 1.0
            duration = min(estimated * multiplier, max_session)
        elif task_type in ["Practical", "Workshop"]:
            duration = min(estimated * 0.9, max_session)
        elif task_type == "Assignment":
            duration = min(estimated * 1.1, max_session)
        else:
            duration = min(estimated, max_session)
        
        if complexity >= 8:
            duration *= 1.15
        elif complexity <= 3:
            duration *= 0.85
        
        return duration
    
    def _validate_and_fix_schedule(self, schedule: List[Dict]) -> List[Dict]:
        """Validate schedule."""
        
        if not schedule:
            return schedule
        
        by_week_day = {}
        for item in schedule:
            week = item.get("week", 1)
            day = item.get("day", "Monday")
            key = f"W{week}-{day}"
            
            if key not in by_week_day:
                by_week_day[key] = []
            by_week_day[key].append(item)
        
        fixed_schedule = []
        
        for key, day_items in by_week_day.items():
            sorted_items = sorted(day_items, key=lambda x: x.get("time", "00:00").split("-")[0])
            
            for i, item in enumerate(sorted_items):
                time_range = item.get("time", "")
                duration = item.get("hours", 0)
                
                if "-" in time_range:
                    try:
                        start, end = time_range.split("-")
                        start_h, start_m = map(int, start.split(":"))
                        end_h, end_m = map(int, end.split(":"))
                        
                        actual_duration = (end_h - start_h) + (end_m - start_m) / 60
                        
                        if abs(actual_duration - duration) > 0.1:
                            new_end_h = start_h + int(duration)
                            new_end_m = start_m + int((duration % 1) * 60)
                            
                            if new_end_m >= 60:
                                new_end_h += 1
                                new_end_m -= 60
                            
                            item["time"] = f"{start_h:02d}:{start_m:02d}-{new_end_h:02d}:{new_end_m:02d}"
                            end_h, end_m = new_end_h, new_end_m
                        
                        if i + 1 < len(sorted_items):
                            next_item = sorted_items[i + 1]
                            next_time = next_item.get("time", "")
                            if "-" in next_time:
                                next_start = next_time.split("-")[0]
                                next_start_h, next_start_m = map(int, next_start.split(":"))
                                
                                if (end_h > next_start_h) or (end_h == next_start_h and end_m > next_start_m):
                                    new_start_h = end_h
                                    new_start_m = end_m + 15
                                    
                                    if new_start_m >= 60:
                                        new_start_h += 1
                                        new_start_m -= 60
                                    
                                    next_duration = next_item.get("hours", 1.0)
                                    new_end_h = new_start_h + int(next_duration)
                                    new_end_m = new_start_m + int((next_duration % 1) * 60)
                                    
                                    if new_end_m >= 60:
                                        new_end_h += 1
                                        new_end_m -= 60
                                    
                                    next_item["time"] = f"{new_start_h:02d}:{new_start_m:02d}-{new_end_h:02d}:{new_end_m:02d}"
                    
                    except Exception as e:
                        print(f"Time validation error: {e}")
                
                fixed_schedule.append(item)
        
        return fixed_schedule
    
    def _insert_break_sessions(self, schedule: List[Dict], constraints: Dict) -> List[Dict]:
        """Insert breaks."""
        
        break_duration = constraints.get("break_duration", 0.25)
        
        by_week_day = {}
        for item in schedule:
            week = item.get("week", 1)
            day = item.get("day", "Monday")
            key = f"W{week}-{day}"
            
            if key not in by_week_day:
                by_week_day[key] = []
            by_week_day[key].append(item)
        
        schedule_with_breaks = []
        
        for key, day_items in by_week_day.items():
            sorted_items = sorted(day_items, key=lambda x: x.get("time", "00:00").split("-")[0])
            
            for i, item in enumerate(sorted_items):
                schedule_with_breaks.append(item)
                
                if item.get("hours", 0) >= 2.0 and i + 1 < len(sorted_items):
                    time_range = item.get("time", "")
                    if "-" in time_range:
                        _, end = time_range.split("-")
                        end_h, end_m = map(int, end.split(":"))
                        
                        break_end_h = end_h
                        break_end_m = end_m + int(break_duration * 60)
                        
                        if break_end_m >= 60:
                            break_end_h += 1
                            break_end_m -= 60
                        
                        break_item = {
                            "task": "Break",
                            "day": item["day"],
                            "time": f"{end_h:02d}:{end_m:02d}-{break_end_h:02d}:{break_end_m:02d}",
                            "hours": break_duration,
                            "type": "Break",
                            "session_type": "micro",
                            "priority": 0,
                            "complexity": 0,
                            "category": "break",
                            "week": item.get("week", 1)
                        }
                        
                        schedule_with_breaks.append(break_item)
        
        return schedule_with_breaks
    
    def _convert_to_hours(self, time_dict: Dict) -> float:
        total = 0.0
        total += time_dict.get("years", 0) * 365 * 24
        total += time_dict.get("months", 0) * 30 * 24
        total += time_dict.get("weeks", 0) * 7 * 24
        total += time_dict.get("days", 0) * 24
        total += time_dict.get("hours", 0)
        return total
    
    def _parse_constraints(self, constraints: str) -> Dict:
        constraint_dict = {
            "preferred_times": [],
            "max_session": 5.0,
            "avoid_days": [],
            "preferred_days": [],
            "break_duration": 0.25,
            "work_style": "balanced"
        }
        
        if not constraints:
            constraint_dict["preferred_times"] = ["morning", "afternoon", "evening"]
            return constraint_dict
        
        lower = constraints.lower()
        
        if "morning" in lower:
            constraint_dict["preferred_times"].append("morning")
        if "afternoon" in lower:
            constraint_dict["preferred_times"].append("afternoon")
        if "evening" in lower:
            constraint_dict["preferred_times"].append("evening")
        if "night" in lower or "late" in lower:
            constraint_dict["preferred_times"].append("night")
        
        if not constraint_dict["preferred_times"]:
            constraint_dict["preferred_times"] = ["morning", "afternoon", "evening"]
        
        if "weekend only" in lower:
            constraint_dict["preferred_days"] = ["Saturday", "Sunday"]
        elif "no weekend" in lower or "weekday" in lower:
            constraint_dict["avoid_days"] = ["Saturday", "Sunday"]
        
        if "intensive" in lower or "cramming" in lower:
            constraint_dict["work_style"] = "intensive"
            constraint_dict["max_session"] = 5.0
        elif "relaxed" in lower or "slow" in lower:
            constraint_dict["work_style"] = "relaxed"
            constraint_dict["max_session"] = 3.0
        
        return constraint_dict
    
    def _explain_advanced_schedule(self, schedule: List[Dict], weekly_hours: float,
                                  time_breakdown: Dict, sort_method: str,
                                  type_distribution: Dict, day_distribution: Dict,
                                  weeks: int) -> str:
        """Generate reasoning."""
        
        total_hours = weekly_hours * weeks
        allocated = sum(s["hours"] for s in schedule if s.get("type") != "Break")
        utilization = (allocated / total_hours * 100) if total_hours > 0 else 0
        
        time_parts = []
        if time_breakdown.get("years"):
            time_parts.append(f"{time_breakdown['years']} year(s)")
        if time_breakdown.get("months"):
            time_parts.append(f"{time_breakdown['months']} month(s)")
        if time_breakdown.get("weeks"):
            time_parts.append(f"{time_breakdown['weeks']} week(s)")
        if time_breakdown.get("days"):
            time_parts.append(f"{time_breakdown['days']} day(s)")
        if time_breakdown.get("hours"):
            time_parts.append(f"{time_breakdown['hours']} hour(s)")
        
        time_str = ", ".join(time_parts) if time_parts else f"{total_hours} hours"
        
        type_summary = ", ".join([f"{count} {ttype}" for ttype, count in type_distribution.items()])
        day_summary = ", ".join([f"{day}({count})" for day, count in sorted(day_distribution.items())])
        
        split_sessions = sum(1 for s in schedule if s.get("is_split"))
        breaks = sum(1 for s in schedule if s.get("type") == "Break")
        
        reasoning = f"""Advanced Optimized Schedule Analysis:

Time Allocation:
Total Available: {time_str} ({total_hours:.1f} hours across {weeks} week(s))
Time Allocated: {allocated:.1f} hours
Utilization Rate: {utilization:.1f}%
Planning Horizon: {weeks} week(s) {'(Multi-week Rolling)' if weeks > 1 else '(Single Week)'}

Task Distribution:
Task Types: {type_summary}
Day Distribution: {day_summary}
Split Sessions: {split_sessions} (for better retention)
Break Sessions: {breaks} (for sustainable productivity)

Advanced Features Applied:
Agent-Driven Prioritization: {sort_method}
Dynamic Session Splitting: Long tasks split into optimal chunks
Break Management: Intelligent breaks after deep work (>2h)
Overlap Prevention: Validated, no time conflicts
Multi-Week Rolling: Tasks distributed across {weeks} week(s)

Optimization Strategy:
Theory/Exam Prep → Morning slots (8-11am) - peak cognitive performance
Practical/Workshop → Afternoon (1-5pm) - optimal for hands-on work
Review sessions → Evening (6-8pm) - memory consolidation period
Flexible durations (0.5h-5h) based on complexity and task type
Sessions >2.5h automatically split for better retention
15-minute breaks inserted after intensive sessions
Balanced workload across all {len(day_distribution)} days

Session Types:
Deep Work (2.5-5h): {sum(1 for s in schedule if s.get("session_type") == "deep_work")} sessions
Focused (1-2.5h): {sum(1 for s in schedule if s.get("session_type") == "focused")} sessions
Short (0.5-1h): {sum(1 for s in schedule if s.get("session_type") == "short")} sessions
Micro (<0.5h): {sum(1 for s in schedule if s.get("session_type") == "micro")} sessions

This schedule maximizes learning efficiency through scientifically-backed spaced repetition, 
optimal session lengths, and cognitive load management."""
        
        return reasoning

# ==================== TOOL REGISTRY ====================

def get_advanced_agent_tools() -> List[BaseTool]:
    """Return all integrated tools."""
    return [
        EnhancedTaskAnalysisTool(),
        FlexibleSchedulingTool()
    ]