# core/agent_tools_advanced.py - Enhanced with Task Type Detection

from langchain.tools import BaseTool
from typing import Type, List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import json
import time
import re
from enum import Enum
from .knowledge_weighting import enhance_task_with_knowledge, SchemaHandler

# ==================== ENUMS FOR FLEXIBILITY ====================

class SortMethod(str, Enum):
    """Flexible sorting methods"""
    CONTENT = "content"  # By topic/content importance
    PAGES = "pages"  # By number of pages
    URGENCY = "urgency"  # By deadline
    COMPLEXITY = "complexity"  # By difficulty
    FOUNDATIONAL = "foundational"  # Prerequisites first
    HYBRID = "hybrid"  # AI-driven hybrid approach

class TimeUnit(str, Enum):
    """Time unit flexibility"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"

# ==================== HELPER FUNCTIONS ====================

def _clean_task_name(task_name: str) -> str:
    """
    Remove prefixes and clean task names for display.
    """
    # Remove Thai prefixes like "ชุดที่X"
    cleaned = re.sub(r'^ชุดที่\d+\s*[-:]\s*', '', task_name)
    # Remove English prefixes
    cleaned = re.sub(r'^(Set|Part|Chapter|Unit|Section|Module|Topic)\s*\d+\s*[-:]\s*', '', cleaned, flags=re.IGNORECASE)
    # Remove file extensions if present
    cleaned = re.sub(r'\.(pdf|docx?|txt)$', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def _determine_task_type(task_name: str, task_analysis: Dict) -> str:
    """
    Intelligently determine the type of study activity based on task characteristics.
    Returns: Theory, Practical, Exam Prep, Assignment, Review, or Workshop
    """
    name_lower = task_name.lower()
    category = task_analysis.get('category', '').lower()
    complexity = task_analysis.get('complexity', 5)
    urgency = task_analysis.get('urgency_score', 5)
    is_foundational = task_analysis.get('is_foundational', False)
    
    # Priority-based type detection with expanded keywords
    type_keywords = {
        'Exam Prep': ['exam', 'test', 'quiz', 'assessment', 'midterm', 'final', 'evaluation', 'examination'],
        'Assignment': ['assignment', 'homework', 'project', 'submission', 'coursework', 'paper', 'report', 'essay'],
        'Practical': ['lab', 'practical', 'experiment', 'hands-on', 'implementation', 'coding', 'exercise', 'practice'],
        'Workshop': ['workshop', 'tutorial', 'seminar', 'problem set', 'workbook', 'case study'],
        'Review': ['review', 'revision', 'summary', 'recap', 'overview', 'refresher', 'notes']
    }
    
    # Check for specific keywords
    for task_type, keywords in type_keywords.items():
        if any(keyword in name_lower for keyword in keywords):
            return task_type
    
    # Category-based classification
    if category in ['exam_prep']:
        return "Exam Prep"
    elif category in ['programming', 'data_science']:
        return "Practical"
    elif category in ['research']:
        return "Assignment"
    
    # Intelligent fallback based on analysis
    if is_foundational:
        return "Theory"  # Foundational materials are usually theoretical
    elif complexity >= 8:
        if urgency >= 8:
            return "Exam Prep"  # High complexity + high urgency = likely exam
        else:
            return "Theory"  # High complexity = theoretical material
    elif complexity <= 4:
        if urgency >= 7:
            return "Review"  # Low complexity + high urgency = review material
        else:
            return "Practical"  # Low complexity = hands-on work
    else:
        # Medium complexity - check other factors
        if 'application' in category or 'case' in name_lower:
            return "Practical"
        elif 'foundation' in category or 'principle' in name_lower:
            return "Theory"
        else:
            return "Workshop"  # Default for medium complexity

# ==================== TOOL LOGGING ====================

class ToolLogger:
    """Enhanced logging with performance metrics"""
    
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
        """Log a tool call with performance tracking"""
        duration = output_data.get('duration', 0)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "input": input_data,
            "output": json.dumps(output_data)[:500] if isinstance(output_data, dict) else str(output_data)[:500],
            "duration_ms": round(duration * 1000, 2) if duration else 0
        }
        cls.logs.append(log_entry)
        
        # Update performance metrics
        if duration:
            cls.performance_metrics["total_execution_time"] += duration
            cls.performance_metrics["tool_call_count"] += 1
            cls.performance_metrics["avg_time_per_call"] = (
                cls.performance_metrics["total_execution_time"] / 
                cls.performance_metrics["tool_call_count"]
            ) * 1000
            
            # Track slowest/fastest
            if (cls.performance_metrics["slowest_tool"] is None or 
                duration > cls.performance_metrics["slowest_tool"]["duration"]):
                cls.performance_metrics["slowest_tool"] = {
                    "tool": tool_name,
                    "duration": duration * 1000
                }
            
            if (cls.performance_metrics["fastest_tool"] is None or 
                duration < cls.performance_metrics["fastest_tool"]["duration"]):
                cls.performance_metrics["fastest_tool"] = {
                    "tool": tool_name,
                    "duration": duration * 1000
                }
    
    @classmethod
    def get_logs(cls) -> List[Dict]:
        """Return all log entries"""
        return cls.logs.copy()

    @classmethod
    def get_summary(cls) -> Dict:
        """Get detailed performance summary"""
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

# ==================== ENHANCED TASK ANALYSIS ====================

class EnhancedTaskAnalysisInput(BaseModel):
    """Enhanced input with sorting preferences"""
    task_name: str = Field(description="Task/document name")
    content_summary: str = Field(description="Content summary")
    metadata: Dict = Field(description="Metadata including pages, deadline, etc.")
    sort_preference: Optional[str] = Field(default="hybrid", description="User's sorting preference")
    use_knowledge_grounding: Optional[bool] = Field(default=True, description="Enable KB comparison")

# core/agent_tools_advanced.py - UPDATED with Knowledge Grounding Integration

from langchain.tools import BaseTool
from typing import Type, List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import json
import time
import re
from enum import Enum

# Import knowledge grounding
from .knowledge_weighting import enhance_task_with_knowledge, SchemaHandler

# ... [Keep all existing code from your agent_tools_advanced.py] ...

class EnhancedTaskAnalysisInput(BaseModel):
    """Enhanced input with sorting preferences"""
    task_name: str = Field(description="Task/document name")
    content_summary: str = Field(description="Content summary")
    metadata: Dict = Field(description="Metadata including pages, deadline, etc.")
    sort_preference: Optional[str] = Field(default="hybrid", description="User's sorting preference")
    # NEW: Flag to enable knowledge grounding
    use_knowledge_grounding: Optional[bool] = Field(default=True, description="Enable KB comparison")

class EnhancedTaskAnalysisTool(BaseTool):
    """Fast, flexible task analysis with knowledge grounding"""
    name: str = "analyze_task_enhanced"
    description: str = """
    Enhanced task analysis with knowledge base grounding.
    Compares incoming materials against knowledge base to determine contextual relevance.
    Extracts: complexity, time estimate, category, urgency, knowledge relevance, task type.
    """
    args_schema: Type[BaseModel] = EnhancedTaskAnalysisInput
    
    def _run(
        self, 
        task_name: str, 
        content_summary: str, 
        metadata: Dict, 
        sort_preference: str = "hybrid",
        use_knowledge_grounding: bool = True
    ) -> str:
        start = time.time()
        
        # Standard analysis (existing code)
        complexity = self._estimate_complexity(task_name, content_summary, metadata)
        pages = metadata.get("pages", 0)
        chunks = metadata.get("chunk_count", 0)
        estimated_hours = self._estimate_time(pages, chunks, complexity)
        category = self._categorize_topic(task_name, content_summary)
        is_foundational = self._check_foundational(task_name, content_summary)
        urgency = self._calculate_urgency(metadata.get("deadline"))
        
        # Task type determination
        analysis_dict = {
            'complexity': complexity,
            'category': category,
            'urgency_score': urgency,
            'is_foundational': is_foundational
        }
        task_type = _determine_task_type(task_name, analysis_dict)
        
        # Calculate base scores
        scores = {
            "content": self._score_by_content(category, is_foundational),
            "pages": self._score_by_pages(pages),
            "urgency": urgency,
            "complexity": 10 - complexity,
            "foundational": 10 if is_foundational else 5,
            "hybrid": self._calculate_hybrid_score(urgency, is_foundational, complexity, category)
        }
        
        # Build initial result
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
                "scores": scores,
                "preferred_score": scores.get(sort_preference, scores["hybrid"]),
                "sort_method": sort_preference
            }
        }
        
        # ===== NEW: Knowledge Grounding Integration =====
        if use_knowledge_grounding and content_summary and len(content_summary) > 50:
            try:
                print(f"   🔍 Applying knowledge grounding for: {task_name}")
                
                # Enhance with knowledge base comparison
                result["analysis"] = enhance_task_with_knowledge(
                    task_analysis=result["analysis"],
                    text_content=content_summary,
                    blend_weight=0.30,  # 30% KB influence
                    min_confidence=0.30,  # Min confidence threshold
                    enable_knowledge_boost=True  # Boost for KB gaps
                )
                
                # Use knowledge-adjusted score if available
                if 'knowledge_adjusted_score' in result["analysis"]:
                    result["analysis"]["preferred_score"] = result["analysis"]["knowledge_adjusted_score"]
                    result["analysis"]["scores"]["knowledge_weighted"] = result["analysis"]["knowledge_adjusted_score"]
                
                print(f"   ✅ Knowledge grounding applied")
                
            except Exception as e:
                print(f"   ⚠️ Knowledge grounding failed: {e}")
                pass
        
        duration = time.time() - start
        output = json.dumps(result, indent=2)
        
        ToolLogger.log_call(
            self.name, 
            {"task": task_name, "sort": sort_preference, "kb_enabled": use_knowledge_grounding}, 
            {"result": result, "duration": duration}
        )
        
        return output
    
    def _estimate_complexity(self, name: str, summary: str, metadata: Dict) -> int:
        """Fast complexity estimation"""
        score = 5
        lower = (name + " " + summary).lower()
        
        # Keywords impact
        if any(kw in lower for kw in ["advanced", "complex", "graduate", "phd", "research", "theoretical"]):
            score += 3
        if any(kw in lower for kw in ["intro", "basic", "fundamental", "101", "beginner", "elementary"]):
            score -= 2
        if any(kw in lower for kw in ["intermediate", "applied", "practical"]):
            score += 1
        
        # Page count impact
        pages = metadata.get("pages", 0)
        if pages > 300:
            score += 2
        elif pages > 150:
            score += 1
        elif pages < 50:
            score -= 1
        
        return max(1, min(10, score))
    
    def _estimate_time(self, pages: int, chunks: int, complexity: int) -> float:
        """Realistic time estimation"""
        if pages > 0:
            # Adjust reading speed by complexity
            pages_per_hour = 10 - (complexity * 0.5)  # Harder = slower
            hours = pages / max(pages_per_hour, 3)
        else:
            hours = chunks * 0.15
        
        # Add practice/exercise time for certain types
        if complexity >= 7:
            hours *= 1.3  # Add 30% for difficult material
        
        return round(hours, 1)
    
    def _categorize_topic(self, name: str, summary: str) -> str:
        """Fast categorization"""
        text = (name + " " + summary).lower()
        
        categories = {
            "mathematics": ["math", "calculus", "algebra", "geometry", "statistics", "probability", "equation"],
            "data_science": ["data", "analysis", "machine learning", "ai", "analytics", "ml", "neural", "deep learning"],
            "finance": ["finance", "accounting", "economics", "investment", "trading", "market", "portfolio"],
            "programming": ["programming", "code", "software", "algorithm", "python", "java", "javascript", "coding"],
            "exam_prep": ["exam", "test", "midterm", "final", "quiz", "practice", "assessment"],
            "research": ["research", "thesis", "dissertation", "study", "paper", "journal", "publication"],
            "business": ["business", "management", "strategy", "marketing", "leadership", "entrepreneurship"],
            "science": ["physics", "chemistry", "biology", "science", "laboratory", "experiment"],
        }
        
        for category, keywords in categories.items():
            if any(kw in text for kw in keywords):
                return category
        return "general"
    
    def _check_foundational(self, name: str, summary: str) -> bool:
        """Check if foundational/prerequisite"""
        text = (name + " " + summary).lower()
        indicators = [
            "intro", "introduction", "fundamental", "basic", "foundation",
            "101", "chapter 1", "ch1", "ch.1", "prerequisite", "essentials",
            "beginning", "primer", "overview"
        ]
        return any(ind in text for ind in indicators)
    
    def _calculate_urgency(self, deadline: Optional[str]) -> int:
        """Calculate urgency 1-10"""
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
        """Score by content importance"""
        base_score = 5.0
        
        # Priority categories
        if category in ["mathematics", "data_science", "exam_prep"]:
            base_score += 2.0
        elif category in ["programming", "science"]:
            base_score += 1.5
        
        if is_foundational:
            base_score += 3.0
        
        return base_score
    
    def _score_by_pages(self, pages: int) -> float:
        """Score by page count (shorter first for quick wins)"""
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
        """AI-driven hybrid scoring"""
        score = 0.0
        
        # Urgency (highest weight)
        score += urgency * 2.5
        
        # Foundational (high weight)
        if is_foundational:
            score += 8.0
        
        # Complexity (moderate penalty for very hard)
        if complexity > 8:
            score -= 2.0
        elif complexity < 4:
            score += 1.0
        
        # Category bonus
        if category in ["exam_prep", "mathematics", "data_science"]:
            score += 3.0
        elif category in ["programming", "science"]:
            score += 2.0
        
        return round(score, 2)

# ==================== FLEXIBLE SCHEDULING WITH TASK TYPES ====================

class FlexibleSchedulingInput(BaseModel):
    """Flexible time input"""
    prioritized_tasks: str = Field(description="JSON of prioritized tasks")
    available_time: Dict = Field(description="Time availability (years/months/days/hours)")
    constraints: str = Field(description="Schedule constraints")
    sort_method: str = Field(default="hybrid", description="Sorting method used")

class FlexibleSchedulingTool(BaseTool):
    """Enhanced scheduler with task type classification"""
    name: str = "create_flexible_schedule"
    description: str = """
    Creates schedule from flexible time inputs with task type classification.
    Supports: years, months, days, hours, or combinations.
    Assigns appropriate task types (Theory, Practical, Exam Prep, Assignment, Review, Workshop).
    Adapts to user's sorting preference for optimal planning.
    """
    args_schema: Type[BaseModel] = FlexibleSchedulingInput
    
    def _run(self, prioritized_tasks: str, available_time: Dict, 
             constraints: str, sort_method: str = "hybrid") -> str:
        start = time.time()
        
        tasks = json.loads(prioritized_tasks) if isinstance(prioritized_tasks, str) else prioritized_tasks
        
        # Convert time to hours
        total_hours = self._convert_to_hours(available_time)
        
        # Parse constraints
        constraint_dict = self._parse_constraints(constraints)
        
        # Generate schedule with task types
        schedule = self._allocate_time_with_types(
            tasks, total_hours, constraint_dict, sort_method
        )
        
        # Calculate statistics
        allocated_hours = sum(s["hours"] for s in schedule)
        utilization = (allocated_hours / total_hours * 100) if total_hours > 0 else 0
        
        # Task type distribution
        type_distribution = {}
        for item in schedule:
            task_type = item.get('type', 'Study')
            type_distribution[task_type] = type_distribution.get(task_type, 0) + 1
        
        result = {
            "schedule": schedule,
            "reasoning": self._explain_schedule_with_types(
                schedule, total_hours, available_time, sort_method, type_distribution
            ),
            "total_allocated_hours": allocated_hours,
            "available_hours": total_hours,
            "time_breakdown": available_time,
            "utilization_percent": round(utilization, 1),
            "task_types_distribution": type_distribution
        }
        
        output = json.dumps(result, indent=2)
        ToolLogger.log_call(self.name, {"tasks": len(tasks), "hours": total_hours}, result)
        return output
    
    def _convert_to_hours(self, time_dict: Dict) -> float:
        """Convert flexible time to hours"""
        total = 0.0
        total += time_dict.get("years", 0) * 365 * 24
        total += time_dict.get("months", 0) * 30 * 24
        total += time_dict.get("weeks", 0) * 7 * 24
        total += time_dict.get("days", 0) * 24
        total += time_dict.get("hours", 0)
        return total
    
    def _parse_constraints(self, constraints: str) -> Dict:
        """Enhanced constraint parsing"""
        constraint_dict = {
            "preferred_times": [],
            "max_session": 3.0,
            "avoid_days": [],
            "preferred_days": [],
            "break_duration": 0.25,
            "work_style": "balanced"  # balanced, intensive, relaxed
        }
        
        if not constraints:
            constraint_dict["preferred_times"] = ["morning", "afternoon"]
            return constraint_dict
        
        lower = constraints.lower()
        
        # Time preferences
        if "morning" in lower:
            constraint_dict["preferred_times"].append("morning")
        if "afternoon" in lower:
            constraint_dict["preferred_times"].append("afternoon")
        if "evening" in lower:
            constraint_dict["preferred_times"].append("evening")
        if "night" in lower or "late" in lower:
            constraint_dict["preferred_times"].append("night")
        
        # Default if no preference specified
        if not constraint_dict["preferred_times"]:
            constraint_dict["preferred_times"] = ["morning", "afternoon"]
        
        # Day preferences
        if "weekend only" in lower:
            constraint_dict["preferred_days"] = ["Saturday", "Sunday"]
        elif "no weekend" in lower or "weekday" in lower:
            constraint_dict["avoid_days"] = ["Saturday", "Sunday"]
        
        # Work style
        if "intensive" in lower or "cramming" in lower or "focused" in lower:
            constraint_dict["work_style"] = "intensive"
            constraint_dict["max_session"] = 4.0
        elif "relaxed" in lower or "slow" in lower or "easy" in lower:
            constraint_dict["work_style"] = "relaxed"
            constraint_dict["max_session"] = 2.0
        
        return constraint_dict
    
    def _allocate_time_with_types(self, tasks: List[Dict], total_hours: float, 
                                  constraints: Dict, sort_method: str) -> List[Dict]:
        """Allocate time with task type assignment"""
        schedule = []
        hours_remaining = total_hours
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Apply day preferences
        if constraints.get("preferred_days"):
            available_days = constraints["preferred_days"]
        else:
            avoid = constraints.get("avoid_days", [])
            available_days = [d for d in days if d not in avoid]
        
        if not available_days:
            available_days = days  # Fallback to all days
        
        time_slots = {
            "morning": ["08:00-10:00", "09:00-11:00", "10:00-12:00"],
            "afternoon": ["14:00-16:00", "13:00-15:00", "15:00-17:00"],
            "evening": ["18:00-20:00", "19:00-21:00", "17:00-19:00"],
            "night": ["20:00-22:00", "21:00-23:00", "22:00-00:00"]
        }
        
        preferred_times = constraints.get("preferred_times", ["morning", "afternoon"])
        max_session = constraints.get("max_session", 3.0)
        work_style = constraints.get("work_style", "balanced")
        
        day_idx = 0
        slot_idx = 0
        
        for task in tasks:
            if hours_remaining <= 0.5:  # Min 30 min
                break
            
            task_analysis = task.get("analysis", {})
            task_name = _clean_task_name(task["task"])
            estimated_hours = task_analysis.get("estimated_hours", 2)
            
            # Determine task type
            task_type = task_analysis.get("task_type") or _determine_task_type(task_name, task_analysis)
            
            # Adjust time allocation based on task type and work style
            if task_type == "Exam Prep":
                session_hours = min(estimated_hours * 1.2, max_session)  # Extra time for exam prep
            elif task_type == "Review":
                session_hours = min(estimated_hours * 0.7, 1.5)  # Shorter for reviews
            elif task_type == "Theory" and work_style == "intensive":
                session_hours = min(estimated_hours, max_session * 1.2)  # Longer theory sessions if intensive
            else:
                session_hours = min(estimated_hours * 0.85, max_session)
            
            # Minimum session duration
            session_hours = max(0.5, min(session_hours, hours_remaining))
            
            # Select appropriate time slot based on task type
            if task_type in ["Theory", "Exam Prep"]:
                # Prefer mornings for complex cognitive tasks
                time_pref = "morning" if "morning" in preferred_times else preferred_times[0]
            elif task_type in ["Practical", "Workshop"]:
                # Prefer afternoons for hands-on work
                time_pref = "afternoon" if "afternoon" in preferred_times else preferred_times[0]
            elif task_type == "Review":
                # Evening for review sessions
                time_pref = "evening" if "evening" in preferred_times else preferred_times[-1]
            else:
                time_pref = preferred_times[slot_idx % len(preferred_times)]
            
            # Get specific time slot
            time_options = time_slots.get(time_pref, time_slots["afternoon"])
            specific_time = time_options[slot_idx % len(time_options)]
            
            # Select day
            day = available_days[day_idx % len(available_days)]
            
            schedule.append({
                "task": task_name,
                "day": day,
                "time": specific_time,
                "hours": round(session_hours, 1),
                "type": task_type,  # Include task type
                "priority": task.get("priority", 999),
                "complexity": task_analysis.get("complexity", 5)
            })
            
            hours_remaining -= session_hours
            slot_idx += 1
            
            # Move to next day after filling time slots
            if slot_idx % (len(preferred_times) * 2) == 0:
                day_idx += 1
        
        return schedule
    
    def _explain_schedule_with_types(self, schedule: List[Dict], total_hours: float, 
                                     time_breakdown: Dict, sort_method: str,
                                     type_distribution: Dict) -> str:
        """Generate enhanced reasoning with task types"""
        allocated = sum(s["hours"] for s in schedule)
        utilization = (allocated / total_hours * 100) if total_hours > 0 else 0
        
        # Build time breakdown string
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
        
        # Type distribution summary
        type_summary = ", ".join([f"{count} {ttype}" for ttype, count in type_distribution.items()])
        
        reasoning = f"""**Optimized Weekly Schedule Analysis:**

Total Available Time: {time_str} ({total_hours:.1f} hours total)
Time Allocated: {allocated:.1f} hours
Utilization Rate: {utilization:.1f}%
Prioritization Method: {sort_method}

**Task Type Distribution:**
{type_summary}

**Optimization Strategy:**
The schedule has been algorithmically optimized to:
• Place Theory and Exam Prep sessions during peak cognitive hours (mornings)
• Schedule Practical work when hands-on engagement is optimal (afternoons)
• Position Review sessions for end-of-day consolidation
• Balance session lengths based on complexity and task type
• Maintain sustainable study patterns with appropriate breaks

The {sort_method} prioritization ensures critical materials are addressed first while maintaining learning progression."""
        
        return reasoning

# ==================== TOOL REGISTRY ====================

def get_advanced_agent_tools() -> List[BaseTool]:
    """Return all enhanced tools with knowledge grounding"""
    return [
        EnhancedTaskAnalysisTool(),
        FlexibleSchedulingTool()
    ]