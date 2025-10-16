# core/agent_tools_advanced.py - Enhanced Flexible Agent Tools

from langchain.tools import BaseTool
from typing import Type, List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import json
import time
from enum import Enum

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
    def log(cls, tool_name: str, input_data: Dict, output_data: str, duration: float):
        """Log with performance tracking"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "input": input_data,
            "output": output_data[:500],
            "duration_ms": round(duration * 1000, 2)
        }
        cls.logs.append(log_entry)
        
        # Update performance metrics
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

class EnhancedTaskAnalysisTool(BaseTool):
    """Fast, flexible task analysis"""
    name: str = "analyze_task_enhanced"
    description: str = """
    Enhanced task analysis that considers user preferences.
    Extracts: complexity, time estimate, category, urgency, foundational importance.
    Respects user's sorting preference (content/pages/urgency/complexity/foundational/hybrid).
    """
    args_schema: Type[BaseModel] = EnhancedTaskAnalysisInput
    
    def _run(self, task_name: str, content_summary: str, metadata: Dict, sort_preference: str = "hybrid") -> str:
        start = time.time()
        
        # Core analysis
        complexity = self._estimate_complexity(task_name, content_summary, metadata)
        pages = metadata.get("pages", 0)
        chunks = metadata.get("chunk_count", 0)
        estimated_hours = self._estimate_time(pages, chunks, complexity)
        category = self._categorize_topic(task_name, content_summary)
        is_foundational = self._check_foundational(task_name, content_summary)
        urgency = self._calculate_urgency(metadata.get("deadline"))
        
        # Calculate base scores for different sorting methods
        scores = {
            "content": self._score_by_content(category, is_foundational),
            "pages": self._score_by_pages(pages),
            "urgency": urgency,
            "complexity": 10 - complexity,  # Lower complexity = higher priority initially
            "foundational": 10 if is_foundational else 5,
            "hybrid": self._calculate_hybrid_score(
                urgency, is_foundational, complexity, category
            )
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
                "scores": scores,
                "preferred_score": scores.get(sort_preference, scores["hybrid"]),
                "sort_method": sort_preference
            }
        }
        
        output = json.dumps(result, indent=2)
        ToolLogger.log(self.name, {"task": task_name, "sort": sort_preference}, output, time.time() - start)
        return output
    
    def _estimate_complexity(self, name: str, summary: str, metadata: Dict) -> int:
        """Fast complexity estimation"""
        score = 5
        lower = (name + " " + summary).lower()
        
        # Keywords impact
        if any(kw in lower for kw in ["advanced", "complex", "graduate", "phd", "research"]):
            score += 3
        if any(kw in lower for kw in ["intro", "basic", "fundamental", "101", "beginner"]):
            score -= 2
        if any(kw in lower for kw in ["intermediate", "applied"]):
            score += 1
        
        # Page count impact
        pages = metadata.get("pages", 0)
        if pages > 300:
            score += 2
        elif pages > 150:
            score += 1
        
        return max(1, min(10, score))
    
    def _estimate_time(self, pages: int, chunks: int, complexity: int) -> float:
        """Realistic time estimation"""
        if pages > 0:
            # Adjust reading speed by complexity
            pages_per_hour = 10 - (complexity * 0.5)  # Harder = slower
            hours = pages / max(pages_per_hour, 3)
        else:
            hours = chunks * 0.15
        
        return round(hours, 1)
    
    def _categorize_topic(self, name: str, summary: str) -> str:
        """Fast categorization"""
        text = (name + " " + summary).lower()
        
        categories = {
            "mathematics": ["math", "calculus", "algebra", "geometry", "statistics", "probability"],
            "data_science": ["data", "analysis", "machine learning", "ai", "analytics", "ml"],
            "finance": ["finance", "accounting", "economics", "investment", "trading"],
            "programming": ["programming", "code", "software", "algorithm", "python", "java"],
            "exam_prep": ["exam", "test", "midterm", "final", "quiz", "practice"],
            "research": ["research", "thesis", "dissertation", "study", "paper"],
            "business": ["business", "management", "strategy", "marketing"],
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
            "101", "chapter 1", "ch1", "prerequisite", "essentials"
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
        
        return round(score, 2)

# ==================== FLEXIBLE SCHEDULING ====================

class FlexibleSchedulingInput(BaseModel):
    """Flexible time input"""
    prioritized_tasks: str = Field(description="JSON of prioritized tasks")
    available_time: Dict = Field(description="Time availability (years/months/days/hours)")
    constraints: str = Field(description="Schedule constraints")
    sort_method: str = Field(default="hybrid", description="Sorting method used")

class FlexibleSchedulingTool(BaseTool):
    """Flexible scheduler supporting multiple time units"""
    name: str = "create_flexible_schedule"
    description: str = """
    Creates schedule from flexible time inputs.
    Supports: years, months, days, hours, or combinations.
    Adapts to user's sorting preference for optimal planning.
    """
    args_schema: Type[BaseModel] = FlexibleSchedulingInput
    
    def _run(self, prioritized_tasks: str, available_time: Dict, 
             constraints: str, sort_method: str = "hybrid") -> str:
        start = time.time()
        
        tasks = json.loads(prioritized_tasks)
        
        # Convert time to hours
        total_hours = self._convert_to_hours(available_time)
        
        # Parse constraints
        constraint_dict = self._parse_constraints(constraints)
        
        # Generate schedule
        schedule = self._allocate_time_flexible(
            tasks, total_hours, constraint_dict, sort_method
        )
        
        result = {
            "schedule": schedule,
            "reasoning": self._explain_flexible_schedule(
                schedule, total_hours, available_time, sort_method
            ),
            "total_allocated_hours": sum(s["hours"] for s in schedule),
            "available_hours": total_hours,
            "time_breakdown": available_time,
            "utilization_percent": round(
                (sum(s["hours"] for s in schedule) / total_hours * 100) if total_hours > 0 else 0, 1
            )
        }
        
        output = json.dumps(result, indent=2)
        ToolLogger.log(self.name, {"tasks": len(tasks), "hours": total_hours}, output, time.time() - start)
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
            "preferred_times": ["afternoon", "evening"],
            "max_session": 3.0,
            "avoid_days": [],
            "preferred_days": [],
            "break_duration": 0.25,
            "work_style": "balanced"  # balanced, intensive, relaxed
        }
        
        if not constraints:
            return constraint_dict
        
        lower = constraints.lower()
        
        # Time preferences
        if "morning" in lower:
            constraint_dict["preferred_times"].insert(0, "morning")
        if "night" in lower or "late" in lower:
            constraint_dict["preferred_times"].append("night")
        
        # Day preferences
        if "weekend only" in lower:
            constraint_dict["preferred_days"] = ["Saturday", "Sunday"]
        elif "no weekend" in lower or "weekday" in lower:
            constraint_dict["avoid_days"] = ["Saturday", "Sunday"]
        
        # Work style
        if "intensive" in lower or "cramming" in lower:
            constraint_dict["work_style"] = "intensive"
            constraint_dict["max_session"] = 5.0
        elif "relaxed" in lower or "slow" in lower:
            constraint_dict["work_style"] = "relaxed"
            constraint_dict["max_session"] = 2.0
        
        return constraint_dict
    
    def _allocate_time_flexible(self, tasks: List[Dict], total_hours: float, 
                                constraints: Dict, sort_method: str) -> List[Dict]:
        """Flexible time allocation"""
        schedule = []
        hours_remaining = total_hours
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Apply day preferences
        if constraints.get("preferred_days"):
            available_days = constraints["preferred_days"]
        else:
            avoid = constraints.get("avoid_days", [])
            available_days = [d for d in days if d not in avoid]
        
        time_slots = {
            "morning": "08:00-12:00",
            "afternoon": "13:00-17:00",
            "evening": "18:00-22:00",
            "night": "22:00-01:00"
        }
        
        preferred_times = constraints.get("preferred_times", ["afternoon"])
        max_session = constraints.get("max_session", 3.0)
        work_style = constraints.get("work_style", "balanced")
        
        day_idx, time_idx = 0, 0
        
        for task in tasks:
            if hours_remaining <= 0.5:  # Min 30 min
                break
            
            task_hours = task["analysis"]["estimated_hours"]
            
            # Adjust allocation by work style
            if work_style == "intensive":
                allocated = min(task_hours, hours_remaining, max_session)
            elif work_style == "relaxed":
                allocated = min(task_hours * 0.7, hours_remaining, max_session)
            else:
                allocated = min(task_hours * 0.85, hours_remaining, max_session)
            
            allocated = max(0.5, allocated)  # Minimum 30 min
            
            day = available_days[day_idx % len(available_days)]
            time_pref = preferred_times[time_idx % len(preferred_times)]
            time_slot = time_slots.get(time_pref, "13:00-17:00")
            
            schedule.append({
                "task": task["task"],
                "day": day,
                "time": time_slot,
                "hours": round(allocated, 1),
                "priority": task.get("priority", 999),
                "category": task["analysis"].get("category", "general"),
                "complexity": task["analysis"].get("complexity", 5)
            })
            
            hours_remaining -= allocated
            time_idx += 1
            
            if time_idx % len(preferred_times) == 0:
                day_idx += 1
        
        return schedule
    
    def _explain_flexible_schedule(self, schedule: List[Dict], total_hours: float, 
                                   time_breakdown: Dict, sort_method: str) -> str:
        """Generate detailed reasoning"""
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
        
        reasoning = (
            f"Created schedule using '{sort_method}' prioritization method. "
            f"Allocated {allocated}h out of your available {time_str} ({total_hours}h total), "
            f"achieving {utilization:.1f}% utilization. "
        )
        
        if schedule:
            top = schedule[0]
            reasoning += (
                f"Started with '{top['task']}' on {top['day']} at {top['time']} "
                f"for {top['hours']}h due to highest priority. "
            )
        
        if utilization < 80:
            remaining = total_hours - allocated
            reasoning += (
                f"You have {remaining:.1f}h remaining for review, breaks, or additional study. "
            )
        
        return reasoning

# ==================== TOOL REGISTRY ====================

def get_advanced_agent_tools() -> List[BaseTool]:
    """Return all enhanced tools"""
    return [
        EnhancedTaskAnalysisTool(),
        FlexibleSchedulingTool()
    ]