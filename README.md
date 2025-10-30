# Student Assistant: Agentic Academic Planning Platform

## 1. Project Overview

Student Assistant is an agent-driven platform for personalized study planning, leveraging a knowledge-grounded Retrieval-Augmented Generation (RAG) pipeline, dynamic agent prioritization, and robust feedback integration. It enables students to upload study materials, which are processed and chunked, then embedded into a vector database for semantic search and reasoning. Multiple AI agents handle prioritization, scheduling, quiz generation, and feedback, orchestrated asynchronously for scalability and real-time interaction.

Main Components:
- Web Backend: Django + REST (with JWT support)
- Agent Orchestration: LangChain, OpenAI LLMs
- Vector DB: Pinecone (semantic search/knowledge RAG)
- Background Tasks: Celery + Redis for async/parallel computation
- Feedback: User feedback/analytics powering RL-based score adjustment
- Quiz Subsystem: Auto-generates, grades, and explains quiz questions

Directory Structure
- `core/`   - Main business logic (agents, async tasks, RAG, feedback, models)
- `student_assistant/`   - Django project/supporting config
- `RAG/`   - Document processing/utilities (chunking, filtering, embedding, etc.)
- `staticfiles/`, `media/`   - User uploads, static assets, OCR outputs

## 2. Agent System (Core of the Project)

### Agent Prioritization Mechanism

The orchestrator handles two primary types of agents:
- **Prioritization Agent**: Uses LLM-powered routines to analyze, score, and rank study tasks. It blends urgency, foundational status, KB context, and user goals, providing transparent reasoning for each rank decision.
- **Scheduling Agent**: Transforms prioritized lists into realistic weekly study schedules, balancing time availability, task type, cognitive load, and constraints.

#### Prioritization Flow:
1. **Task Analysis** (`EnhancedTaskAnalysisTool`):
    - Extracts from each task: complexity, urgency, foundational, category, pages, KB relevance, and more.
    - Determines task type: theory, assignment, exam prep, review, etc.
2. **Knowledge Grounding**:
    - Blends task intrinsic scores with context from the vector KB (see RAG below).
    - Scores are dynamically blended (user- and data-adaptive weighting from real RL feedback).
3. **LLM-Based Ranking**:
    - Prepares all tasks with guidance scores (from step 1/2), then submits to an LLM which does ranking and produces detailed XML-based reasoning per task (with cross comparisons and justification).
4. **Scheduling**:
    - Schedules are LLM/algorithmically generated based on prioritized order, task types, time blocks, and user constraints.

#### Task Management & Async Execution
- The system uses background Celery tasks (`generate_optimized_plan_async`) for large uploads, multi-task planning, and evaluation.
- Every major processing step is done asynchronously, with real-time progress/status tracked in the user interface.
- Results are validated and sequential by design (no priority gaps/duplicates).

#### Core Agent Tools
Agents use the following tools and utilities:
- Task analysis & flexible scheduling (with preferential time/constraint handling)
- Tool logging and analytics, tracking performance and bottlenecks
- Reasoning integration modules for explainability

## 3. RAG (Retrieval-Augmented Generation) Integration

### Knowledge Base Weighting

- **Knowledge Grounding Engine**: Integrates Pinecone vector search with multi-stage optimizations:
    - Adaptive top-K retrieval based on category size and query complexity
    - Metadata pre-filtering for efficient/accurate search
    - Hybrid retrieval (vector+keyword fallback)
    - Multi-dimensional KB depth analysis (subtopic, source, quality, recency, diversity)
    - Feedback-integrated score blending, adapting weighting factors per user and system analytics
- **Category & Schema Handling**: Dynamic mappers normalize imported content into canonical topics. Automated discovery and periodic recalibration keep the KB up-to-date, balanced, and semantically organized.
- **How Weights Enhance Reasoning**:
    - All agent decisions are directly influenced by semantic KB scores, not just document counts.
    - Lower-coverage (gap) topics are prioritized, while redundancy is penalized, helping students focus on least-covered, most-important foundations.
    - These weights are dynamically tuned via RL from user feedback and analytics.

## 4. Feedback System

- **User Feedback**: Collected at multiple touchpoints: prioritization thumbs/stars/detailed-aspect ratings, task-level suggestions, and overall plan quality.
- **Reinforcement Learning Engine**:
    - Aggregates feedback, analyzes patterns and trends, and proposes/apply scoring model weight adjustments at user, category, or global level.
    - Personalized weights are propagated to agent tools, ensuring the system adapts to real user needs over time.
- **Models & History**: All feedback and analytics are stored for traceability and iterative, data-driven quality improvement.

## 5. Process Flow (End-to-End)

```
[User Input]
   |
   V
(PDF Upload Interface)
   |
   V
[PDF Processing Pipeline]
   - Text extraction + multi-lang OCR (if needed)
   - Intelligent chunking (1000-char chunks, 200-char overlap)
   - Embedding via OpenAI
   - Vector insertion (Pinecone DB)
   |
   V
[Agentic Planning Orchestration]
   |
   |-- [Prioritization Agent (analyzes and ranks tasks with explainability)]
   |
   |-- [Scheduling Agent (allocates time, respects task type and user constraints)]
   |
   V
[Output: Personalized, Knowledge-Grounded Plan]
   |
   V
[Frontend: Progress/status, results dashboard, quiz generator, and feedback control]
   |
   V
[Feedback captured → Reinforcement learning → Adjust agent weights → ... (loop)]
```

**Coordination of Subtasks/Async Processing:**
- Celery tasks orchestrate long-running work (PDF processing, plan generation, KB maintenance), with worker updates and history persisted in the DB and visible in real-time on the frontend.
- User analytics, quiz history, and feedback are updated automatically with every major interaction.

## 6. Evaluation (Analytical Summary)

- **Efficiency**: Highly scalable, async task orchestration, and batch vector operations enable handling of multiple large files.
- **Accuracy**: Blending KB context and real user feedback enables deeply contextual, robust prioritization and scheduling.
- **Flexibility**: Modular agent tool design, dynamic schema/category mapping, and feedback-driven weights allow fast adaptation to new subjects, data types, or planning strategies.
- **Recall/Speed**: Token-aware vector search, dynamic top-K, and batch embeddings maximize retrieval performance and KB utilization.
- **Production Readiness**: Robust error handling, progress tracking, and extensive monitoring/validation are built-in; stateless async workers and modular Django infrastructure ensure maintainability.

## 7. Preprocessing Requirement (Emoji Removal)

All system components preprocess and sanitize all input files, specifically ensuring that emojis are removed from every file before further processing, as required. No other text modification is performed.

---

## Diagram: Textual System Architecture

```
Student Assistant
-------------------------
| Agent Layer   |   RAG KB   |  Plan Module  |
| (Django App)  | (Pinecone) | (Scheduling)  |
-------------------------
| Prioritize    |   Chunk    |   Schedule    |
| (LLM Rank)    | Embedding/ |
| Sched/Quiz    | VectorDB   |
-------------------------
Feedback Loop: RL Scores to Weights
```

## Code/Workflow Summary

- **Task Preprocessing & Emoji Stripping**: All file content is sanitized on ingestion to ensure no emojis disrupt vector DB or LLM processes.
- **Agentic Planning Code** (`core/agentic_core.py`, `tasks_agentic_optimized.py`): Orchestrates prioritization and scheduling, invoking reasoning templates and tools with dynamic feedback-influenced settings.
- **RAG Code** (`core/knowledge_weighting.py` and helpers): Powers all contextual grounding, adaptive querying, gap detection, and feedback-personalized weighting.
- **Feedback Handling** (`core/feedback_system.py`): Robust models and RL updating ensure user interaction is harnessed for ongoing improvement.

---

## Conclusion

Student Assistant is a robust, production-ready platform at the intersection of explainable agent orchestration, knowledge-grounded LLM reasoning (RAG), and actionable feedback reinforcement. Its modular design makes it ideal for students and educators needing transparent, adaptive, and context-rich learning support at scale.

---

This README is autogenerated, code-level detailed, and all content has been sanitized as specified (emojis removed).