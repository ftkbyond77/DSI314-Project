# core/quiz_agent.py

import os
import json
import random
from typing import Dict, List, Any, Optional
from openai import OpenAI
from django.conf import settings

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class QuizGenerationAgent:
    """
    AI Agent for generating quiz questions based on study materials.
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = client
    
    def generate_quiz(
        self,
        study_plan_data: Dict,
        uploaded_content: List[Dict],
        kb_content: Optional[List[Dict]] = None,
        num_questions: int = 5,
        difficulty_mix: bool = True
    ) -> List[Dict]:
        """Generate quiz questions from content."""
        
        print(f"Generating {num_questions} quiz questions...")
        
        # Generate difficulty distribution
        if difficulty_mix:
            difficulties = self._create_difficulty_distribution(num_questions)
        else:
            difficulties = ['medium'] * num_questions
            
        questions = []
        
        # Generate questions from uploaded content
        # We pass the calculated difficulties to the generator
        upload_questions = self._generate_from_uploads(
            uploaded_content,
            num_questions,
            difficulties
        )
        questions.extend(upload_questions)
        
        # Shuffle and number questions
        random.shuffle(questions)
        for idx, q in enumerate(questions, 1):
            q['question_number'] = idx
        
        return questions

    def _create_difficulty_distribution(self, total: int) -> List[str]:
        """
        Create a balanced difficulty distribution.
        Approximately: 30% easy, 40% medium, 30% hard
        """
        easy_count = max(1, int(total * 0.3))
        hard_count = max(1, int(total * 0.3))
        medium_count = total - easy_count - hard_count
        
        distribution = (
            ['easy'] * easy_count +
            ['medium'] * medium_count +
            ['hard'] * hard_count
        )
        
        # Adjust length if rounding caused mismatch
        if len(distribution) > total:
            distribution = distribution[:total]
        elif len(distribution) < total:
            distribution.extend(['medium'] * (total - len(distribution)))
            
        random.shuffle(distribution)
        return distribution
    
    def _generate_from_uploads(
        self,
        uploaded_content: List[Dict],
        num_questions: int,
        difficulties: List[str]
    ) -> List[Dict]:
        """Generate questions from uploaded files"""
        
        questions = []
        if not uploaded_content:
            return []

        # Resample content if we have fewer chunks than questions
        if len(uploaded_content) < num_questions:
            selected_sources = random.choices(uploaded_content, k=num_questions)
        else:
            selected_sources = random.sample(uploaded_content, num_questions)
        
        for idx, (source, difficulty) in enumerate(zip(selected_sources, difficulties)):
            question = self._generate_single_question(
                content=source.get('text', ''),
                topic=source.get('topic', 'General'),
                source_file=source.get('filename', 'Unknown'),
                difficulty=difficulty
            )
            
            if question:
                questions.append(question)
        
        return questions
    
    def _generate_single_question(
        self,
        content: str,
        topic: str,
        source_file: str,
        difficulty: str
    ) -> Optional[Dict]:
        """Generate a single quiz question using LLM."""
        
        # Truncate content
        max_len = 2500
        if len(content) > max_len:
            content = content[:max_len] + "..."
        
        # Instructions based on difficulty
        diff_instruction = ""
        if difficulty == 'easy':
            diff_instruction = "Create a straightforward question testing basic recall."
        elif difficulty == 'hard':
            diff_instruction = "Create a complex question requiring analysis or synthesis."
        else:
            diff_instruction = "Create a standard question testing comprehension."

        prompt = f"""Based on the following text from "{source_file}", create ONE multiple-choice question.

CONTENT:
{content}

REQUIREMENTS:
- {diff_instruction}
- 4 distinct options (A, B, C, D).
- Only ONE correct answer.
- Return strictly JSON.

FORMAT:
{{
    "question_text": "The question",
    "option_a": "Option A",
    "option_b": "Option B",
    "option_c": "Option C",
    "option_d": "Option D",
    "correct_answer": "a",
    "explanation": "Why it is correct"
}}
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a quiz generator. Output JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            text = response.choices[0].message.content.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
                
            data = json.loads(text)
            
            # Add metadata
            data['difficulty_level'] = difficulty
            data['source_topic'] = topic
            data['source_file'] = source_file
            
            return data
            
        except Exception as e:
            print(f"Error generating question: {e}")
            return None

# ==================== CONVENIENCE FUNCTIONS ====================

def generate_quiz_for_study_plan(
    study_plan,
    num_questions: int = 5,
    focus_task_name: str = None
) -> List[Dict]:
    """
    Generate quiz for a study plan, optionally focusing on one file.
    """
    agent = QuizGenerationAgent()
    
    uploaded_content = []
    
    # 1. Filter uploads based on focus_task_name
    target_uploads = study_plan.uploads.all()
    
    if focus_task_name:
        # Find the file that matches the task name
        filtered = [u for u in target_uploads if u.filename == focus_task_name]
        if filtered:
            target_uploads = filtered
        else:
            # Fallback: If specific file not found, use all (prevents crash)
            print(f"Warning: No upload found matching task '{focus_task_name}'. Using all files.")

    # 2. Extract chunks
    for upload in target_uploads:
        # If focused, get more chunks for better coverage
        limit = 15 if focus_task_name else 5
        chunks = upload.chunks.all().order_by('start_page')[:limit]
        
        for chunk in chunks:
            uploaded_content.append({
                'filename': upload.filename,
                'text': chunk.text,
                'topic': 'General',
                'pages': f"{chunk.start_page}-{chunk.end_page}"
            })
            
    if not uploaded_content:
        print("Error: No content extracted from uploads.")
        return []
        
    # 3. Generate
    questions = agent.generate_quiz(
        study_plan_data=study_plan.plan_json,
        uploaded_content=uploaded_content,
        num_questions=num_questions,
        difficulty_mix=True
    )
    
    return questions