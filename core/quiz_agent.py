# core/quiz_agent.py - Quiz Generation and Grading Agent

import os
import json
import random
from typing import Dict, List, Any, Optional
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


class QuizGenerationAgent:
    """
    AI Agent for generating quiz questions based on study materials.
    Uses LLM to create contextual, difficulty-appropriate questions.
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
        """
        Generate quiz questions from study plan materials.
        
        Args:
            study_plan_data: StudyPlanHistory plan_json data
            uploaded_content: List of {filename, text, topic} from uploaded files
            kb_content: Optional list of relevant KB content
            num_questions: Total questions to generate (default 5)
            difficulty_mix: Whether to mix difficulties (True) or keep uniform
        
        Returns:
            List of question dictionaries with structure:
            {
                'question_text': str,
                'option_a': str, 'option_b': str, 'option_c': str, 'option_d': str,
                'correct_answer': str ('a'/'b'/'c'/'d'),
                'difficulty_level': str ('easy'/'medium'/'hard'),
                'source_topic': str,
                'source_file': str,
                'explanation': str,
                'explanation_wrong': dict
            }
        """
        
        print(f"🎯 Generating {num_questions} quiz questions...")
        
        # Determine source distribution (80% uploaded, 20% KB)
        num_from_uploads = max(1, int(num_questions * 0.8))
        num_from_kb = num_questions - num_from_uploads
        
        # Generate difficulty distribution
        if difficulty_mix:
            difficulties = self._create_difficulty_distribution(num_questions)
        else:
            difficulties = ['medium'] * num_questions
        
        questions = []
        
        # Generate questions from uploaded content
        upload_questions = self._generate_from_uploads(
            uploaded_content,
            num_from_uploads,
            difficulties[:num_from_uploads]
        )
        questions.extend(upload_questions)
        
        # Generate questions from KB (if available)
        if kb_content and num_from_kb > 0:
            kb_questions = self._generate_from_kb(
                kb_content,
                num_from_kb,
                difficulties[num_from_uploads:]
            )
            questions.extend(kb_questions)
        
        # Shuffle and number questions
        random.shuffle(questions)
        for idx, q in enumerate(questions, 1):
            q['question_number'] = idx
        
        print(f"✅ Generated {len(questions)} questions successfully")
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
        
        # Sample content sources
        if len(uploaded_content) < num_questions:
            # If fewer files than questions, reuse files
            selected_sources = random.choices(uploaded_content, k=num_questions)
        else:
            # Otherwise, sample without replacement
            selected_sources = random.sample(uploaded_content, num_questions)
        
        for idx, (source, difficulty) in enumerate(zip(selected_sources, difficulties)):
            print(f"  Generating question {idx + 1}/{num_questions} (difficulty: {difficulty})...")
            
            question = self._generate_single_question(
                content=source.get('text', ''),
                topic=source.get('topic', 'General'),
                source_file=source.get('filename', 'Unknown'),
                difficulty=difficulty,
                source_type='uploaded'
            )
            
            if question:
                questions.append(question)
        
        return questions
    
    def _generate_from_kb(
        self,
        kb_content: List[Dict],
        num_questions: int,
        difficulties: List[str]
    ) -> List[Dict]:
        """Generate questions from knowledge base"""
        
        questions = []
        
        # Sample KB sources
        if len(kb_content) < num_questions:
            selected_sources = random.choices(kb_content, k=num_questions)
        else:
            selected_sources = random.sample(kb_content, num_questions)
        
        for idx, (source, difficulty) in enumerate(zip(selected_sources, difficulties)):
            print(f"  Generating KB question {idx + 1}/{num_questions}...")
            
            question = self._generate_single_question(
                content=source.get('text', ''),
                topic=source.get('category', 'General'),
                source_file='Knowledge Base',
                difficulty=difficulty,
                source_type='knowledge_base'
            )
            
            if question:
                questions.append(question)
        
        return questions
    
    def _generate_single_question(
        self,
        content: str,
        topic: str,
        source_file: str,
        difficulty: str,
        source_type: str
    ) -> Optional[Dict]:
        """
        Generate a single quiz question using LLM.
        """
        
        # Truncate content if too long (keep within token limits)
        max_content_length = 2000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        # Create prompt based on difficulty
        difficulty_instructions = {
            'easy': "Create a straightforward question testing basic recall or simple understanding.",
            'medium': "Create a moderately challenging question requiring comprehension and application.",
            'hard': "Create a challenging question requiring analysis, synthesis, or deep understanding."
        }
        
        prompt = f"""Based on the following content from "{source_file}" about {topic}, generate ONE high-quality multiple-choice quiz question.

CONTENT:
{content}

REQUIREMENTS:
- {difficulty_instructions[difficulty]}
- Create 4 distinct options (A, B, C, D)
- Only ONE option should be clearly correct
- Other options should be plausible but incorrect
- Provide a clear explanation for the correct answer
- Explain why each wrong answer is incorrect

IMPORTANT: Respond with ONLY valid JSON in this exact format:
{{
    "question_text": "The actual question",
    "option_a": "First option",
    "option_b": "Second option",
    "option_c": "Third option",
    "option_d": "Fourth option",
    "correct_answer": "a",
    "explanation": "Why this answer is correct",
    "explanation_wrong": {{
        "b": "Why B is wrong",
        "c": "Why C is wrong",
        "d": "Why D is wrong"
    }}
}}

Generate the question now:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert quiz generator. Create educational, fair, and well-structured multiple-choice questions. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=800
            )
            
            # Parse LLM response
            response_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON if wrapped in code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            question_data = json.loads(response_text)
            
            # Add metadata
            question_data['difficulty_level'] = difficulty
            question_data['source_topic'] = topic
            question_data['source_file'] = source_file
            question_data['source_type'] = source_type
            
            return question_data
            
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON parsing error: {e}")
            print(f"  Response was: {response_text[:200]}")
            return None
        except Exception as e:
            print(f"  ⚠️  Error generating question: {e}")
            return None
    
    def grade_quiz(
        self,
        questions: List[Dict],
        user_answers: Dict[int, str]
    ) -> Dict[str, Any]:
        """
        Grade user's quiz responses.
        
        Args:
            questions: List of question dictionaries
            user_answers: Dict mapping question_number -> user's answer ('a'/'b'/'c'/'d')
        
        Returns:
            Grading results with score, feedback, and analysis
        """
        
        results = {
            'total_questions': len(questions),
            'correct_count': 0,
            'incorrect_count': 0,
            'score': 0.0,
            'question_results': [],
            'performance_by_difficulty': {
                'easy': {'correct': 0, 'total': 0},
                'medium': {'correct': 0, 'total': 0},
                'hard': {'correct': 0, 'total': 0}
            },
            'performance_by_topic': {}
        }
        
        for question in questions:
            qnum = question['question_number']
            user_answer = user_answers.get(qnum, '')
            correct_answer = question['correct_answer']
            is_correct = (user_answer.lower() == correct_answer.lower())
            
            # Update counts
            if is_correct:
                results['correct_count'] += 1
            else:
                results['incorrect_count'] += 1
            
            # Track by difficulty
            difficulty = question.get('difficulty_level', 'medium')
            results['performance_by_difficulty'][difficulty]['total'] += 1
            if is_correct:
                results['performance_by_difficulty'][difficulty]['correct'] += 1
            
            # Track by topic
            topic = question.get('source_topic', 'General')
            if topic not in results['performance_by_topic']:
                results['performance_by_topic'][topic] = {'correct': 0, 'total': 0}
            results['performance_by_topic'][topic]['total'] += 1
            if is_correct:
                results['performance_by_topic'][topic]['correct'] += 1
            
            # Question-level result
            question_result = {
                'question_number': qnum,
                'question_text': question['question_text'],
                'user_answer': user_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct,
                'difficulty': difficulty,
                'topic': topic,
                'explanation': question.get('explanation', ''),
                'options': {
                    'a': question.get('option_a', ''),
                    'b': question.get('option_b', ''),
                    'c': question.get('option_c', ''),
                    'd': question.get('option_d', '')
                }
            }
            
            # Add explanation for wrong answer if applicable
            if not is_correct and user_answer:
                explanation_wrong = question.get('explanation_wrong', {})
                question_result['why_wrong'] = explanation_wrong.get(user_answer, '')
            
            results['question_results'].append(question_result)
        
        # Calculate final score
        if results['total_questions'] > 0:
            results['score'] = (results['correct_count'] / results['total_questions']) * 100
        
        # Generate performance feedback
        results['feedback'] = self._generate_feedback(results)
        
        return results
    
    def _generate_feedback(self, results: Dict) -> Dict[str, str]:
        """Generate personalized feedback based on performance"""
        
        score = results['score']
        feedback = {}
        
        # Overall feedback
        if score >= 90:
            feedback['overall'] = "🎉 Outstanding! You demonstrated excellent mastery of the material."
        elif score >= 75:
            feedback['overall'] = "👏 Great job! You have a strong understanding of most concepts."
        elif score >= 60:
            feedback['overall'] = "👍 Good effort! You understand the basics but could benefit from review."
        else:
            feedback['overall'] = "💪 Keep studying! Focus on understanding the core concepts better."
        
        # Difficulty-specific feedback
        diff_feedback = []
        for difficulty, stats in results['performance_by_difficulty'].items():
            if stats['total'] > 0:
                pct = (stats['correct'] / stats['total']) * 100
                if pct < 50:
                    diff_feedback.append(f"Focus on {difficulty} concepts")
        
        if diff_feedback:
            feedback['improvement_areas'] = ", ".join(diff_feedback)
        
        # Topic-specific feedback
        weak_topics = []
        for topic, stats in results['performance_by_topic'].items():
            if stats['total'] > 0:
                pct = (stats['correct'] / stats['total']) * 100
                if pct < 60:
                    weak_topics.append(topic)
        
        if weak_topics:
            feedback['topics_to_review'] = ", ".join(weak_topics)
        
        return feedback


# ==================== CONVENIENCE FUNCTIONS ====================

def generate_quiz_for_study_plan(
    study_plan,
    num_questions: int = 5
) -> List[Dict]:
    """
    Main entry point for quiz generation.
    
    Args:
        study_plan: StudyPlanHistory instance
        num_questions: Number of questions to generate
    
    Returns:
        List of generated question dictionaries
    """
    
    agent = QuizGenerationAgent()
    
    # Extract content from uploaded files
    uploaded_content = []
    for upload in study_plan.uploads.all():
        chunks = upload.chunks.all()[:10]  # Sample up to 10 chunks per file
        
        for chunk in chunks:
            uploaded_content.append({
                'filename': upload.filename,
                'text': chunk.text,
                'topic': 'General',  # Could be enhanced with topic extraction
                'pages': f"{chunk.start_page}-{chunk.end_page}"
            })
    
    # TODO: Optionally fetch relevant KB content for the 20% mix
    kb_content = None  # Implement KB sampling if needed
    
    # Generate quiz
    questions = agent.generate_quiz(
        study_plan_data=study_plan.plan_json,
        uploaded_content=uploaded_content,
        kb_content=kb_content,
        num_questions=num_questions,
        difficulty_mix=True
    )
    
    return questions


def grade_user_quiz(questions: List[Dict], user_answers: Dict[int, str]) -> Dict:
    """
    Grade a completed quiz.
    
    Args:
        questions: List of question dictionaries
        user_answers: Dict mapping question_number -> answer
    
    Returns:
        Grading results dictionary
    """
    
    agent = QuizGenerationAgent()
    return agent.grade_quiz(questions, user_answers)