#!/usr/bin/env python3
"""
quiz_debug.py - Debug and Testing Tool for Quiz Generation and Grading System

This script provides comprehensive testing and debugging capabilities for the 
Student Assistant's AI-powered quiz generation and grading system.

Features:
- Test quiz generation with various content types
- Validate question quality and structure
- Test grading accuracy and performance
- Simulate user quiz sessions
- Test LLM integration and response parsing
- Performance benchmarking and optimization
- Database integrity checks for quiz data

Usage:
    python quiz_debug.py [command] [options]
    
Commands:
    test-generation     - Test quiz generation functionality
    test-grading        - Test quiz grading and scoring
    simulate-session    - Simulate complete quiz session
    validate-questions  - Validate question quality and structure
    test-llm-integration - Test LLM API integration
    benchmark          - Performance benchmarking
    check-db           - Database integrity checks
    reset-test-data    - Clean up test data
    full-test          - Run complete test suite
"""

import os
import sys
import django
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'student_assistant.settings')
django.setup()

from django.contrib.auth import get_user_model
from django.utils import timezone
from django.db import transaction

from core.models import (
    StudyPlanHistory, QuizSession, QuizQuestion, QuizAnswer,
    UserAnalytics, Upload, Chunk
)
from core.quiz_agent import (
    QuizGenerationAgent, generate_quiz_for_study_plan,
    grade_user_quiz
)

User = get_user_model()


@dataclass
class TestResult:
    """Container for test results"""
    test_name: str
    success: bool
    message: str
    duration: float
    data: Optional[Dict] = None


class QuizDebugger:
    """Main debugger class for quiz system testing"""
    
    def __init__(self):
        self.test_user = None
        self.test_plan = None
        self.test_quiz = None
        self.test_results = []
        self.agent = QuizGenerationAgent()
        
    def setup_test_environment(self) -> bool:
        """Setup test user, study plan, and quiz for testing"""
        try:
            print("🔧 Setting up test environment...")
            
            # Create or get test user
            self.test_user, created = User.objects.get_or_create(
                username='quiz_debug_user',
                defaults={
                    'email': 'quiz_debug@test.com',
                    'first_name': 'Quiz',
                    'last_name': 'Debug'
                }
            )
            
            if created:
                print(f"✅ Created test user: {self.test_user.username}")
            else:
                print(f"✅ Using existing test user: {self.test_user.username}")
            
            # Create test study plan with realistic content
            self.test_plan, created = StudyPlanHistory.objects.get_or_create(
                user=self.test_user,
                user_goal="Quiz testing study plan",
                defaults={
                    'sort_method': 'hybrid',
                    'constraints': 'Test constraints for quiz generation',
                    'time_input': {'weeks': 3, 'days': 2},
                    'total_hours': 75.0,
                    'total_files': 2,
                    'total_pages': 25,
                    'total_chunks': 12,
                    'plan_json': [
                        {
                            'file': 'machine_learning_basics.pdf',
                            'task': 'Study supervised learning concepts',
                            'priority': 9,
                            'reasoning': 'High priority foundational ML content'
                        },
                        {
                            'file': 'machine_learning_basics.pdf',
                            'task': 'Practice with linear regression',
                            'priority': 7,
                            'reasoning': 'Medium priority practical application'
                        },
                        {
                            'file': 'deep_learning_advanced.pdf',
                            'task': 'Learn neural network architectures',
                            'priority': 8,
                            'reasoning': 'High priority advanced concepts'
                        },
                        {
                            'file': 'deep_learning_advanced.pdf',
                            'task': 'Implement CNN from scratch',
                            'priority': 6,
                            'reasoning': 'Medium priority hands-on practice'
                        }
                    ],
                    'execution_time': 4.2,
                    'tool_calls': 15,
                    'status': 'active'
                }
            )
            
            if created:
                print(f"✅ Created test study plan: {self.test_plan.id}")
            else:
                print(f"✅ Using existing test study plan: {self.test_plan.id}")
            
            # Create test uploads and chunks for realistic content
            self._create_test_content()
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting up test environment: {e}")
            return False
    
    def _create_test_content(self):
        """Create test uploads and chunks with realistic content"""
        try:
            # Create test upload
            upload, created = Upload.objects.get_or_create(
                user=self.test_user,
                filename='machine_learning_basics.pdf',
                defaults={
                    'pages': 15,
                    'status': 'processed',
                    'ocr_pages': 0,
                    'ocr_used': False
                }
            )
            
            if created:
                # Add to study plan
                self.test_plan.uploads.add(upload)
                
                # Create realistic chunks
                test_chunks = [
                    {
                        'text': 'Supervised learning is a type of machine learning where algorithms learn from labeled training data. The goal is to learn a mapping from inputs to outputs based on example input-output pairs. Common supervised learning algorithms include linear regression, logistic regression, decision trees, and support vector machines.',
                        'start_page': 1,
                        'end_page': 1
                    },
                    {
                        'text': 'Linear regression is a fundamental supervised learning algorithm used for predicting continuous values. It assumes a linear relationship between the input features and the target variable. The algorithm finds the best line that minimizes the sum of squared errors between predicted and actual values.',
                        'start_page': 2,
                        'end_page': 2
                    },
                    {
                        'text': 'Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers. Each connection has a weight that is adjusted during training. Deep learning uses neural networks with multiple hidden layers to learn complex patterns in data.',
                        'start_page': 3,
                        'end_page': 3
                    },
                    {
                        'text': 'Convolutional Neural Networks (CNNs) are specialized neural networks for processing grid-like data such as images. They use convolutional layers to detect local features and pooling layers to reduce spatial dimensions. CNNs have revolutionized computer vision tasks like image classification and object detection.',
                        'start_page': 4,
                        'end_page': 4
                    },
                    {
                        'text': 'Overfitting occurs when a model learns the training data too well, including noise and irrelevant patterns. This leads to poor performance on new, unseen data. Common techniques to prevent overfitting include regularization, dropout, early stopping, and cross-validation.',
                        'start_page': 5,
                        'end_page': 5
                    }
                ]
                
                for i, chunk_data in enumerate(test_chunks):
                    Chunk.objects.create(
                        upload=upload,
                        chunk_id=f'ml_basics_chunk_{i+1}',
                        text=chunk_data['text'],
                        start_page=chunk_data['start_page'],
                        end_page=chunk_data['end_page'],
                        embedding_id=f'emb_{i+1}'
                    )
                
                print(f"✅ Created test content with {len(test_chunks)} chunks")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not create test content: {e}")
    
    def test_quiz_generation(self) -> TestResult:
        """Test quiz generation functionality"""
        start_time = time.time()
        
        try:
            print("\n🧪 Testing quiz generation...")
            
            # Test different generation scenarios
            scenarios = [
                {
                    'name': 'Standard Generation',
                    'num_questions': 5,
                    'difficulty_mix': True
                },
                {
                    'name': 'Easy Questions Only',
                    'num_questions': 3,
                    'difficulty_mix': False
                },
                {
                    'name': 'Large Quiz',
                    'num_questions': 10,
                    'difficulty_mix': True
                }
            ]
            
            generated_quizzes = []
            
            for scenario in scenarios:
                print(f"  Testing: {scenario['name']}")
                
                # Generate quiz using the agent
                questions = self.agent.generate_quiz(
                    study_plan_data=self.test_plan.plan_json,
                    uploaded_content=self._get_uploaded_content(),
                    kb_content=None,
                    num_questions=scenario['num_questions'],
                    difficulty_mix=scenario['difficulty_mix']
                )
                
                # Validate questions
                validation_result = self._validate_questions(questions)
                
                generated_quizzes.append({
                    'scenario': scenario['name'],
                    'questions': questions,
                    'validation': validation_result
                })
                
                print(f"    ✅ Generated {len(questions)} questions")
                print(f"    Validation: {validation_result['valid']} ({validation_result['issues']} issues)")
            
            duration = time.time() - start_time
            
            total_questions = sum(len(q['questions']) for q in generated_quizzes)
            total_issues = sum(q['validation']['issues'] for q in generated_quizzes)
            
            return TestResult(
                test_name="Quiz Generation",
                success=total_issues == 0,
                message=f"Generated {total_questions} questions across {len(scenarios)} scenarios",
                duration=duration,
                data={
                    'total_questions': total_questions,
                    'total_issues': total_issues,
                    'scenarios': len(scenarios)
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Quiz Generation",
                success=False,
                message=f"Error: {str(e)}",
                duration=duration
            )
    
    def _get_uploaded_content(self) -> List[Dict]:
        """Get uploaded content for quiz generation"""
        content = []
        
        for upload in self.test_plan.uploads.all():
            chunks = upload.chunks.all()[:5]  # Sample first 5 chunks
            
            for chunk in chunks:
                content.append({
                    'filename': upload.filename,
                    'text': chunk.text,
                    'topic': 'Machine Learning',
                    'pages': f"{chunk.start_page}-{chunk.end_page}"
                })
        
        return content
    
    def _validate_questions(self, questions: List[Dict]) -> Dict:
        """Validate question structure and quality"""
        issues = []
        
        for i, question in enumerate(questions):
            # Check required fields
            required_fields = [
                'question_text', 'option_a', 'option_b', 'option_c', 'option_d',
                'correct_answer', 'explanation'
            ]
            
            for field in required_fields:
                if field not in question or not question[field]:
                    issues.append(f"Question {i+1}: Missing {field}")
            
            # Check correct answer format
            if 'correct_answer' in question:
                if question['correct_answer'] not in ['a', 'b', 'c', 'd']:
                    issues.append(f"Question {i+1}: Invalid correct_answer format")
            
            # Check question text length
            if 'question_text' in question and len(question['question_text']) < 20:
                issues.append(f"Question {i+1}: Question text too short")
            
            # Check options are different
            if all(field in question for field in ['option_a', 'option_b', 'option_c', 'option_d']):
                options = [question['option_a'], question['option_b'], question['option_c'], question['option_d']]
                if len(set(options)) < 4:
                    issues.append(f"Question {i+1}: Duplicate options found")
        
        return {
            'valid': len(issues) == 0,
            'issues': len(issues),
            'details': issues
        }
    
    def test_quiz_grading(self) -> TestResult:
        """Test quiz grading and scoring functionality"""
        start_time = time.time()
        
        try:
            print("\n📊 Testing quiz grading...")
            
            # Generate test questions
            questions = self.agent.generate_quiz(
                study_plan_data=self.test_plan.plan_json,
                uploaded_content=self._get_uploaded_content(),
                num_questions=5,
                difficulty_mix=True
            )
            
            if not questions:
                return TestResult(
                    test_name="Quiz Grading",
                    success=False,
                    message="No questions generated for testing",
                    duration=time.time() - start_time
                )
            
            # Test different answer patterns
            test_scenarios = [
                {
                    'name': 'All Correct',
                    'answers': {i+1: q['correct_answer'] for i, q in enumerate(questions)}
                },
                {
                    'name': 'All Wrong',
                    'answers': {i+1: 'a' if q['correct_answer'] != 'a' else 'b' for i, q in enumerate(questions)}
                },
                {
                    'name': 'Mixed Performance',
                    'answers': {i+1: q['correct_answer'] if i % 2 == 0 else 'a' for i, q in enumerate(questions)}
                },
                {
                    'name': 'Partial Completion',
                    'answers': {i+1: q['correct_answer'] for i, q in enumerate(questions[:3])}
                }
            ]
            
            grading_results = []
            
            for scenario in test_scenarios:
                print(f"  Testing: {scenario['name']}")
                
                # Grade the quiz
                result = self.agent.grade_quiz(questions, scenario['answers'])
                
                # Validate grading result
                expected_correct = sum(
                    1 for i, q in enumerate(questions)
                    if scenario['answers'].get(i+1) == q['correct_answer']
                )
                
                if result['correct_count'] != expected_correct:
                    issues.append(f"Grading mismatch: expected {expected_correct}, got {result['correct_count']}")
                
                grading_results.append({
                    'scenario': scenario['name'],
                    'result': result,
                    'expected_correct': expected_correct
                })
                
                print(f"    Score: {result['score']:.1f}% ({result['correct_count']}/{result['total_questions']})")
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="Quiz Grading",
                success=True,
                message=f"Tested {len(test_scenarios)} grading scenarios",
                duration=duration,
                data={'scenarios_tested': len(test_scenarios)}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Quiz Grading",
                success=False,
                message=f"Error: {str(e)}",
                duration=duration
            )
    
    def simulate_quiz_session(self) -> TestResult:
        """Simulate complete quiz session with database storage"""
        start_time = time.time()
        
        try:
            print("\n🎮 Simulating quiz session...")
            
            # Generate quiz
            questions_data = generate_quiz_for_study_plan(
                study_plan=self.test_plan,
                num_questions=5
            )
            
            if not questions_data:
                return TestResult(
                    test_name="Quiz Session Simulation",
                    success=False,
                    message="Failed to generate quiz questions",
                    duration=time.time() - start_time
                )
            
            # Create quiz session
            quiz_session = QuizSession.objects.create(
                study_plan=self.test_plan,
                user=self.test_user,
                total_questions=len(questions_data),
                difficulty='mixed',
                status='generated'
            )
            
            # Create questions
            for question_data in questions_data:
                QuizQuestion.objects.create(
                    quiz_session=quiz_session,
                    question_number=question_data['question_number'],
                    question_text=question_data['question_text'],
                    option_a=question_data['option_a'],
                    option_b=question_data['option_b'],
                    option_c=question_data['option_c'],
                    option_d=question_data['option_d'],
                    correct_answer=question_data['correct_answer'],
                    difficulty_level=question_data.get('difficulty_level', 'medium'),
                    source_topic=question_data.get('source_topic', 'General'),
                    source_file=question_data.get('source_file', ''),
                    explanation=question_data.get('explanation', ''),
                    explanation_wrong=question_data.get('explanation_wrong', {})
                )
            
            # Simulate user taking quiz
            quiz_session.mark_started()
            
            # Simulate answers with realistic timing
            user_answers = {}
            for question in quiz_session.questions.all():
                # Simulate realistic answer patterns
                if random.random() < 0.8:  # 80% chance of answering correctly
                    user_answer = question.correct_answer
                else:
                    # Random wrong answer
                    user_answer = random.choice(['a', 'b', 'c', 'd'])
                
                user_answers[question.question_number] = user_answer
                
                # Simulate time spent (10-60 seconds per question)
                time_spent = random.randint(10, 60)
                
                # Create answer record
                QuizAnswer.objects.create(
                    question=question,
                    user=self.test_user,
                    user_answer=user_answer,
                    is_correct=(user_answer == question.correct_answer),
                    time_spent_seconds=time_spent
                )
            
            # Complete quiz
            quiz_session.mark_completed()
            
            # Update user analytics
            analytics, created = UserAnalytics.objects.get_or_create(user=self.test_user)
            analytics.update_from_quiz(quiz_session)
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="Quiz Session Simulation",
                success=True,
                message=f"Simulated quiz session with {quiz_session.total_questions} questions, score: {quiz_session.score:.1f}%",
                duration=duration,
                data={
                    'quiz_id': quiz_session.id,
                    'score': quiz_session.score,
                    'time_spent': quiz_session.time_spent_seconds
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Quiz Session Simulation",
                success=False,
                message=f"Error: {str(e)}",
                duration=duration
            )
    
    def test_llm_integration(self) -> TestResult:
        """Test LLM integration and response parsing"""
        start_time = time.time()
        
        try:
            print("\n🤖 Testing LLM integration...")
            
            # Test single question generation
            test_content = "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data."
            
            question = self.agent._generate_single_question(
                content=test_content,
                topic="Machine Learning",
                source_file="test.pdf",
                difficulty="medium",
                source_type="test"
            )
            
            if not question:
                return TestResult(
                    test_name="LLM Integration",
                    success=False,
                    message="Failed to generate question from LLM",
                    duration=time.time() - start_time
                )
            
            # Validate LLM response structure
            validation = self._validate_questions([question])
            
            # Test error handling
            error_handling_tests = [
                {
                    'name': 'Empty Content',
                    'content': '',
                    'should_fail': True
                },
                {
                    'name': 'Very Long Content',
                    'content': 'A' * 10000,
                    'should_fail': False
                },
                {
                    'name': 'Non-English Content',
                    'content': 'El aprendizaje automático es un subconjunto de la inteligencia artificial.',
                    'should_fail': False
                }
            ]
            
            error_tests_passed = 0
            for test in error_handling_tests:
                try:
                    result = self.agent._generate_single_question(
                        content=test['content'],
                        topic="Test",
                        source_file="test.pdf",
                        difficulty="medium",
                        source_type="test"
                    )
                    
                    if test['should_fail']:
                        if result is None:
                            error_tests_passed += 1
                    else:
                        if result is not None:
                            error_tests_passed += 1
                            
                except Exception:
                    if test['should_fail']:
                        error_tests_passed += 1
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="LLM Integration",
                success=validation['valid'] and error_tests_passed == len(error_handling_tests),
                message=f"LLM integration {'working' if validation['valid'] else 'failed'}, error handling: {error_tests_passed}/{len(error_handling_tests)}",
                duration=duration,
                data={
                    'validation_issues': validation['issues'],
                    'error_tests_passed': error_tests_passed,
                    'total_error_tests': len(error_handling_tests)
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="LLM Integration",
                success=False,
                message=f"Error: {str(e)}",
                duration=duration
            )
    
    def benchmark_performance(self) -> TestResult:
        """Benchmark quiz system performance"""
        start_time = time.time()
        
        try:
            print("\n⚡ Running performance benchmarks...")
            
            # Benchmark question generation
            generation_times = []
            for i in range(5):
                start = time.time()
                questions = self.agent.generate_quiz(
                    study_plan_data=self.test_plan.plan_json,
                    uploaded_content=self._get_uploaded_content(),
                    num_questions=3,
                    difficulty_mix=True
                )
                generation_times.append(time.time() - start)
            
            avg_generation_time = sum(generation_times) / len(generation_times)
            
            # Benchmark grading
            test_questions = self.agent.generate_quiz(
                study_plan_data=self.test_plan.plan_json,
                uploaded_content=self._get_uploaded_content(),
                num_questions=5,
                difficulty_mix=True
            )
            
            grading_times = []
            for i in range(10):
                start = time.time()
                self.agent.grade_quiz(
                    test_questions,
                    {i+1: q['correct_answer'] for i, q in enumerate(test_questions)}
                )
                grading_times.append(time.time() - start)
            
            avg_grading_time = sum(grading_times) / len(grading_times)
            
            # Database operations benchmark
            db_times = []
            for i in range(5):
                start = time.time()
                QuizSession.objects.filter(user=self.test_user).count()
                db_times.append(time.time() - start)
            
            avg_db_time = sum(db_times) / len(db_times)
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="Performance Benchmark",
                success=True,
                message=f"Generation: {avg_generation_time:.3f}s, Grading: {avg_grading_time:.3f}s, DB: {avg_db_time:.3f}s",
                duration=duration,
                data={
                    'avg_generation_time': avg_generation_time,
                    'avg_grading_time': avg_grading_time,
                    'avg_db_time': avg_db_time
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Performance Benchmark",
                success=False,
                message=f"Error: {str(e)}",
                duration=duration
            )
    
    def check_database_integrity(self) -> TestResult:
        """Check database integrity for quiz data"""
        start_time = time.time()
        
        try:
            print("\n🗄️ Checking database integrity...")
            
            issues = []
            
            # Check for orphaned quiz questions
            orphaned_questions = QuizQuestion.objects.filter(
                quiz_session__isnull=True
            ).count()
            if orphaned_questions > 0:
                issues.append(f"{orphaned_questions} orphaned quiz questions")
            
            # Check for orphaned quiz answers
            orphaned_answers = QuizAnswer.objects.filter(
                question__isnull=True
            ).count()
            if orphaned_answers > 0:
                issues.append(f"{orphaned_answers} orphaned quiz answers")
            
            # Check for invalid question numbers
            invalid_numbers = QuizQuestion.objects.filter(
                question_number__lt=1
            ).count()
            if invalid_numbers > 0:
                issues.append(f"{invalid_numbers} questions with invalid numbers")
            
            # Check for duplicate question numbers in same quiz
            duplicate_numbers = 0
            for quiz in QuizSession.objects.all():
                numbers = list(quiz.questions.values_list('question_number', flat=True))
                if len(numbers) != len(set(numbers)):
                    duplicate_numbers += 1
            if duplicate_numbers > 0:
                issues.append(f"{duplicate_numbers} quizzes with duplicate question numbers")
            
            # Check for invalid correct answers
            invalid_answers = QuizQuestion.objects.exclude(
                correct_answer__in=['a', 'b', 'c', 'd']
            ).count()
            if invalid_answers > 0:
                issues.append(f"{invalid_answers} questions with invalid correct answers")
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="Database Integrity",
                success=len(issues) == 0,
                message=f"{'No issues found' if len(issues) == 0 else f'{len(issues)} issues found'}",
                duration=duration,
                data={'issues': issues}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Database Integrity",
                success=False,
                message=f"Error: {str(e)}",
                duration=duration
            )
    
    def reset_test_data(self) -> TestResult:
        """Clean up test data"""
        start_time = time.time()
        
        try:
            print("\n🧹 Cleaning up test data...")
            
            # Delete test quiz sessions and related data
            quiz_count = QuizSession.objects.filter(user=self.test_user).count()
            QuizSession.objects.filter(user=self.test_user).delete()
            
            # Delete test study plan
            if self.test_plan:
                self.test_plan.delete()
            
            # Delete test uploads and chunks
            upload_count = Upload.objects.filter(user=self.test_user).count()
            Upload.objects.filter(user=self.test_user).delete()
            
            # Delete test user analytics
            UserAnalytics.objects.filter(user=self.test_user).delete()
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name="Data Cleanup",
                success=True,
                message=f"Cleaned {quiz_count} quizzes, {upload_count} uploads",
                duration=duration,
                data={
                    'quizzes_deleted': quiz_count,
                    'uploads_deleted': upload_count
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name="Data Cleanup",
                success=False,
                message=f"Error: {str(e)}",
                duration=duration
            )
    
    def run_full_test_suite(self) -> List[TestResult]:
        """Run complete quiz system test suite"""
        print("🚀 Running full quiz system test suite...")
        print("=" * 60)
        
        self.test_results = []
        
        # Setup
        if not self.setup_test_environment():
            print("❌ Failed to setup test environment")
            return self.test_results
        
        # Run all tests
        tests = [
            self.test_quiz_generation,
            self.test_quiz_grading,
            self.simulate_quiz_session,
            self.test_llm_integration,
            self.benchmark_performance,
            self.check_database_integrity
        ]
        
        for test_func in tests:
            try:
                result = test_func()
                self.test_results.append(result)
                self._print_test_result(result)
            except Exception as e:
                error_result = TestResult(
                    test_name=test_func.__name__,
                    success=False,
                    message=f"Unexpected error: {str(e)}",
                    duration=0.0
                )
                self.test_results.append(error_result)
                self._print_test_result(error_result)
        
        # Summary
        self._print_test_summary()
        
        return self.test_results
    
    def _print_test_result(self, result: TestResult):
        """Print individual test result"""
        status = "✅" if result.success else "❌"
        print(f"\n{status} {result.test_name}")
        print(f"   {result.message}")
        print(f"   Duration: {result.duration:.3f}s")
    
    def _print_test_summary(self):
        """Print test suite summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.success)
        total_duration = sum(r.duration for r in self.test_results)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print(f"Total Duration: {total_duration:.3f}s")
        
        if total_tests - passed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.test_results:
                if not result.success:
                    print(f"   - {result.test_name}: {result.message}")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1]
    debugger = QuizDebugger()
    
    if command == "test-generation":
        debugger.setup_test_environment()
        result = debugger.test_quiz_generation()
        debugger._print_test_result(result)
        
    elif command == "test-grading":
        debugger.setup_test_environment()
        result = debugger.test_quiz_grading()
        debugger._print_test_result(result)
        
    elif command == "simulate-session":
        debugger.setup_test_environment()
        result = debugger.simulate_quiz_session()
        debugger._print_test_result(result)
        
    elif command == "validate-questions":
        debugger.setup_test_environment()
        result = debugger.test_quiz_generation()
        debugger._print_test_result(result)
        
    elif command == "test-llm-integration":
        debugger.setup_test_environment()
        result = debugger.test_llm_integration()
        debugger._print_test_result(result)
        
    elif command == "benchmark":
        debugger.setup_test_environment()
        result = debugger.benchmark_performance()
        debugger._print_test_result(result)
        
    elif command == "check-db":
        result = debugger.check_database_integrity()
        debugger._print_test_result(result)
        
    elif command == "reset-test-data":
        result = debugger.reset_test_data()
        debugger._print_test_result(result)
        
    elif command == "full-test":
        debugger.run_full_test_suite()
        
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
