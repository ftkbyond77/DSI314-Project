# core/views_quiz.py - Enhanced with Auto Plan Detection

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.http import require_http_methods
from django.contrib import messages

import json
from typing import Dict

from .models import (
    StudyPlanHistory, QuizSession, QuizQuestion, 
    QuizAnswer, UserAnalytics
)
from .quiz_agent import generate_quiz_for_study_plan, grade_user_quiz


# ==================== AUTO PLAN DETECTION ====================

@login_required
@require_http_methods(["POST"])
def generate_quiz_auto(request):
    try:
        latest_plan = StudyPlanHistory.objects.filter(
            user=request.user,
            status='active'
        ).order_by('-created_at').first()
        
        if not latest_plan:
            messages.error(request, "ไม่พบแผนการเรียน กรุณาสร้างก่อน")
            return redirect('upload_page_optimized')
        
        request.session['latest_plan_id'] = latest_plan.id

        return redirect('generate_quiz', plan_id=latest_plan.id)

    except Exception as e:
        messages.error(request, "เกิดข้อผิดพลาด")
        return redirect('upload_page_optimized')

# ==================== QUIZ GENERATION ====================


# core/views_quiz.py

@login_required
@require_http_methods(["GET", "POST"])  
def generate_quiz(request, plan_id):

    try:
        study_plan = get_object_or_404(
            StudyPlanHistory,
            id=plan_id,
            user=request.user
        )
        
        request.session['latest_plan_id'] = plan_id

        existing_quiz = QuizSession.objects.filter(
            study_plan=study_plan,
            user=request.user,
            status='generated'
        ).first()

        if existing_quiz:
            return redirect('quiz_test', quiz_id=existing_quiz.id)

        questions_data = generate_quiz_for_study_plan(
            study_plan=study_plan,
            num_questions=5
        )

        if not questions_data:
            messages.error(request, "ไม่สามารถสร้างคำถามได้ ลองใหม่")
            return redirect('result_page_optimized')

        quiz_session = QuizSession.objects.create(
            study_plan=study_plan,
            user=request.user,
            total_questions=len(questions_data),
            difficulty='mixed',
            status='generated'
        )

        for q in questions_data:
            QuizQuestion.objects.create(quiz_session=quiz_session, **q)

        study_plan.quiz_generated = True
        study_plan.save(update_fields=['quiz_generated'])

        print(f"Quiz {quiz_session.id} created via {'GET' if request.method == 'GET' else 'POST'}")

        return redirect('quiz_test', quiz_id=quiz_session.id)

    except Exception as e:
        print(f"Error in generate_quiz: {e}")
        messages.error(request, "เกิดข้อผิดพลาดในการสร้างแบบทดสอบ")
        return redirect('result_page_optimized')


# ==================== QUIZ TEST PAGE ====================

@login_required
def quiz_test(request, quiz_id):
    """
    Display the quiz test page (Microsoft Form style).
    Shows one question at a time.
    """
    
    try:
        # Get quiz session
        quiz_session = get_object_or_404(
            QuizSession,
            id=quiz_id,
            user=request.user
        )
        
        # Mark as started if not already
        if not quiz_session.started_at:
            quiz_session.mark_started()
        
        # Get all questions
        questions = quiz_session.questions.all().order_by('question_number')
        
        # Get current question from query param (default to 1)
        current_question_num = int(request.GET.get('q', 1))
        
        # Ensure valid question number
        if current_question_num < 1:
            current_question_num = 1
        elif current_question_num > questions.count():
            current_question_num = questions.count()
        
        # Get current question
        current_question = questions.filter(question_number=current_question_num).first()
        
        if not current_question:
            messages.error(request, "Question not found.")
            return redirect('result_page_optimized')
        
        # Get existing answer if any
        existing_answer = QuizAnswer.objects.filter(
            question=current_question,
            user=request.user
        ).first()
        
        context = {
            'quiz_session': quiz_session,
            'current_question': current_question,
            'current_num': current_question_num,
            'total_questions': questions.count(),
            'existing_answer': existing_answer.user_answer if existing_answer else None,
            'is_first': current_question_num == 1,
            'is_last': current_question_num == questions.count(),
            'progress_percentage': (current_question_num / questions.count()) * 100,
        }
        
        return render(request, 'core/quiz_test.html', context)
        
    except Exception as e:
        print(f"❌ Error loading quiz: {e}")
        messages.error(request, f"Error loading quiz: {str(e)}")
        return redirect('result_page_optimized')


# ==================== QUIZ ANSWER SUBMISSION ====================

@login_required
@require_http_methods(["POST"])
def submit_quiz_answer(request, quiz_id):
    """
    Submit answer for a single question.
    AJAX endpoint for real-time answer submission.
    """
    
    try:
        data = json.loads(request.body)
        question_number = data.get('question_number')
        user_answer = data.get('answer', '').lower()
        time_spent = data.get('time_spent', 0)
        
        # Get quiz and question
        quiz_session = get_object_or_404(QuizSession, id=quiz_id, user=request.user)
        question = get_object_or_404(
            QuizQuestion,
            quiz_session=quiz_session,
            question_number=question_number
        )
        
        # Check if answer exists
        answer_obj, created = QuizAnswer.objects.update_or_create(
            question=question,
            user=request.user,
            defaults={
                'user_answer': user_answer,
                'is_correct': (user_answer == question.correct_answer),
                'time_spent_seconds': time_spent
            }
        )
        
        return JsonResponse({
            'success': True,
            'is_correct': answer_obj.is_correct,
            'message': 'Answer saved successfully'
        })
        
    except Exception as e:
        print(f"❌ Error submitting answer: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)


# ==================== QUIZ COMPLETION ====================

@login_required
@require_http_methods(["POST"])
def submit_quiz(request, quiz_id):
    """
    Complete the quiz and calculate final score.
    Redirects to results page.
    """
    
    try:
        # Get quiz session
        quiz_session = get_object_or_404(
            QuizSession,
            id=quiz_id,
            user=request.user
        )
        
        # Get all questions and answers
        questions = quiz_session.questions.all()
        user_answers = {}
        
        for question in questions:
            answer = QuizAnswer.objects.filter(
                question=question,
                user=request.user
            ).first()
            
            if answer:
                user_answers[question.question_number] = answer.user_answer
        
        # Count correct answers
        correct_count = QuizAnswer.objects.filter(
            question__quiz_session=quiz_session,
            user=request.user,
            is_correct=True
        ).count()
        
        # Update quiz session
        quiz_session.correct_answers = correct_count
        quiz_session.mark_completed()
        
        # Update user analytics
        analytics, created = UserAnalytics.objects.get_or_create(user=request.user)
        analytics.update_from_quiz(quiz_session)
        
        print(f"✅ Quiz {quiz_id} completed: {correct_count}/{questions.count()} correct")
        
        # Redirect to results
        return redirect('quiz_results', quiz_id=quiz_id)
        
    except Exception as e:
        print(f"❌ Error submitting quiz: {e}")
        messages.error(request, f"Error submitting quiz: {str(e)}")
        return redirect('quiz_test', quiz_id=quiz_id)


# ==================== QUIZ RESULTS PAGE ====================

@login_required
def quiz_results(request, quiz_id):
    """
    Display quiz results with score, explanations, and analysis.
    """
    
    try:
        # Get quiz session
        quiz_session = get_object_or_404(
            QuizSession,
            id=quiz_id,
            user=request.user
        )
        
        # Get all questions with user answers
        questions = quiz_session.questions.all().order_by('question_number')
        
        results_data = []
        performance_by_difficulty = {
            'easy': {'correct': 0, 'total': 0},
            'medium': {'correct': 0, 'total': 0},
            'hard': {'correct': 0, 'total': 0}
        }
        performance_by_topic = {}
        
        for question in questions:
            # Get user answer
            user_answer_obj = QuizAnswer.objects.filter(
                question=question,
                user=request.user
            ).first()
            
            user_answer = user_answer_obj.user_answer if user_answer_obj else None
            is_correct = user_answer_obj.is_correct if user_answer_obj else False
            time_spent = user_answer_obj.time_spent_seconds if user_answer_obj else 0
            
            # Track performance
            difficulty = question.difficulty_level
            topic = question.source_topic
            
            performance_by_difficulty[difficulty]['total'] += 1
            if is_correct:
                performance_by_difficulty[difficulty]['correct'] += 1
            
            if topic not in performance_by_topic:
                performance_by_topic[topic] = {'correct': 0, 'total': 0}
            performance_by_topic[topic]['total'] += 1
            if is_correct:
                performance_by_topic[topic]['correct'] += 1
            
            # Get explanation for wrong answer
            why_wrong = ""
            if not is_correct and user_answer:
                why_wrong = question.explanation_wrong.get(user_answer, "")
            
            results_data.append({
                'question': question,
                'user_answer': user_answer,
                'is_correct': is_correct,
                'time_spent': time_spent,
                'why_wrong': why_wrong,
                'options': {
                    'a': question.option_a,
                    'b': question.option_b,
                    'c': question.option_c,
                    'd': question.option_d
                }
            })
        
        # Generate feedback
        score_pct = quiz_session.score or 0
        if score_pct >= 90:
            overall_feedback = "🎉 Outstanding! You demonstrated excellent mastery of the material."
        elif score_pct >= 75:
            overall_feedback = "👏 Great job! You have a strong understanding of most concepts."
        elif score_pct >= 60:
            overall_feedback = "👍 Good effort! You understand the basics but could benefit from review."
        else:
            overall_feedback = "💪 Keep studying! Focus on understanding the core concepts better."
        
        # Find weak areas
        weak_topics = [
            topic for topic, stats in performance_by_topic.items()
            if stats['total'] > 0 and (stats['correct'] / stats['total']) < 0.6
        ]
        
        context = {
            'quiz_session': quiz_session,
            'results_data': results_data,
            'performance_by_difficulty': performance_by_difficulty,
            'performance_by_topic': performance_by_topic,
            'overall_feedback': overall_feedback,
            'weak_topics': weak_topics,
            'study_plan': quiz_session.study_plan,
        }
        
        return render(request, 'core/result_quiz.html', context)
        
    except Exception as e:
        print(f"❌ Error loading quiz results: {e}")
        messages.error(request, f"Error loading results: {str(e)}")
        return redirect('result_page_optimized')


# ==================== QUIZ HISTORY ====================

@login_required
def quiz_history(request):
    """
    Display user's quiz history with scores and dates.
    """
    
    quizzes = QuizSession.objects.filter(
        user=request.user,
        status='completed'
    ).select_related('study_plan').order_by('-completed_at')
    
    context = {
        'quizzes': quizzes
    }
    
    return render(request, 'core/quiz_history.html', context)