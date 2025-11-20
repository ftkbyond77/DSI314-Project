# core/views_quiz.py

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.utils import timezone

from .models import StudyPlanHistory, QuizSession, QuizQuestion, QuizAnswer
from .quiz_agent import generate_quiz_for_study_plan

@login_required
@require_http_methods(["POST"])
def generate_quiz_specific_auto(request):
    """
    Auto-detects latest plan and generates a quiz for a SPECIFIC file/task.
    """
    try:
        task_name = request.POST.get('task_name')
        
        # Normalize task name: treat empty or "General Quiz" as None (All files)
        if not task_name or task_name.strip() == "" or task_name == 'General Quiz':
            task_name = None 

        # Find latest active plan
        latest_plan = StudyPlanHistory.objects.filter(
            user=request.user,
            status='active'
        ).order_by('-created_at').first()
        
        if not latest_plan:
            messages.error(request, "No active study plan found. Please upload files first.")
            return redirect('upload_page_optimized')

        # Define a search name for database lookup
        search_name = task_name if task_name else "General Quiz"
        
        # Check for existing incomplete quiz
        existing_quiz = QuizSession.objects.filter(
            study_plan=latest_plan,
            user=request.user,
            focus_task_name=search_name,
            status__in=['generated', 'in_progress']
        ).first()

        if existing_quiz:
            return redirect('quiz_test', quiz_id=existing_quiz.id)

        # Generate new quiz
        questions_data = generate_quiz_for_study_plan(
            study_plan=latest_plan,
            num_questions=10,
            focus_task_name=task_name
        )

        if not questions_data:
            messages.error(request, "Could not generate quiz questions. No content available.")
            return redirect('result_page_optimized')

        # Create Session
        quiz_session = QuizSession.objects.create(
            study_plan=latest_plan,
            user=request.user,
            total_questions=len(questions_data),
            status='generated',
            focus_task_name=search_name
        )

        # Create Questions
        for q in questions_data:
            QuizQuestion.objects.create(quiz_session=quiz_session, **q)

        return redirect('quiz_test', quiz_id=quiz_session.id)

    except Exception as e:
        print(f"Error generating quiz: {e}")
        messages.error(request, f"An error occurred: {str(e)}")
        return redirect('result_page_optimized')


@login_required
def quiz_test(request, quiz_id):
    quiz_session = get_object_or_404(QuizSession, id=quiz_id, user=request.user)
    
    if quiz_session.status == 'completed':
        return redirect('quiz_results', quiz_id=quiz_session.id)
    
    if not quiz_session.started_at:
        quiz_session.started_at = timezone.now()
        quiz_session.status = 'in_progress'
        quiz_session.save()
        
    questions = quiz_session.questions.all().order_by('question_number')
    
    return render(request, 'core/quiz_test.html', {
        'quiz': quiz_session,
        'questions': questions,
        'task_name': quiz_session.focus_task_name or "General Quiz"
    })


@login_required
@require_http_methods(["POST"])
def submit_quiz(request, quiz_id):
    quiz_session = get_object_or_404(QuizSession, id=quiz_id, user=request.user)
    questions = quiz_session.questions.all()
    
    correct_count = 0
    
    for question in questions:
        user_choice = request.POST.get(str(question.id))
        
        if user_choice:
            is_correct = (user_choice.lower() == question.correct_answer.lower())
            if is_correct:
                correct_count += 1
                
            QuizAnswer.objects.create(
                question=question,
                user=request.user,
                user_answer=user_choice,
                is_correct=is_correct
            )
            
    quiz_session.status = 'completed'
    quiz_session.completed_at = timezone.now()
    quiz_session.correct_answers = correct_count
    quiz_session.score = (correct_count / quiz_session.total_questions) * 100 if quiz_session.total_questions > 0 else 0
    quiz_session.save()
    
    return redirect('quiz_results', quiz_id=quiz_session.id)


@login_required
def quiz_results(request, quiz_id):
    quiz_session = get_object_or_404(QuizSession, id=quiz_id, user=request.user)
    questions = quiz_session.questions.all().order_by('question_number')
    
    results_data = []
    for q in questions:
        answer = QuizAnswer.objects.filter(question=q, user=request.user).first()
        results_data.append({
            'question': q,
            'user_answer': answer.user_answer if answer else None,
            'is_correct': answer.is_correct if answer else False,
            'options': {
                'a': q.option_a,
                'b': q.option_b,
                'c': q.option_c,
                'd': q.option_d
            }
        })
        
    return render(request, 'core/result_quiz.html', {
        'quiz': quiz_session,
        'results': results_data,
        'score': quiz_session.correct_answers,
        'total': quiz_session.total_questions,
        'task_name': quiz_session.focus_task_name or "General Quiz"
    })


@login_required
def quiz_history(request):
    quizzes = QuizSession.objects.filter(
        user=request.user
    ).order_by('-created_at')
    
    return render(request, 'core/quiz_history.html', {
        'quizzes': quizzes
    })