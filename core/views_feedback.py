# core/views_feedback.py - ปรับให้ใช้ ?feedback=success แทน messages

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.utils import timezone
from django.urls import reverse

import json
from typing import Dict, Optional

from .models import (
    StudyPlanHistory, PrioritizationFeedback, 
    ScoringModelAdjustment, UserAnalytics
)
from .feedback_system import trigger_reinforcement_learning


# ==================== AUTO PLAN DETECTION ====================

def get_latest_plan(user):
    """Helper function to get user's latest study plan"""
    return StudyPlanHistory.objects.filter(
        user=user,
        status='active'
    ).order_by('-created_at').first()


# ==================== QUICK FEEDBACK (AUTO) ====================

@login_required
@require_http_methods(["POST"])
def quick_feedback_auto(request):
    """
    Quick feedback with automatic plan detection.
    """
    try:
        study_plan = get_latest_plan(request.user)
        
        if not study_plan:
            return redirect('upload_page_optimized')
        
        thumbs_up = request.POST.get('thumbs_up', '').lower() == 'true'
        
        # Create feedback
        feedback = PrioritizationFeedback.objects.create(
            user=request.user,
            study_plan=study_plan,
            feedback_type='overall',
            rating_type='thumbs',
            thumbs_up=thumbs_up,
            star_rating=5 if thumbs_up else 1,
            feedback_text=f"Quick feedback: {'Positive' if thumbs_up else 'Negative'}"
        )
        
        # Update analytics
        analytics, _ = UserAnalytics.objects.get_or_create(user=request.user)
        analytics.update_from_feedback(feedback)
        
        # Trigger RL if needed
        unprocessed_count = PrioritizationFeedback.objects.filter(processed=False).count()
        if unprocessed_count >= 10:
            print(f"Triggering RL (unprocessed: {unprocessed_count})")
            trigger_reinforcement_learning()
        
        # Redirect ด้วย ?feedback=success
        redirect_url = f"{reverse('history_detail', args=[study_plan.id])}?feedback=success"
        return redirect(redirect_url)
        
    except Exception as e:
        print(f"Error submitting quick feedback: {e}")
        return redirect('result_page_optimized')


# ==================== QUICK FEEDBACK (WITH PLAN ID) ====================

@login_required
@require_http_methods(["POST"])
def quick_feedback(request, plan_id):
    """
    Quick thumbs up/down feedback for a specific plan.
    """
    try:
        study_plan = get_object_or_404(
            StudyPlanHistory,
            id=plan_id,
            user=request.user
        )
        
        thumbs_up = request.POST.get('thumbs_up', '').lower() == 'true'
        
        feedback = PrioritizationFeedback.objects.create(
            user=request.user,
            study_plan=study_plan,
            feedback_type='overall',
            rating_type='thumbs',
            thumbs_up=thumbs_up,
            star_rating=5 if thumbs_up else 1,
            feedback_text=f"Quick feedback: {'Positive' if thumbs_up else 'Negative'}"
        )
        
        analytics, _ = UserAnalytics.objects.get_or_create(user=request.user)
        analytics.update_from_feedback(feedback)
        
        unprocessed_count = PrioritizationFeedback.objects.filter(processed=False).count()
        if unprocessed_count >= 10:
            print(f"Triggering RL (unprocessed: {unprocessed_count})")
            trigger_reinforcement_learning()
        
        # ใช้ ?feedback=success
        redirect_url = f"{reverse('history_detail', args=[plan_id])}?feedback=success"
        return redirect(redirect_url)
        
    except Exception as e:
        print(f"Error submitting quick feedback: {e}")
        return redirect('result_page_optimized')


# ==================== DETAILED FEEDBACK PAGE (AUTO) ====================

@login_required
def detailed_feedback_auto(request):
    """
    Redirect to detailed feedback with latest plan.
    """
    try:
        study_plan = get_latest_plan(request.user)
        if not study_plan:
            return redirect('upload_page_optimized')
        return redirect('detailed_feedback_page', plan_id=study_plan.id)
    except Exception as e:
        print(f"Error: {e}")
        return redirect('result_page_optimized')


# ==================== DETAILED FEEDBACK PAGE ====================

@login_required
def detailed_feedback_page(request, plan_id):
    """
    Detailed feedback form + submission.
    """
    try:
        study_plan = get_object_or_404(
            StudyPlanHistory,
            id=plan_id,
            user=request.user
        )
        
        if request.method == 'POST':
            star_rating = int(request.POST.get('star_rating', 3))
            aspect_urgency = int(request.POST.get('aspect_urgency', 3))
            aspect_schedule = int(request.POST.get('aspect_schedule', 3))
            aspect_complexity = int(request.POST.get('aspect_complexity', 3))
            aspect_relevance = int(request.POST.get('aspect_relevance', 3))
            feedback_text = request.POST.get('feedback_text', '')
            
            feedback = PrioritizationFeedback.objects.create(
                user=request.user,
                study_plan=study_plan,
                feedback_type='overall',
                rating_type='detailed',
                star_rating=star_rating,
                feedback_text=feedback_text,
                aspects={
                    'urgency': aspect_urgency,
                    'complexity': aspect_complexity,
                    'relevance': aspect_relevance,
                    'schedule_quality': aspect_schedule
                }
            )
            
            analytics, _ = UserAnalytics.objects.get_or_create(user=request.user)
            analytics.update_from_feedback(feedback)
            
            unprocessed_count = PrioritizationFeedback.objects.filter(processed=False).count()
            if unprocessed_count >= 10:
                trigger_reinforcement_learning()
            
            # ใช้ ?feedback=success
            redirect_url = f"{reverse('history_detail', args=[plan_id])}?feedback=success"
            return redirect(redirect_url)
        
        context = {'study_plan': study_plan}
        return render(request, 'core/feedback_detailed.html', context)
        
    except Exception as e:
        print(f"Error: {e}")
        return redirect('result_page_optimized')


# ==================== SUBMIT FEEDBACK (AJAX) ====================

@login_required
@require_http_methods(["POST"])
def submit_feedback(request):
    """
    AJAX feedback submission.
    """
    try:
        data = json.loads(request.body)
        plan_id = data.get('plan_id')
        
        if plan_id:
            study_plan = get_object_or_404(StudyPlanHistory, id=plan_id, user=request.user)
        else:
            study_plan = get_latest_plan(request.user)
            if not study_plan:
                return JsonResponse({'success': False, 'error': 'No plan found'}, status=400)
        
        feedback = PrioritizationFeedback.objects.create(
            user=request.user,
            study_plan=study_plan,
            feedback_type=data.get('feedback_type', 'overall'),
            rating_type=data.get('rating_type', 'detailed'),
            star_rating=data.get('star_rating', 0),
            feedback_text=data.get('feedback_text', ''),
            aspects=data.get('aspects', {})
        )
        
        analytics, _ = UserAnalytics.objects.get_or_create(user=request.user)
        analytics.update_from_feedback(feedback)
        
        unprocessed_count = PrioritizationFeedback.objects.filter(processed=False).count()
        if unprocessed_count >= 10:
            trigger_reinforcement_learning()
        
        # ส่ง URL กลับไปให้ frontend redirect
        redirect_url = f"{reverse('history_detail', args=[study_plan.id])}?feedback=success"
        return JsonResponse({
            'success': True,
            'message': 'Feedback submitted',
            'redirect_url': redirect_url
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        print(f"Error: {e}")
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


# ==================== FEEDBACK HISTORY ====================

@login_required
def feedback_history(request):
    feedbacks = PrioritizationFeedback.objects.filter(
        user=request.user
    ).select_related('study_plan').order_by('-created_at')[:50]
    
    return render(request, 'core/feedback_history.html', {'feedbacks': feedbacks})


# ==================== ADMIN: TRIGGER RL ====================

@login_required
def trigger_rl_adjustment(request):
    if not request.user.is_staff:
        return redirect('result_page_optimized')
    
    if request.method == 'POST':
        try:
            result = trigger_reinforcement_learning()
            if result.get('success'):
                messages.success(request, f"RL triggered! Processed {result.get('feedback_count', 0)} items.")
            else:
                messages.warning(request, result.get('message', 'No feedback'))
            return redirect('view_adjustments')
        except Exception as e:
            messages.error(request, f"Error: {e}")
    
    unprocessed_count = PrioritizationFeedback.objects.filter(processed=False).count()
    return render(request, 'core/admin_trigger_rl.html', {'unprocessed_count': unprocessed_count})


# ==================== ADMIN: VIEW ADJUSTMENTS ====================

@login_required
def view_adjustments(request):
    if not request.user.is_staff:
        return redirect('result_page_optimized')
    
    adjustments = ScoringModelAdjustment.objects.all().order_by('-created_at')[:50]
    recent_feedback = PrioritizationFeedback.objects.order_by('-created_at')[:20]
    
    from django.db.models import Avg
    avg_rating = PrioritizationFeedback.objects.aggregate(Avg('star_rating'))
    
    context = {
        'adjustments': adjustments,
        'recent_feedback': recent_feedback,
        'feedback_stats': {
            'total': PrioritizationFeedback.objects.count(),
            'processed': PrioritizationFeedback.objects.filter(processed=True).count(),
            'unprocessed': PrioritizationFeedback.objects.filter(processed=False).count(),
            'avg_star_rating': round(avg_rating['star_rating__avg'] or 0, 2),
        }
    }
    return render(request, 'core/admin_adjustments.html', context)