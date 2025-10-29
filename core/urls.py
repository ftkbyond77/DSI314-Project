# core/urls.py - Complete Working URLs

from django.urls import path
from . import views_optimized

# Import quiz and feedback views
from . import views_quiz
from . import views_feedback

urlpatterns = [
    # ==================== MAIN ROUTES ====================
    path('', views_optimized.upload_page_optimized, name='upload_page_optimized'),
    path('upload/', views_optimized.upload_page_optimized, name='upload_page_optimized'),
    path('result/', views_optimized.result_page_optimized, name='result_page_optimized'),
    
    # ==================== HISTORY ====================
    path('history/<int:history_id>/', views_optimized.history_detail, name='history_detail'),
    
    # ==================== ASYNC PROGRESS ====================
    path('planning-progress/', views_optimized.planning_progress, name='planning_progress'),
    path('api/planning-status/<str:task_id>/', views_optimized.planning_status_api, name='planning_status_api'),
    
    # ==================== ADMIN ====================
    path('admin/agent-logs/', views_optimized.agent_logs, name='agent_logs'),
    
    # ==================== AUTH ====================
    path('register/', views_optimized.register_page, name='register_page'),
    path('login/', views_optimized.login_page, name='login_page'),
    path('logout/', views_optimized.logout_view, name='logout_view'),
    
    # ==================== QUIZ SYSTEM ====================
    # Auto-detection route (finds latest plan automatically)
    path('generate-quiz-auto/', views_quiz.generate_quiz_auto, name='generate_quiz_auto'),
    
    # Regular routes
    path('generate-quiz/<int:plan_id>/', views_quiz.generate_quiz, name='generate_quiz'),
    path('quiz/<int:quiz_id>/', views_quiz.quiz_test, name='quiz_test'),
    path('quiz/<int:quiz_id>/submit-answer/', views_quiz.submit_quiz_answer, name='submit_quiz_answer'),
    path('quiz/<int:quiz_id>/submit/', views_quiz.submit_quiz, name='submit_quiz'),
    path('quiz/<int:quiz_id>/results/', views_quiz.quiz_results, name='quiz_results'),
    path('quiz-history/', views_quiz.quiz_history, name='quiz_history'),
    
    # ==================== FEEDBACK SYSTEM ====================
    # Auto-detection routes (finds latest plan automatically)
    path('feedback/quick-auto/', views_feedback.quick_feedback_auto, name='quick_feedback_auto'),
    path('feedback/detailed-auto/', views_feedback.detailed_feedback_auto, name='detailed_feedback_auto'),
    
    # Regular routes
    path('feedback/submit/', views_feedback.submit_feedback, name='submit_feedback'),
    path('feedback/quick/<int:plan_id>/', views_feedback.quick_feedback, name='quick_feedback'),
    path('feedback/detailed/<int:plan_id>/', views_feedback.detailed_feedback_page, name='detailed_feedback_page'),
    path('feedback/history/', views_feedback.feedback_history, name='feedback_history'),
    
    # Admin routes
    path('admin/trigger-rl/', views_feedback.trigger_rl_adjustment, name='trigger_rl_adjustment'),
    path('admin/view-adjustments/', views_feedback.view_adjustments, name='view_adjustments'),
]