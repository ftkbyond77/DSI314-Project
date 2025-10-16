from django.urls import path
from . import views_optimized

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
]