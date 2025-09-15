from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_files, name='upload_files'),
    path('plan/', views.view_plan, name='view_plan'),
    path('quiz/<int:file_id>/', views.generate_quiz, name='generate_quiz'),
]