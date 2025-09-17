from django.urls import path
from . import views

app_name = 'assistant'

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_files, name='upload'),
    path('run/', views.run_pipeline, name='run'),
]

