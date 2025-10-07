# core/urls.py
from django.urls import path
# from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from . import views  

urlpatterns = [
    path('', views.upload_page, name='upload_page'),  # root path
    path('upload/', views.upload_page, name='upload_page'),
    path('result/', views.result_page, name='result_page'),
    path('register/', views.register_page, name='register_page'),
    path('login/', views.login_page, name='login_page'),
    path('logout/', views.logout_view, name='logout_view'),

]
