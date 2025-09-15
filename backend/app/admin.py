# backend/app/admin.py
from django.contrib import admin
from .models import FileUpload, UserProgress, Quiz

# Register your models here
admin.site.register(FileUpload)
admin.site.register(UserProgress)
admin.site.register(Quiz)
