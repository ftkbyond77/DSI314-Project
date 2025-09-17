from django.db import models
from django.contrib.auth.models import User

class FileUpload(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file_name = models.CharField(max_length=255)
    file_content = models.TextField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

class UserProgress(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    last_read_file = models.ForeignKey(FileUpload, null=True, blank=True, on_delete=models.SET_NULL)
    progress_percent = models.FloatField(default=0.0)

class Quiz(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    question = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    score = models.FloatField(default=0.0)
