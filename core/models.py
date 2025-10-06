from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()

class Upload(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='uploads/%Y/%m/%d/')
    filename = models.CharField(max_length=512)
    pages = models.IntegerField(null=True, blank=True)
    status = models.CharField(max_length=32, default='uploaded')
    created_at = models.DateTimeField(auto_now_add=True)

class Chunk(models.Model):
    upload = models.ForeignKey(Upload, on_delete=models.CASCADE)
    chunk_id = models.CharField(max_length=64)
    text = models.TextField()
    start_page = models.IntegerField()
    end_page = models.IntegerField()
    embedding_id = models.CharField(max_length=128, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class Plan(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    upload = models.ForeignKey(Upload, on_delete=models.CASCADE)
    plan_json = models.JSONField()
    score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
