from django.db import models


class UploadedDocument(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    original_name = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploads/%Y/%m/%d/')
    text_excerpt = models.TextField(blank=True)

    def __str__(self) -> str:
        return self.original_name


class ProcessingRun(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    user_prompt = models.TextField(blank=True)
    documents = models.ManyToManyField(UploadedDocument, related_name='runs')
    ranking_json = models.JSONField(default=list)
    summary_json = models.JSONField(default=list)
    quiz_json = models.JSONField(default=list)

    def __str__(self) -> str:
        return f"Run {self.id} at {self.created_at}"

