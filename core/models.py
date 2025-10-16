# core/models.py - Enhanced with History Tracking

from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone

User = get_user_model()

class Upload(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='uploads/%Y/%m/%d/')
    filename = models.CharField(max_length=512)
    pages = models.IntegerField(null=True, blank=True)
    status = models.CharField(max_length=32, default='uploaded')
    created_at = models.DateTimeField(auto_now_add=True)
    
    # OCR tracking fields
    ocr_pages = models.IntegerField(default=0, help_text="Number of pages processed with OCR")
    ocr_used = models.BooleanField(default=False, help_text="Whether OCR was used for this upload")
    
    def __str__(self):
        return f"{self.filename} ({self.status})"
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'status']),
            models.Index(fields=['created_at']),
        ]


class Chunk(models.Model):
    upload = models.ForeignKey(Upload, on_delete=models.CASCADE, related_name='chunks')
    chunk_id = models.CharField(max_length=64, unique=True)
    text = models.TextField()
    start_page = models.IntegerField()
    end_page = models.IntegerField()
    embedding_id = models.CharField(max_length=128, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Chunk {self.chunk_id} (pages {self.start_page}-{self.end_page})"
    
    class Meta:
        ordering = ['start_page']
        indexes = [
            models.Index(fields=['upload', 'start_page']),
        ]


class StudyPlanHistory(models.Model):
    """
    Complete history tracking for all study plan generations.
    Stores all user inputs and outputs for analytics and retrieval.
    """
    # User & Timestamp
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='study_plans')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # User Inputs
    user_goal = models.TextField(blank=True, null=True, help_text="User's stated goal")
    sort_method = models.CharField(max_length=50, default='hybrid', 
                                   help_text="Sorting method: hybrid/urgency/foundational/pages/content/complexity")
    constraints = models.TextField(blank=True, null=True, help_text="User preferences and constraints")
    
    # Time Availability (stored as JSON for flexibility)
    time_input = models.JSONField(default=dict, help_text="Time availability: {years, months, weeks, days, hours}")
    total_hours = models.FloatField(default=0, help_text="Total hours calculated from time_input")
    
    # Uploaded Materials
    uploads = models.ManyToManyField(Upload, related_name='study_plans', 
                                    help_text="PDFs used in this plan")
    total_files = models.IntegerField(default=0)
    total_pages = models.IntegerField(default=0)
    total_chunks = models.IntegerField(default=0)
    
    # Generated Plan (stored as JSON)
    plan_json = models.JSONField(default=dict, help_text="Complete generated plan with tasks and schedule")
    
    # Execution Metrics
    execution_time = models.FloatField(default=0, help_text="Time taken to generate plan (seconds)")
    tool_calls = models.IntegerField(default=0, help_text="Number of AI tool calls made")
    
    # OCR Usage
    ocr_pages_total = models.IntegerField(default=0, help_text="Total OCR pages across all uploads")
    
    # Status
    status = models.CharField(max_length=20, default='active', 
                             choices=[('active', 'Active'), ('archived', 'Archived'), ('deleted', 'Deleted')])
    
    def __str__(self):
        return f"{self.user.username} - {self.created_at.strftime('%Y-%m-%d %H:%M')} ({self.total_files} files)"
    
    def get_schedule(self):
        """Extract schedule from plan_json"""
        for item in self.plan_json:
            if item.get('file') == '📅 WEEKLY SCHEDULE':
                return item.get('schedule', [])
        return []
    
    def get_tasks(self):
        """Extract prioritized tasks from plan_json"""
        return [
            item for item in self.plan_json 
            if item.get('file') != '📅 WEEKLY SCHEDULE'
        ]
    
    def get_summary(self):
        """Get quick summary for display"""
        return {
            'files': self.total_files,
            'pages': self.total_pages,
            'sort_method': self.sort_method,
            'has_schedule': len(self.get_schedule()) > 0,
            'task_count': len(self.get_tasks()),
            'created': self.created_at,
        }
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Study Plan History"
        verbose_name_plural = "Study Plan Histories"
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['status']),
            models.Index(fields=['sort_method']),
        ]


class Plan(models.Model):
    """
    Legacy model - kept for backwards compatibility.
    New implementations should use StudyPlanHistory.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    upload = models.ForeignKey(Upload, on_delete=models.CASCADE)
    plan_json = models.JSONField()
    score = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    version = models.IntegerField(default=1)
    
    def __str__(self):
        return f"Plan for {self.user.username} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    class Meta:
        ordering = ['-created_at']