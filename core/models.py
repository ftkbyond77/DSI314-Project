# core/models.py - Enhanced with Quiz, Feedback, and Analytics Support

from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator

User = get_user_model()

# ==================== EXISTING MODELS (PRESERVED) ====================

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
    
    # ==================== ANALYTICS ENHANCEMENTS ====================
    # User behavior tracking
    viewed_count = models.IntegerField(default=0, help_text="Number of times user viewed this plan")
    last_viewed_at = models.DateTimeField(null=True, blank=True, help_text="Last time user viewed this plan")
    
    # Engagement metrics
    time_spent_seconds = models.IntegerField(default=0, help_text="Total time user spent on this plan")
    quiz_generated = models.BooleanField(default=False, help_text="Whether quiz was generated for this plan")
    feedback_provided = models.BooleanField(default=False, help_text="Whether user provided feedback")
    
    # Plan effectiveness (for future ML)
    completion_rate = models.FloatField(null=True, blank=True, help_text="User-reported completion rate (0-1)")
    effectiveness_score = models.FloatField(null=True, blank=True, help_text="User-reported effectiveness (1-5)")
    
    # Metadata for flexible analytics
    analytics_metadata = models.JSONField(default=dict, blank=True, 
                                         help_text="Flexible JSON field for additional analytics data")
    
    project_name = models.CharField(max_length=255, blank=True, null=True, help_text="User-defined project name")
    kanban_state = models.JSONField(default=dict, blank=True, help_text="Saved state of kanban columns")
    
    def __str__(self):
        # Update string representation to show project name
        name = self.project_name if self.project_name else "Untitled"
        return f"{name} - {self.user.username} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
    
    def get_schedule(self):
        """Extract schedule from plan_json"""
        for item in self.plan_json:
            if item.get('file') == 'WEEKLY SCHEDULE':
                return item.get('schedule', [])
        return []
    
    def get_tasks(self):
        """Extract prioritized tasks from plan_json"""
        return [
            item for item in self.plan_json 
            if item.get('file') != 'WEEKLY SCHEDULE'
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
    
    def increment_view_count(self):
        """Track when user views this plan"""
        self.viewed_count += 1
        self.last_viewed_at = timezone.now()
        self.save(update_fields=['viewed_count', 'last_viewed_at'])
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Study Plan History"
        verbose_name_plural = "Study Plan Histories"
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['status']),
            models.Index(fields=['sort_method']),
            models.Index(fields=['quiz_generated']),
            models.Index(fields=['feedback_provided']),
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


# ==================== NEW QUIZ SYSTEM MODELS ====================

class QuizSession(models.Model):
    """
    Stores metadata for each quiz generation session.
    Links to the study plan that generated it.
    """
    DIFFICULTY_CHOICES = [
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard'),
        ('mixed', 'Mixed'),
    ]
    
    STATUS_CHOICES = [
        ('generated', 'Generated'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('abandoned', 'Abandoned'),
    ]
    
    # Relationships
    study_plan = models.ForeignKey(StudyPlanHistory, on_delete=models.CASCADE, 
                                   related_name='quizzes', help_text="Study plan this quiz is based on")
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='quizzes')
    
    # Quiz Configuration
    total_questions = models.IntegerField(default=5, help_text="Total number of questions")
    difficulty = models.CharField(max_length=20, choices=DIFFICULTY_CHOICES, default='mixed')
    
    # Timing
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True, help_text="When user started taking quiz")
    completed_at = models.DateTimeField(null=True, blank=True, help_text="When user completed quiz")
    time_limit_seconds = models.IntegerField(null=True, blank=True, help_text="Optional time limit")
    
    # Results
    score = models.FloatField(null=True, blank=True, help_text="Final score (0-100)")
    correct_answers = models.IntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='generated')
    
    # Analytics
    time_spent_seconds = models.IntegerField(default=0, help_text="Total time spent on quiz")
    question_times = models.JSONField(default=list, help_text="Time spent on each question")
    
    # Source tracking (for analytics)
    source_distribution = models.JSONField(default=dict, 
                                          help_text="Distribution of questions from uploaded files vs KB")
    
    # track if this quiz is focused on a specific task/file    
    focus_task_name = models.CharField(max_length=512, null=True, blank=True)
    
    
    def __str__(self):
        return f"Quiz for {self.user.username} - {self.study_plan.id} ({self.status})"
    
    def calculate_score(self):
        """Calculate final score based on correct answers"""
        if self.total_questions == 0:
            return 0
        self.score = (self.correct_answers / self.total_questions) * 100
        self.save(update_fields=['score'])
        return self.score
    
    def mark_started(self):
        """Mark quiz as started"""
        if not self.started_at:
            self.started_at = timezone.now()
            self.status = 'in_progress'
            self.save(update_fields=['started_at', 'status'])
    
    def mark_completed(self):
        """Mark quiz as completed and calculate final metrics"""
        self.completed_at = timezone.now()
        self.status = 'completed'
        
        if self.started_at:
            self.time_spent_seconds = int((self.completed_at - self.started_at).total_seconds())
        
        self.calculate_score()
        self.save(update_fields=['completed_at', 'status', 'time_spent_seconds'])
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Quiz Session"
        verbose_name_plural = "Quiz Sessions"
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['study_plan', 'status']),
            models.Index(fields=['status']),
        ]


class QuizQuestion(models.Model):
    """
    Individual quiz questions generated by LLM.
    Stores question text, options, correct answer, and explanations.
    """
    DIFFICULTY_CHOICES = [
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard'),
    ]
    
    # Relationships
    quiz_session = models.ForeignKey(QuizSession, on_delete=models.CASCADE, 
                                    related_name='questions')
    
    # Question Content
    question_number = models.IntegerField(help_text="Question order (1-5)")
    question_text = models.TextField(help_text="The actual question")
    
    # Options
    option_a = models.TextField(help_text="Option A")
    option_b = models.TextField(help_text="Option B")
    option_c = models.TextField(help_text="Option C", blank=True, null=True)
    option_d = models.TextField(help_text="Option D", blank=True, null=True)
    
    # Answer
    correct_answer = models.CharField(max_length=1, choices=[
        ('a', 'A'), ('b', 'B'), ('c', 'C'), ('d', 'D')
    ], help_text="Correct answer (a/b/c/d)")
    
    # Metadata
    difficulty_level = models.CharField(max_length=20, choices=DIFFICULTY_CHOICES, default='medium')
    source_topic = models.CharField(max_length=256, help_text="Topic/category this question covers")
    source_file = models.CharField(max_length=256, null=True, blank=True, 
                                  help_text="Which uploaded file this question came from")
    
    # AI-generated explanation
    explanation = models.TextField(help_text="Explanation of the correct answer")
    explanation_wrong = models.JSONField(default=dict, blank=True,
                                        help_text="Explanations for why other options are wrong")
    
    # Analytics
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Q{self.question_number}: {self.question_text[:50]}..."
    
    def get_options(self):
        """Get all options as a dictionary"""
        return {
            'a': self.option_a,
            'b': self.option_b,
            'c': self.option_c,
            'd': self.option_d,
        }
    
    class Meta:
        ordering = ['quiz_session', 'question_number']
        verbose_name = "Quiz Question"
        verbose_name_plural = "Quiz Questions"
        indexes = [
            models.Index(fields=['quiz_session', 'question_number']),
        ]
        unique_together = ['quiz_session', 'question_number']


class QuizAnswer(models.Model):
    """
    User's answer to a specific quiz question.
    Tracks correctness and time spent.
    """
    # Relationships
    question = models.ForeignKey(QuizQuestion, on_delete=models.CASCADE, 
                                related_name='user_answers')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # User Response
    user_answer = models.CharField(max_length=1, choices=[
        ('a', 'A'), ('b', 'B'), ('c', 'C'), ('d', 'D')
    ], help_text="User's selected answer")
    
    is_correct = models.BooleanField(help_text="Whether answer was correct")
    
    # Timing
    answered_at = models.DateTimeField(auto_now_add=True)
    time_spent_seconds = models.IntegerField(default=0, help_text="Time spent on this question")
    
    # Optional user feedback on question
    difficulty_rating = models.IntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="User's rating of question difficulty (1-5)"
    )
    
    def __str__(self):
        status = "✓" if self.is_correct else "✗"
        return f"{status} {self.user.username} - Q{self.question.question_number}"
    
    class Meta:
        ordering = ['answered_at']
        verbose_name = "Quiz Answer"
        verbose_name_plural = "Quiz Answers"
        indexes = [
            models.Index(fields=['user', 'question']),
            models.Index(fields=['is_correct']),
        ]


# ==================== FEEDBACK & REINFORCEMENT LEARNING MODELS ====================

class PrioritizationFeedback(models.Model):
    """
    User feedback on prioritization recommendations.
    Used to adjust scoring model over time (reinforcement learning).
    """
    FEEDBACK_TYPE_CHOICES = [
        ('priority', 'Task Priority'),
        ('schedule', 'Schedule Quality'),
        ('content', 'Content Relevance'),
        ('overall', 'Overall Plan'),
    ]
    
    RATING_TYPE_CHOICES = [
        ('thumbs', 'Thumbs Up/Down'),
        ('stars', 'Star Rating'),
        ('detailed', 'Detailed Feedback'),
    ]
    
    # Relationships
    study_plan = models.ForeignKey(StudyPlanHistory, on_delete=models.CASCADE, 
                                   related_name='feedbacks')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='feedbacks')
    
    # Feedback Type
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_TYPE_CHOICES, default='overall')
    rating_type = models.CharField(max_length=20, choices=RATING_TYPE_CHOICES, default='stars')
    
    # Rating Data
    thumbs_up = models.BooleanField(null=True, blank=True, help_text="Thumbs up (True) or down (False)")
    star_rating = models.IntegerField(
        null=True, blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(5)],
        help_text="Star rating (1-5)"
    )
    
    # Task-specific feedback (if applicable)
    task_name = models.CharField(max_length=512, blank=True, help_text="Specific task being rated")
    task_priority_suggested = models.IntegerField(null=True, blank=True, 
                                                  help_text="User's suggested priority for this task")
    
    # Detailed Feedback
    feedback_text = models.TextField(blank=True, help_text="Optional text feedback")
    
    # What aspects were rated
    aspects = models.JSONField(default=dict, blank=True,
                              help_text="Specific aspects rated: {urgency: 4, complexity: 5, etc}")
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True, help_text="For abuse prevention")
    user_agent = models.TextField(blank=True, help_text="Browser/device info")
    
    # Processing Status
    processed = models.BooleanField(default=False, help_text="Whether feedback has been used to adjust model")
    processed_at = models.DateTimeField(null=True, blank=True)
    
    # Additional context for ML
    context_metadata = models.JSONField(default=dict, blank=True,
                                       help_text="Additional context: user experience level, domain, etc")
    
    def __str__(self):
        rating_str = ""
        if self.thumbs_up is not None:
            rating_str = "👍" if self.thumbs_up else "👎"
        elif self.star_rating:
            rating_str = f"{'⭐' * self.star_rating}"
        return f"{self.user.username} - {rating_str} ({self.feedback_type})"
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Prioritization Feedback"
        verbose_name_plural = "Prioritization Feedbacks"
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['study_plan', 'feedback_type']),
            models.Index(fields=['processed', 'created_at']),
            models.Index(fields=['rating_type']),
        ]


class ScoringModelAdjustment(models.Model):
    """
    Tracks adjustments to the scoring model over time.
    Used for reinforcement learning and model improvement.
    """
    SCOPE_CHOICES = [
        ('global', 'Global (All Users)'),
        ('user', 'Per-User Personalized'),
        ('category', 'Per-Category'),
    ]
    
    # Adjustment Metadata
    adjustment_date = models.DateTimeField(auto_now_add=True)
    scope = models.CharField(max_length=20, choices=SCOPE_CHOICES, default='global')
    
    # Scope-specific identifiers
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True,
                            help_text="If scope=user, which user")
    category = models.CharField(max_length=256, blank=True, 
                               help_text="If scope=category, which category")
    
    # Factor Adjustments
    factor_name = models.CharField(max_length=100, 
                                   help_text="Factor being adjusted: urgency/complexity/foundational/kb_weight")
    old_weight = models.FloatField(help_text="Previous weight value")
    new_weight = models.FloatField(help_text="New adjusted weight value")
    adjustment_delta = models.FloatField(help_text="Change amount (new - old)")
    
    # Justification
    reason = models.TextField(help_text="Why this adjustment was made")
    feedback_count_basis = models.IntegerField(default=0, 
                                              help_text="Number of feedback items that led to this adjustment")
    
    # Confidence & Validation
    confidence_score = models.FloatField(default=0.5, 
                                        validators=[MinValueValidator(0), MaxValueValidator(1)],
                                        help_text="Confidence in this adjustment (0-1)")
    
    # Performance Tracking
    applied = models.BooleanField(default=False, help_text="Whether adjustment has been applied to system")
    applied_at = models.DateTimeField(null=True, blank=True)
    
    # Impact metrics (filled after deployment)
    plans_affected = models.IntegerField(default=0, help_text="Number of plans generated with this adjustment")
    avg_feedback_score_before = models.FloatField(null=True, blank=True)
    avg_feedback_score_after = models.FloatField(null=True, blank=True)
    
    # Metadata
    adjustment_metadata = models.JSONField(default=dict, blank=True,
                                          help_text="Additional data: version, algorithm used, etc")
    
    def __str__(self):
        return f"{self.factor_name}: {self.old_weight:.3f} → {self.new_weight:.3f} ({self.scope})"
    
    def apply_adjustment(self):
        """Mark adjustment as applied"""
        self.applied = True
        self.applied_at = timezone.now()
        self.save(update_fields=['applied', 'applied_at'])
    
    class Meta:
        ordering = ['-adjustment_date']
        verbose_name = "Scoring Model Adjustment"
        verbose_name_plural = "Scoring Model Adjustments"
        indexes = [
            models.Index(fields=['scope', '-adjustment_date']),
            models.Index(fields=['user', 'factor_name']),
            models.Index(fields=['category', 'factor_name']),
            models.Index(fields=['applied', 'adjustment_date']),
        ]


# ==================== USER ANALYTICS MODEL ====================

class UserAnalytics(models.Model):
    """
    Aggregate user analytics for personalization and insights.
    Updated periodically based on user activity.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='analytics')
    
    # Usage Statistics
    total_plans_generated = models.IntegerField(default=0)
    total_quizzes_taken = models.IntegerField(default=0)
    total_feedback_provided = models.IntegerField(default=0)
    
    # Performance Metrics
    average_quiz_score = models.FloatField(null=True, blank=True)
    average_plan_rating = models.FloatField(null=True, blank=True)
    
    # Learning Patterns
    preferred_sort_method = models.CharField(max_length=50, blank=True)
    avg_study_hours_allocated = models.FloatField(default=0)
    most_active_time = models.CharField(max_length=50, blank=True, help_text="Time of day most active")
    
    # Subject Preferences (stored as JSON)
    subject_distribution = models.JSONField(default=dict, help_text="Distribution of subjects studied")
    strengths = models.JSONField(default=list, help_text="Subjects/topics user excels at")
    weaknesses = models.JSONField(default=list, help_text="Subjects/topics user struggles with")
    
    # Engagement Metrics
    first_activity = models.DateTimeField(null=True, blank=True)
    last_activity = models.DateTimeField(null=True, blank=True)
    total_time_spent_seconds = models.IntegerField(default=0)
    streak_days = models.IntegerField(default=0, help_text="Consecutive days of activity")
    
    # Personalization Data
    personalized_weights = models.JSONField(default=dict, 
                                           help_text="User-specific factor weights for prioritization")
    
    # Timestamps
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Analytics for {self.user.username}"
    
    def update_from_plan(self, study_plan):
        """Update analytics when user creates a new plan"""
        self.total_plans_generated += 1
        self.last_activity = timezone.now()
        if not self.first_activity:
            self.first_activity = timezone.now()
        self.save()
    
    def update_from_quiz(self, quiz_session):
        """Update analytics when user completes a quiz"""
        self.total_quizzes_taken += 1
        
        # Recalculate average quiz score
        all_quizzes = QuizSession.objects.filter(user=self.user, status='completed')
        scores = [q.score for q in all_quizzes if q.score is not None]
        if scores:
            self.average_quiz_score = sum(scores) / len(scores)
        
        self.last_activity = timezone.now()
        self.save()
    
    def update_from_feedback(self, feedback):
        """Update analytics when user provides feedback"""
        self.total_feedback_provided += 1
        
        # Recalculate average rating
        all_feedback = PrioritizationFeedback.objects.filter(
            user=self.user, 
            star_rating__isnull=False
        )
        ratings = [f.star_rating for f in all_feedback]
        if ratings:
            self.average_plan_rating = sum(ratings) / len(ratings)
        
        self.last_activity = timezone.now()
        self.save()
    
    class Meta:
        verbose_name = "User Analytics"
        verbose_name_plural = "User Analytics"