# student_assistant/celery.py
import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "student_assistant.settings")

app = Celery("student_assistant")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()


from celery.schedules import crontab

CELERY_BEAT_SCHEDULE = {
    # Daily tasks
    'update-kb-stats-daily': {
        'task': 'core.knowledge_maintenance.update_knowledge_base_stats',
        'schedule': crontab(hour=3, minute=0),  # 3 AM daily
    },
    'validate-kb-quality-daily': {
        'task': 'core.knowledge_maintenance.validate_kb_grounding_quality',
        'schedule': crontab(hour=4, minute=0),  # 4 AM daily
    },
    
    # Weekly tasks
    'discover-categories-weekly': {
        'task': 'core.knowledge_maintenance.discover_new_categories',
        'schedule': crontab(day_of_week=1, hour=2, minute=0),  # Monday 2 AM
    },
    'analyze-distribution-weekly': {
        'task': 'core.knowledge_maintenance.analyze_kb_distribution',
        'schedule': crontab(day_of_week=1, hour=3, minute=0),  # Monday 3 AM
    },
    'calibrate-thresholds-weekly': {
        'task': 'core.knowledge_maintenance.calibrate_thresholds',
        'schedule': crontab(day_of_week=1, hour=4, minute=0),  # Monday 4 AM
    },
    'update-calibration-weekly': {
        'task': 'core.knowledge_maintenance.update_calibration_parameters',
        'schedule': crontab(day_of_week=1, hour=5, minute=0),  # Monday 5 AM
    },
    'kb-health-report-weekly': {
        'task': 'core.knowledge_maintenance.generate_kb_health_report',
        'schedule': crontab(day_of_week=1, hour=6, minute=0),  # Monday 6 AM
    },
    
    # Monthly tasks
    'clear-kb-caches-monthly': {
        'task': 'core.knowledge_maintenance.clear_all_kb_caches',
        'schedule': crontab(day_of_month=1, hour=1, minute=0),  # 1st of month, 1 AM
    },
}