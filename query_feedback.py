#!/usr/bin/env python
# query_feedback.py

import os
import django
from pprint import pprint

# กำหนด settings ของ Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "student_assistant.settings")
django.setup()

from core.models import PrioritizationFeedback
from django.contrib.auth import get_user_model

User = get_user_model()

def main():
    # Example 1: ดู feedback ทั้งหมดของผู้ใช้
    username = 'jame'  # เปลี่ยนเป็น username ที่ต้องการ
    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        print(f"User {username} not found")
        return

    feedbacks = PrioritizationFeedback.objects.filter(user=user)
    print(f"Feedbacks for user {username}: {feedbacks.count()}")
    for f in feedbacks:
        pprint({
            "created_at": f.created_at,
            "type": f.feedback_type,
            "thumbs_up": f.thumbs_up,
            "star_rating": f.star_rating,
            "feedback_text": f.feedback_text,
            "aspects": f.aspects,
        })

    # Example 2: Feedback ยังไม่ processed
    unprocessed = PrioritizationFeedback.objects.filter(processed=False)
    print(f"\nUnprocessed feedbacks: {unprocessed.count()}")

if __name__ == "__main__":
    main()
