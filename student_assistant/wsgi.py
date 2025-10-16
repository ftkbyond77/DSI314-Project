"""
WSGI config for student_assistant project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_pinecone")

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'student_assistant.settings')

application = get_wsgi_application()
