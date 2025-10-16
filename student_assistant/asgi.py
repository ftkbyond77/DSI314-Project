"""
ASGI config for student_assistant project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_pinecone")

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'student_assistant.settings')

application = get_asgi_application()
