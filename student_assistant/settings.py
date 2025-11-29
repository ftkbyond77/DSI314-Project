import os
from pathlib import Path
from datetime import timedelta
import dj_database_url

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "unsafe-secret-key-for-dev")

DEBUG = os.getenv("DEBUG", "False") == "True"

ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

# [AZURE FIX] Add Azure Web App Hostname automatically
azure_hostname = os.getenv("WEBSITE_HOSTNAME")  # Azure sets this automatically
if azure_hostname:
    ALLOWED_HOSTS.append(azure_hostname)

# Allow custom domains defined in env vars
custom_hosts = os.getenv("ALLOWED_CUSTOM_HOSTS")
if custom_hosts:
    ALLOWED_HOSTS.extend(custom_hosts.split(","))

# [AZURE FIX] Trust CSRF from Azure domains (Required for Django 4+)
CSRF_TRUSTED_ORIGINS = [
    "https://" + os.getenv("WEBSITE_HOSTNAME", "localhost"),
]
if custom_hosts:
    for host in custom_hosts.split(","):
        CSRF_TRUSTED_ORIGINS.append(f"https://{host}")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "whitenoise.runserver_nostatic",  # Optimized static handling
    "django.contrib.staticfiles",
    "rest_framework",
    "core",
]

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(minutes=60),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=1),
}

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "student_assistant.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        'DIRS': [BASE_DIR / 'templates'],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ]
        },
    }
]

WSGI_APPLICATION = "student_assistant.wsgi.application"

# Database Configuration
DATABASES = {
    "default": dj_database_url.config(
        default=os.getenv("DATABASE_URL", "postgresql://student_user:student_pass@db:5432/student_db"),
        conn_max_age=600,
        ssl_require=True  # Azure Postgres usually requires SSL
    )
}

# Static files (CSS, JavaScript, Images)
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# Media files
# [AZURE FIX] We keep this standard, but startup.sh will link it to /home/media
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# WhiteNoise configuration
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# Celery Configuration
CELERY_BROKER_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("REDIS_URL", "redis://redis:6379/0")

LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/upload/'  
LOGOUT_REDIRECT_URL = '/login/'