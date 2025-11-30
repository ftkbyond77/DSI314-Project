#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "tStarting deployment script..."

# 1. LINK MEDIA FILES TO PERSISTENT STORAGE (Azure /home directory)
# Azure Web Apps persist data in /home. We link /app/media to /home/media
# so your user uploads don't disappear after a restart.
if [ -d "/home" ]; then
    echo "Setting up persistent media storage..."
    mkdir -p /home/media
    # Remove existing media folder if it's not a symlink, then link it
    if [ -d "/app/media" ] && [ ! -L "/app/media" ]; then
        rm -rf /app/media
    fi
    ln -sfn /home/media /app/media
fi

# 2. RUN DATABASE MIGRATIONS
echo "Running database migrations..."
python manage.py migrate --noinput

# 3. COLLECT STATIC FILES
echo "Collecting static files..."
python manage.py collectstatic --noinput

# 4. START CELERY WORKER (BACKGROUND TASK)
# Note: In a production enterprise env, this should be a separate container.
# For Azure Student/Dev, we run it in the background of the web container.
echo "Starting Celery worker..."
celery -A student_assistant worker -l info &

# 5. START GUNICORN (WEB SERVER)
echo "Starting Gunicorn..."
# Uses the PORT environment variable provided by Azure (default 8000)
exec gunicorn student_assistant.wsgi:application --bind 0.0.0.0:8000 --workers 3 --timeout 1200