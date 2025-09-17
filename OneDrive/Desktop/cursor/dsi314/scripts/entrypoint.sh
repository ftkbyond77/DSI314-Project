#!/usr/bin/env sh
set -e

python manage.py collectstatic --noinput || true
python manage.py migrate --noinput

# Use runserver for simplicity in demo
python manage.py runserver 0.0.0.0:8000

