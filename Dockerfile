# 3.11 to support newer libraries
FROM python:3.11-slim

# Ensure python output is sent straight to terminal
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip, setuptools, wheel and install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Collect static files during build
RUN python manage.py collectstatic --noinput

# Run the application
CMD ["gunicorn", "student_assistant.wsgi:application", "--bind", "0.0.0.0:8000"]