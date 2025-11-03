FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "student_assistant.wsgi:application", "--bind", "0.0.0.0:8000"]
