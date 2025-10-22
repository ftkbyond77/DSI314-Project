FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --default-timeout=180 \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        -r requirements.txt
COPY . .

CMD ["gunicorn", "student_assistant.wsgi:application", "--bind", "0.0.0.0:8000"]
