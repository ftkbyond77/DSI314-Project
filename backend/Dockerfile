FROM python:3.11-slim

WORKDIR /app

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy dependencies first
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run migrations and start server
CMD bash -c "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"
