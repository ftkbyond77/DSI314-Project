# ==========================================
# STAGE 1: Builder (Compiles and installs)
# ==========================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies (compilers needed for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment to install dependencies into
RUN python -m venv /opt/venv
# Enable the virtual environment for subsequent commands
ENV PATH="/opt/venv/bin:$PATH"

# 1. Install CPU-only PyTorch first (The heavy hitter)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the requirements
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# STAGE 2: Runtime (Final Clean Image)
# ==========================================
FROM python:3.11-slim

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install ONLY the runtime system libraries needed for your app
# (poppler-utils for PDF, libgl1 for EasyOCR/OpenCV)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Enable the virtual environment in the final image
ENV PATH="/opt/venv/bin:$PATH"

# Copy your application code
COPY . .

# Setup and permissions for startup script
COPY startup.sh /app/startup.sh
RUN chmod +x /app/startup.sh

# Expose port
EXPOSE 8000

# Start command
CMD ["/app/startup.sh"]