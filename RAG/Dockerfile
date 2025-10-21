FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy โค้ดทั้งหมด (optional เพราะเราจะ mount volume)
# COPY . /app/

CMD ["bash"]
