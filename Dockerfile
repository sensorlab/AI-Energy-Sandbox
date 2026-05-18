# 1. Base image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies (Fixes the phik/CMake error)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 4. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy application code
COPY ./app ./app

# 6. Run the app
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]