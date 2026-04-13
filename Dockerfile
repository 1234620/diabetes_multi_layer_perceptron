FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8020

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application source.
COPY backend /app/backend
COPY frontend /app/frontend

EXPOSE 8020

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8020"]