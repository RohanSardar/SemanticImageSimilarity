FROM python:3.13-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y git && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir --target=/app/deps -r requirements.txt

FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /app/deps /usr/local/lib/python3.13/site-packages
ENV PYTHONPATH=/usr/local/lib/python3.13/site-packages

COPY . .

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
