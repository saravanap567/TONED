# Dockerfile for T1D Environment Server
# Meta PyTorch Hackathon - Round 1
# Optimized for vcpu=2, memory=8gb

FROM python:3.10-slim

WORKDIR /app

# Install slim deps only (no gradio/matplotlib — saves ~400MB RAM)
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY . .

# Required environment variables (per hackathon guidelines)
ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""
ENV ENV_SERVER_URL="http://localhost:7860"

EXPOSE 7860

# Server runs as foreground process — keeps container alive for HF Space
CMD ["python", "server.py"]
