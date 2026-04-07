# Dockerfile for T1D Environment Server
# Meta PyTorch Hackathon - Round 1
# OpenEnv-compliant server deployment

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"

# Expose port
EXPOSE 7860

# Run OpenEnv server
CMD ["python", "server.py"]
