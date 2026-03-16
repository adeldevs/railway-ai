FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for some ML packages
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install CPU-only torch
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model - using TinyLlama
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0'; \
    AutoTokenizer.from_pretrained(model_name); \
    AutoModelForCausalLM.from_pretrained(model_name)"

COPY app.py .

EXPOSE 8000
CMD ["python", "app.py"]