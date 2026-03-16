FROM python:3.11-slim

WORKDIR /app

# Install git for Hugging Face downloads
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model to "bake" it into the image
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0'; \
    AutoTokenizer.from_pretrained(model_name); \
    AutoModelForCausalLM.from_pretrained(model_name)"

COPY app.py .

EXPOSE 8000

# Using --workers 1 to save RAM on Railway's free tier
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
