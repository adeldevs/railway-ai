FROM python:3.11-slim

WORKDIR /app

# 1. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Pre-download model weights into the Docker image
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0'; \
    AutoTokenizer.from_pretrained(model_name); \
    AutoModelForCausalLM.from_pretrained(model_name)"

# 3. Copy application code
COPY app.py .

EXPOSE 8000

CMD ["python", "app.py"]