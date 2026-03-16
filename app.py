from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import torch
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# 1. Model and Environment Setup# Change this line
MODEL_NAME = "sshleifer/tiny-gpt2"
HF_TOKEN = os.getenv("HF_TOKEN") 

print(f"Loading model: {MODEL_NAME}")

try:
    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, 
        token=HF_TOKEN
    )
    
    # 3. Load Model (16-bit precision, auto-device mapping)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN
    )
    
    # 4. Initialize Pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200
    )
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    pipe = None

@app.get("/")
async def root():
    return {
        "status": "running",
        "model": MODEL_NAME,
        "endpoints": ["/generate", "/chat"]
    }

@app.post("/generate")
async def generate(prompt: str, max_length: int = 50): # Reduced default length
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        # Explicitly passing parameters to avoid the config warning
        result = pipe(
            prompt, 
            max_new_tokens=20, # Use max_new_tokens instead of max_length for speed
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id # Explicitly set pad token
        )
        return {
            "prompt": prompt,
            "generated_text": result[0]["generated_text"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chat")
async def chat(message: str):
    """Chat endpoint using TinyLlama's specific chat template"""
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Proper token formatting for TinyLlama instruct models
    formatted_prompt = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{message}</s>\n<|assistant|>\n"
    
    try:
        result = pipe(formatted_prompt, max_length=150, num_return_sequences=1)
        response = result[0]["generated_text"].split("<|assistant|>\n")[-1].strip()
        return {
            "user_message": message,
            "assistant_response": response
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Railway automatically injects the PORT environment variable
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
