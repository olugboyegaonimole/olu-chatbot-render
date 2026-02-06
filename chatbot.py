from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests

app = FastAPI()

# Enable CORS so your frontend can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Minimal change: use a working Hugging Face model
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "facebook/blenderbot-3B"  # <-- updated to a currently supported model
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    payload = {"inputs": request.message}

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"HuggingFace API request failed: {e}")

    data = response.json()

    # Minimal robust extraction of response
    bot_reply = None
    if isinstance(data, list) and "generated_text" in data[0]:
        bot_reply = data[0]["generated_text"]
    elif isinstance(data, dict):
        try:
            bot_reply = data["conversation"]["generated_responses"][0]
        except (KeyError, IndexError, TypeError):
            bot_reply = str(data)

    if not bot_reply:
        bot_reply = "Sorry, I could not generate a response."

    return {"response": bot_reply}

@app.get("/", include_in_schema=False)
@app.head("/"):
def root():
    return {"message": "Welcome to the Chatbot API!"}
