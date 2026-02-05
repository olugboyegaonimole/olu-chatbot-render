from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "facebook/blenderbot-400M-distill"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    payload = {"inputs": request.message}

    response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)

    if response.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"HuggingFace API error: {response.text}"
        )

    data = response.json()

    # Robustly extract generated text
    if isinstance(data, list) and "generated_text" in data[0]:
        bot_reply = data[0]["generated_text"]
    elif isinstance(data, dict) and "generated_text" in data:
        bot_reply = data["generated_text"]
    else:
        raise HTTPException(status_code=500, detail=f"Unexpected API response format: {data}")

    return {"response": bot_reply}


@app.get("/", include_in_schema=False)
@app.head("/")
def root():
    return {"message": "Welcome to the Chatbot API!"}
