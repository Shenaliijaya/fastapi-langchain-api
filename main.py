import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# Load the API key from the .env file
load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY was not found in the .env file.")

app = FastAPI(
    title="LLM Communication API",
    description="A simple FastAPI REST API that communicates with Google Gemini through LangChain.",
    version="1.0.0"
)

# Defines the JSON data accepted by POST /chat
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)

# Defines the JSON data returned by POST /chat
class ChatResponse(BaseModel):
    response: str

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
    temperature=0.2
)

@app.get("/")
def read_root():
    return {
        "message": "LLM Communication API is running",
        "documentation": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = llm.invoke(request.message)

        return ChatResponse(
            response=str(result.content)
        )

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="The Gemini service is currently unavailable. Please try again."
        )