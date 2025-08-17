#!/usr/bin/env python3
"""
Simple Resume Chatbot API
A FastAPI-based chatbot that answers questions about Sahibpreet Singh using Groq LLM.
"""

import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Local imports
from simple_chatbot import SimpleChatbot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sahibpreet Singh - Personal Chatbot",
    description="Simple AI-powered chatbot using Groq LLM",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    confidence: float
    filtered: Optional[bool] = False

# Global chatbot instance
chatbot: Optional[SimpleChatbot] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the simple chatbot on startup"""
    global chatbot
    
    try:
        logger.info("🚀 Starting up Simple Resume Chatbot...")
        chatbot = SimpleChatbot()
        logger.info("✅ Simple chatbot ready to serve!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize chatbot: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Sahibpreet Singh's Simple Chatbot API",
        "version": "2.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "chatbot_ready": chatbot is not None,
        "llm_available": chatbot.client is not None if chatbot else False
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    if not chatbot:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    try:
        logger.info(f"💬 Processing chat: {request.message[:50]}...")
        
        response = await chatbot.generate_response(request.message)
        
        return ChatResponse(
            response=response['response'],
            confidence=response['confidence'],
            filtered=response.get('filtered', False)
        )
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/chat/suggestions")
async def get_chat_suggestions():
    """Get suggested questions users can ask"""
    suggestions = [
        "What is Sahibpreet's experience with GenAI?",
        "Tell me about his work at CGI",
        "What are his key technical skills?",
        "What projects has he worked on?",
        "What is his educational background?",
        "How much impact has he delivered in terms of revenue?",
        "What research is he currently working on?",
        "What programming languages does he know?",
        "Tell me about his experience with RAG systems",
        "What makes him unique as an AI consultant?"
    ]
    
    return {"suggestions": suggestions}

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )