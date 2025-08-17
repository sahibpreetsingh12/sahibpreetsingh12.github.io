#!/usr/bin/env python3
"""
Resume-powered Chatbot API
A FastAPI-based chatbot that answers questions about Sahibpreet Singh using RAG on his resume.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Local imports
from resume_processor import ResumeProcessor
from vector_store import VectorStore
from chatbot_engine import ChatbotEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sahibpreet Singh - Personal Chatbot",
    description="AI-powered chatbot that knows everything about Sahibpreet Singh",
    version="1.0.0"
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
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str]
    conversation_id: str

# Global variables
chatbot_engine: Optional[ChatbotEngine] = None
resume_data: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot system on startup"""
    global chatbot_engine, resume_data
    
    try:
        logger.info("🚀 Starting up Resume Chatbot...")
        
        # Initialize resume processor
        resume_path = Path("../assets/resume/sahibpreet-singh-resume.pdf")
        processor = ResumeProcessor()
        
        # Process resume and create vector store
        logger.info("📄 Processing resume...")
        resume_data = await processor.process_resume(resume_path)
        
        # Initialize vector store
        logger.info("🔍 Creating vector embeddings...")
        vector_store = VectorStore()
        await vector_store.initialize(resume_data['chunks'])
        
        # Initialize chatbot engine
        logger.info("🤖 Initializing chatbot engine...")
        chatbot_engine = ChatbotEngine(vector_store, resume_data)
        
        logger.info("✅ Chatbot ready to serve!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize chatbot: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Sahibpreet Singh's Personal Chatbot API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "chatbot_ready": chatbot_engine is not None,
        "resume_loaded": resume_data is not None,
        "total_chunks": len(resume_data['chunks']) if resume_data else 0
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    if not chatbot_engine:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    try:
        logger.info(f"💬 Processing chat: {request.message[:50]}...")
        
        response = await chatbot_engine.generate_response(
            message=request.message,
            conversation_id=request.conversation_id
        )
        
        return ChatResponse(**response)
        
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/resume/summary")
async def get_resume_summary():
    """Get a summary of the resume content"""
    if not resume_data:
        raise HTTPException(status_code=500, detail="Resume not loaded")
    
    return {
        "summary": resume_data.get('summary', ''),
        "sections": list(resume_data.get('sections', {}).keys()),
        "total_chunks": len(resume_data.get('chunks', [])),
        "key_skills": resume_data.get('key_skills', []),
        "experience_years": resume_data.get('experience_years', 0)
    }

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