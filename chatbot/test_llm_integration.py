#!/usr/bin/env python3
"""
Test script to verify LLM integration
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_service import LLMService
from resume_processor import ResumeProcessor
from vector_store import VectorStore
from chatbot_engine import ChatbotEngine

async def test_integration():
    """Test the complete LLM integration"""
    
    print("🧪 Testing LLM Integration...")
    
    # Test 1: LLM Service
    print("\n1. Testing LLM Service...")
    llm_service = LLMService()
    
    if llm_service.is_available():
        print("✅ LLM Service initialized successfully")
        
        # Test basic generation
        response = await llm_service.generate_response(
            message="What is Sahibpreet's experience with AI?",
            context=["Sahibpreet Singh is a GenAI Consultant at CGI specializing in production AI systems."]
        )
        print(f"✅ LLM Response: {response['response'][:100]}...")
        print(f"✅ Model Used: {response.get('model_used')}")
        print(f"✅ Confidence: {response.get('confidence')}")
        
    else:
        print("❌ LLM Service not available")
        return
    
    # Test 2: Resume Processing
    print("\n2. Testing Resume Processing...")
    processor = ResumeProcessor()
    resume_path = Path("../assets/resume/sahibpreet-singh-resume.pdf")
    resume_data = await processor.process_resume(resume_path)
    print(f"✅ Resume processed with {len(resume_data['chunks'])} chunks")
    
    # Test 3: Vector Store
    print("\n3. Testing Vector Store...")
    vector_store = VectorStore()
    await vector_store.initialize(resume_data['chunks'])
    print("✅ Vector store initialized")
    
    # Test 4: Full Chatbot Engine
    print("\n4. Testing Full Chatbot Engine...")
    chatbot = ChatbotEngine(vector_store, resume_data)
    
    # Test different types of questions
    test_questions = [
        "What is Sahibpreet's experience with GenAI?",
        "Tell me about his work at CGI",
        "What are his key technical skills?",
        "What projects has he worked on?",
        "Inappropriate question about other people"  # Should be filtered
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n4.{i} Testing: {question}")
        response = await chatbot.generate_response(question)
        print(f"    Response: {response['response'][:150]}...")
        print(f"    Confidence: {response['confidence']}")
        print(f"    Model Used: {response.get('model_used', 'N/A')}")
        print(f"    Filtered: {response.get('filtered', False)}")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(test_integration())