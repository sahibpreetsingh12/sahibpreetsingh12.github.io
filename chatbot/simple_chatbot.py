#!/usr/bin/env python3
"""
Simple Chatbot - Groq LLM + Markdown Resume + Simple Guardrails
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    print("Groq not found. Install with: pip install groq")
    GROQ_AVAILABLE = False

logger = logging.getLogger(__name__)

class SimpleChatbot:
    """Simple chatbot with Groq LLM + markdown resume + guardrails"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        self.api_key = os.getenv('GROQ_API_KEY')
        self.model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        self.client = None
        self.resume_content = ""
        
        # Initialize Groq client
        if self.api_key and GROQ_AVAILABLE:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info(f"✅ Groq client initialized with {self.model}")
            except Exception as e:
                logger.error(f"❌ Groq initialization failed: {e}")
        
        # Load resume content
        self._load_resume()
    
    def _load_resume(self):
        """Load markdown resume content"""
        resume_path = Path(__file__).parent / "resume.md"
        try:
            with open(resume_path, 'r', encoding='utf-8') as f:
                self.resume_content = f.read()
            logger.info("✅ Resume content loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load resume: {e}")
            self.resume_content = "Resume not available"
    
    def _check_guardrails(self, message: str) -> Dict[str, Any]:
        """Simple guardrails check"""
        message_lower = message.lower().strip()
        
        # Block completely off-topic questions
        off_topic_keywords = [
            'other people', 'someone else', 'another person',
            'weather', 'politics', 'religion', 'personal life',
            'family', 'relationship', 'dating', 'marriage',
            'illegal', 'harmful', 'dangerous', 'violence'
        ]
        
        if any(keyword in message_lower for keyword in off_topic_keywords):
            return {
                'is_appropriate': False,
                'response': "I'm designed to share information about Sahibpreet Singh's professional background. Please ask about his work experience, skills, education, or projects."
            }
        
        # Block requests for other people's information
        other_people_patterns = [
            'tell me about john', 'who is mary', 'what about alex',
            'information about someone', 'details of another'
        ]
        
        if any(pattern in message_lower for pattern in other_people_patterns):
            return {
                'is_appropriate': False,
                'response': "I can only provide information about Sahibpreet Singh. Please ask about his professional background, skills, or experience."
            }
        
        # All other questions are allowed (Sahibpreet-related or general professional)
        return {'is_appropriate': True}
    
    async def generate_response(self, message: str) -> Dict[str, Any]:
        """Generate response with guardrails + LLM"""
        
        # Step 1: Check guardrails
        guardrail_check = self._check_guardrails(message)
        if not guardrail_check['is_appropriate']:
            return {
                'response': guardrail_check['response'],
                'confidence': 0.9,
                'filtered': True
            }
        
        # Step 2: Generate LLM response
        if not self.client:
            return {
                'response': "I'm currently unable to process your request. Please ensure the Groq API is properly configured.",
                'confidence': 0.1,
                'error': True
            }
        
        try:
            # Create system prompt with resume
            system_prompt = f"""You are a professional AI assistant representing Sahibpreet Singh. Your role is to provide accurate, helpful information about his professional background based on his resume.

GUIDELINES:
1. Answer questions about Sahibpreet Singh's professional background only
2. Base responses on the resume information provided below
3. Be professional, friendly, and concise
4. Format responses with proper bullet points using • symbols for better readability
5. If asked about something not in the resume, politely say you don't have that information
6. Don't make up information not present in the resume

RESUME INFORMATION:
{self.resume_content}

Please answer the user's question based on this information. Format your response with clear bullet points and structure where appropriate."""

            # Generate response
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                temperature=0.7,
                max_tokens=500,
                top_p=0.9
            )
            
            response_text = completion.choices[0].message.content.strip()
            
            return {
                'response': response_text,
                'confidence': 0.85,
                'model_used': self.model,
                'tokens_used': completion.usage.total_tokens if completion.usage else 0
            }
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return {
                'response': "I apologize, but I'm having trouble generating a response right now. Please try again.",
                'confidence': 0.1,
                'error': str(e)
            }

# Test function
async def test_chatbot():
    """Test the simplified chatbot"""
    chatbot = SimpleChatbot()
    
    test_questions = [
        "What is Sahibpreet's experience with AI?",
        "Tell me about his work at CGI",
        "What are his technical skills?",
        "What projects has he worked on?", 
        "Tell me about the weather",  # Should be blocked
        "Who is John Smith?"  # Should be blocked
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        response = await chatbot.generate_response(question)
        print(f"🤖 Response: {response['response'][:200]}...")
        print(f"📊 Filtered: {response.get('filtered', False)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_chatbot())