#!/usr/bin/env python3
"""
LLM Service
Secure Groq LLM integration for dynamic response generation.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    print("Groq not found. Install with: pip install groq")
    GROQ_AVAILABLE = False

logger = logging.getLogger(__name__)

class LLMService:
    """Secure Groq LLM service for dynamic response generation"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        self.api_key = os.getenv('GROQ_API_KEY')
        self.model = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        self.client = None
        
        if not self.api_key:
            logger.warning("⚠️ GROQ_API_KEY not found in environment variables")
            return
            
        if GROQ_AVAILABLE:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info(f"✅ Groq LLM service initialized with model: {self.model}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Groq client: {e}")
        else:
            logger.warning("⚠️ Groq library not available")
    
    def is_available(self) -> bool:
        """Check if LLM service is available"""
        return self.client is not None
    
    async def generate_response(self, 
                              message: str, 
                              context: List[str], 
                              conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Generate dynamic response using Groq LLM
        
        Args:
            message: User's question
            context: Retrieved context chunks from RAG
            conversation_history: Previous conversation turns
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.is_available():
            return {
                'response': "I'm currently unable to generate dynamic responses. Please try again later.",
                'confidence': 0.1,
                'model_used': 'fallback'
            }
        
        try:
            # Prepare context
            context_text = "\n".join(context) if context else "No specific context available."
            
            # Prepare conversation history
            history_text = ""
            if conversation_history:
                recent_history = conversation_history[-3:]  # Last 3 turns
                for turn in recent_history:
                    history_text += f"User: {turn.get('user_message', '')}\nAssistant: {turn.get('bot_response', '')}\n"
            
            # Create the prompt
            system_prompt = """You are a professional AI assistant representing Sahibpreet Singh, a GenAI Consultant at CGI. Your role is to provide accurate, helpful information about his professional background based on the provided context.

IMPORTANT GUIDELINES:
1. Only answer questions about Sahibpreet Singh's professional background
2. Base your responses on the provided context
3. If information isn't in the context, politely say you don't have that specific information
4. Maintain a professional, friendly tone
5. Be concise but informative
6. Do NOT make up information not present in the context
7. Focus on his experience, skills, projects, education, and achievements
8. If asked about other people or unrelated topics, politely redirect to Sahibpreet's background

Context about Sahibpreet Singh:
{context}

Recent conversation:
{history}"""

            user_prompt = f"Question: {message}"
            
            # Generate response
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": system_prompt.format(
                            context=context_text,
                            history=history_text
                        )
                    },
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                top_p=0.9
            )
            
            response_text = completion.choices[0].message.content.strip()
            
            # Calculate confidence based on context relevance
            confidence = self._calculate_confidence(message, context, response_text)
            
            logger.info(f"✅ Generated LLM response for: {message[:50]}...")
            
            return {
                'response': response_text,
                'confidence': confidence,
                'model_used': self.model,
                'tokens_used': completion.usage.total_tokens if completion.usage else 0
            }
            
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return {
                'response': "I apologize, but I'm having trouble generating a response right now. Please try rephrasing your question about Sahibpreet Singh's background.",
                'confidence': 0.1,
                'model_used': 'error_fallback',
                'error': str(e)
            }
    
    async def validate_query_appropriateness(self, message: str) -> Dict[str, Any]:
        """
        Use LLM to validate if query is appropriate for Sahibpreet's chatbot
        
        Args:
            message: User's input message
            
        Returns:
            Dictionary with validation results
        """
        if not self.is_available():
            return {'is_appropriate': True, 'reason': 'LLM validation unavailable'}
        
        try:
            validation_prompt = """You are a content moderator for a professional resume chatbot about Sahibpreet Singh, a GenAI Consultant at CGI.

Determine if the following user query is appropriate for a professional resume chatbot. The query should be:
✅ APPROPRIATE if it asks about:
- Professional experience, skills, education
- Work projects, achievements, career background
- Technical expertise, certifications
- Contact information for professional purposes
- General professional background questions

❌ INAPPROPRIATE if it asks about:
- Personal life, family, relationships
- Other people's information
- Inappropriate, offensive, or harmful content
- Topics completely unrelated to professional background
- Requests for illegal or harmful activities
- General tech support or tutorials not related to Sahibpreet

Respond with just "APPROPRIATE" or "INAPPROPRIATE" followed by a brief reason.

Query: {message}"""

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": validation_prompt.format(message=message)}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            response = completion.choices[0].message.content.strip().upper()
            
            if "INAPPROPRIATE" in response:
                return {
                    'is_appropriate': False,
                    'reason': 'LLM flagged as inappropriate',
                    'response': "I'm designed to share information about Sahibpreet Singh's professional background. Could you please ask about his work experience, skills, or projects instead?"
                }
            else:
                return {'is_appropriate': True, 'reason': 'LLM approved'}
                
        except Exception as e:
            logger.error(f"❌ LLM validation failed: {e}")
            # Fallback to allowing the query if LLM fails
            return {'is_appropriate': True, 'reason': f'LLM validation error: {str(e)}'}
    
    def _calculate_confidence(self, message: str, context: List[str], response: str) -> float:
        """Calculate confidence score based on context relevance"""
        base_confidence = 0.7
        
        # Increase confidence if context is provided
        if context and len(context) > 0:
            base_confidence += 0.1
        
        # Increase confidence if response is substantial
        if len(response) > 50:
            base_confidence += 0.1
            
        # Decrease confidence if response indicates uncertainty
        uncertainty_phrases = ["i don't have", "not sure", "unclear", "don't know"]
        if any(phrase in response.lower() for phrase in uncertainty_phrases):
            base_confidence -= 0.2
            
        return min(max(base_confidence, 0.1), 0.95)  # Keep between 0.1 and 0.95