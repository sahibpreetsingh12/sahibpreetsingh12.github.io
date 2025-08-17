#!/usr/bin/env python3
"""
Chatbot Engine
Main conversational AI engine with RAG and guardrails for Sahibpreet Singh's resume chatbot.
"""

import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from vector_store import VectorStore
from guardrails import GuardrailsEngine

logger = logging.getLogger(__name__)

class ChatbotEngine:
    """Main chatbot engine with RAG capabilities and guardrails"""
    
    def __init__(self, vector_store: VectorStore, resume_data: Dict[str, Any]):
        self.vector_store = vector_store
        self.resume_data = resume_data
        self.guardrails = GuardrailsEngine()
        self.conversations = {}  # Store conversation history
        
        # Predefined responses for common questions
        self.template_responses = {
            'greeting': [
                "Hello! I'm here to help you learn about Sahibpreet Singh's professional background. What would you like to know about his experience, skills, or projects?",
                "Hi there! I can provide information about Sahibpreet Singh's career, technical expertise, and achievements. How can I assist you?",
                "Welcome! I'm designed to share details about Sahibpreet Singh's professional journey. What aspect of his background interests you?"
            ],
            'general_about': [
                "Sahibpreet Singh is a GenAI Consultant at CGI specializing in production-scale AI systems. He has delivered over $700K in project value through Agentic RAG systems and achieved significant efficiency improvements. He's an expert in PyTorch, Transformers, Azure ML, and custom CUDA kernel development.",
                "Sahibpreet is a highly skilled GenAI professional currently working at CGI, where he builds AI systems that scale beyond demos to real-world applications. His expertise spans from LLM optimization to enterprise RAG architectures."
            ]
        }
    
    async def generate_response(self, message: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response to user message with guardrails and RAG
        
        Args:
            message: User's input message
            conversation_id: Optional conversation ID for context
            
        Returns:
            Dictionary containing response and metadata
        """
        logger.info(f"💬 Processing message: {message[:50]}...")
        
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Initialize conversation if new
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                'messages': [],
                'created_at': datetime.now().isoformat()
            }
        
        try:
            # Step 1: Check message appropriateness
            guardrail_check = self.guardrails.check_query_appropriateness(message)
            
            if not guardrail_check['is_appropriate']:
                response = {
                    'response': guardrail_check['response'],
                    'confidence': 0.9,
                    'sources': [],
                    'conversation_id': conversation_id,
                    'filtered': True,
                    'filter_reason': guardrail_check['reason']
                }
                self._store_conversation_turn(conversation_id, message, response['response'])
                return response
            
            # Step 2: Check for template responses (greetings, common questions)
            template_response = self._check_template_responses(message)
            if template_response:
                response = {
                    'response': template_response,
                    'confidence': 0.95,
                    'sources': ['template'],
                    'conversation_id': conversation_id
                }
                self._store_conversation_turn(conversation_id, message, response['response'])
                return response
            
            # Step 3: RAG-based response generation
            rag_response = await self._generate_rag_response(message, conversation_id)
            
            # Step 4: Apply guardrails to response
            filtered_response = self.guardrails.filter_sensitive_information(rag_response['response'])
            enhanced_response = self.guardrails.enhance_response_with_context(filtered_response, message)
            
            # Step 5: Validate response quality
            quality_check = self.guardrails.validate_response_quality(enhanced_response, message)
            
            if not quality_check['is_valid']:
                logger.warning(f"⚠️ Response quality issues: {quality_check['issues']}")
                # Fallback to a safe generic response
                enhanced_response = "I'd be happy to help you learn about Sahibpreet Singh's professional background. Could you please rephrase your question or ask about a specific aspect of his experience?"
            
            final_response = {
                'response': enhanced_response,
                'confidence': rag_response['confidence'],
                'sources': rag_response['sources'],
                'conversation_id': conversation_id
            }
            
            # Store conversation
            self._store_conversation_turn(conversation_id, message, final_response['response'])
            
            return final_response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            
            # Safe fallback response
            fallback_response = {
                'response': "I apologize, but I'm having trouble processing your request right now. Please try asking about Sahibpreet Singh's experience, skills, or projects.",
                'confidence': 0.1,
                'sources': [],
                'conversation_id': conversation_id,
                'error': True
            }
            
            return fallback_response
    
    def _check_template_responses(self, message: str) -> Optional[str]:
        """Check if message matches predefined templates"""
        message_lower = message.lower().strip()
        
        # Greeting patterns
        greeting_patterns = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon']
        if any(pattern in message_lower for pattern in greeting_patterns) and len(message_lower) < 20:
            import random
            return random.choice(self.template_responses['greeting'])
        
        # General "about" questions
        about_patterns = ['who is sahibpreet', 'tell me about sahibpreet', 'about sahibpreet singh', 'who are you']
        if any(pattern in message_lower for pattern in about_patterns):
            import random
            return random.choice(self.template_responses['general_about'])
        
        return None
    
    async def _generate_rag_response(self, message: str, conversation_id: str) -> Dict[str, Any]:
        """Generate response using RAG (Retrieval-Augmented Generation)"""
        
        # Step 1: Retrieve relevant chunks
        relevant_chunks = await self.vector_store.search(message, top_k=3)
        
        if not relevant_chunks:
            return {
                'response': "I don't have specific information about that aspect of Sahibpreet Singh's background. Could you try asking about his work experience, technical skills, or projects?",
                'confidence': 0.3,
                'sources': []
            }
        
        # Step 2: Prepare context from retrieved chunks
        context_pieces = []
        sources = []
        
        for chunk, score in relevant_chunks:
            if score > 0.1:  # Only use chunks with reasonable similarity
                context_pieces.append(chunk['content'])
                sources.append(chunk.get('metadata', {}).get('section', 'resume'))
        
        # Step 3: Generate response using template-based approach
        response = self._generate_template_based_response(message, context_pieces)
        
        # Calculate confidence based on similarity scores
        max_score = max([score for _, score in relevant_chunks]) if relevant_chunks else 0
        confidence = min(max_score * 1.2, 0.95)  # Cap at 95%
        
        return {
            'response': response,
            'confidence': confidence,
            'sources': sources
        }
    
    def _generate_template_based_response(self, message: str, context_pieces: List[str]) -> str:
        """Generate response using template-based approach with context"""
        
        message_lower = message.lower()
        
        # Combine context
        combined_context = ' '.join(context_pieces)
        
        # Question type detection and response generation
        if any(word in message_lower for word in ['experience', 'work', 'job', 'role', 'career']):
            return self._extract_experience_info(combined_context)
        
        elif any(word in message_lower for word in ['skill', 'technology', 'technical', 'programming']):
            return self._extract_skills_info(combined_context)
        
        elif any(word in message_lower for word in ['education', 'degree', 'study', 'university', 'college']):
            return self._extract_education_info(combined_context)
        
        elif any(word in message_lower for word in ['project', 'built', 'developed', 'created']):
            return self._extract_projects_info(combined_context)
        
        elif any(word in message_lower for word in ['research', 'publication', 'paper']):
            return self._extract_research_info(combined_context)
        
        elif any(word in message_lower for word in ['achievement', 'accomplishment', 'award', 'success']):
            return self._extract_achievements_info(combined_context)
        
        elif any(word in message_lower for word in ['contact', 'email', 'reach', 'linkedin', 'github']):
            return self._extract_contact_info(combined_context)
        
        else:
            # General response using available context
            return f"Based on Sahibpreet Singh's background: {combined_context[:300]}..."
    
    def _extract_experience_info(self, context: str) -> str:
        """Extract and format experience information"""
        if 'cgi' in context.lower():
            return """Sahibpreet Singh is currently a GenAI Consultant at CGI, where he has made significant impact:

• Architected Agentic RAG systems resulting in a $700K project win
• Developed Zero-Trust RAG system achieving 65% faster recruitment processes  
• Optimized Databricks + PySpark pipelines with 31% performance improvement
• Led cross-functional teams in implementing enterprise AI solutions

Previously, he worked at AI Talentflow as a GenAI Engineer, Tatras Data as a Data Scientist, and ZS Associates as an ML Engineer, consistently delivering high-impact results across different industries."""
        
        return "Sahibpreet Singh has extensive experience in GenAI and ML engineering across multiple companies, currently serving as a GenAI Consultant at CGI where he specializes in production-scale AI systems."
    
    def _extract_skills_info(self, context: str) -> str:
        """Extract and format technical skills information"""
        return """Sahibpreet Singh's technical expertise includes:

**AI/ML Frameworks:** PyTorch, Transformers, Langchain, LlamaIndex
**Agentic AI:** CrewAI, Langgraph, Autogen, SmolAgents  
**Cloud Platforms:** Azure (Promptflow, ML Studio, Key Vault), AWS (ECS, Lambda, SageMaker)
**Data Engineering:** Databricks, PySpark, Neo4j, MongoDB, CosmosDB
**DevOps & Infrastructure:** Docker, Terraform, GitHub Actions, Kubernetes
**Programming Languages:** Python, CUDA, SQL, JavaScript

He specializes in building production-scale AI systems, custom CUDA kernels, and enterprise RAG architectures."""
    
    def _extract_education_info(self, context: str) -> str:
        """Extract and format education information"""
        return """Sahibpreet Singh's educational background includes:

• **Post Graduate Certificate in AI & Machine Learning** - Lambton College (2024-2025)
• **Bachelor of Technology in Computer Science** - Punjab Technical University (2017-2021)

He is currently pursuing advanced studies in AI/ML to stay at the forefront of rapidly evolving technologies in the field."""
    
    def _extract_projects_info(self, context: str) -> str:
        """Extract and format projects information"""
        return """Sahibpreet Singh has led several high-impact projects:

• **Zero-Trust RAG System:** Enterprise AI security solution with semantic matching and Azure Key Vault integration
• **Custom CUDA Kernels:** Optimized GPU acceleration for LLM inference using Triton
• **Resume-2-ResumeRAG:** Production GenAI system with fine-tuned LLMs deployed on AWS ECS  
• **Tokenizer Fertility Research:** Novel insights into subword optimization for production LLMs

These projects demonstrate his ability to deliver enterprise-scale AI solutions that drive real business value."""
    
    def _extract_research_info(self, context: str) -> str:
        """Extract and format research information"""
        return """Sahibpreet Singh's current research focuses on:

• **Tokenizer fertility rates** and their impact on LLM performance in production systems
• **Zero-trust AI architectures** for secure enterprise deployment
• **Custom CUDA kernels** for accelerated inference optimization
• **Multi-agent RAG evaluation frameworks** for production systems

His research combines theoretical insights with practical applications, contributing to the advancement of production-ready AI systems."""
    
    def _extract_achievements_info(self, context: str) -> str:
        """Extract and format achievements information"""
        return """Sahibpreet Singh's key achievements include:

• **$700K+ project value** delivered at CGI through innovative RAG systems
• **65% efficiency improvement** in recruitment processes using AI
• **31% ML pipeline optimization** using Databricks and PySpark
• **100+ engineers mentored** in AI/ML practices and best practices
• **IEEE Hackathon Winner (2nd Place)** for Explainable AI solution
• **Technical content creator** with growing audience on LinkedIn

These accomplishments demonstrate his ability to drive significant business impact through AI innovation."""
    
    def _extract_contact_info(self, context: str) -> str:
        """Extract and format contact information"""
        return """You can connect with Sahibpreet Singh through:

• **Email:** ss9334931@gmail.com
• **LinkedIn:** linkedin.com/in/sahibpreetsinghh/
• **GitHub:** github.com/sahibpreetsingh12  
• **Kaggle:** kaggle.com/sahib12

He's always open to discussing GenAI systems, production ML challenges, LLM evaluation frameworks, and collaboration opportunities in the AI space."""
    
    def _store_conversation_turn(self, conversation_id: str, user_message: str, bot_response: str):
        """Store conversation turn for context"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]['messages'].append({
                'timestamp': datetime.now().isoformat(),
                'user_message': user_message,
                'bot_response': bot_response
            })
            
            # Keep only last 10 turns to manage memory
            if len(self.conversations[conversation_id]['messages']) > 10:
                self.conversations[conversation_id]['messages'] = \
                    self.conversations[conversation_id]['messages'][-10:]
    
    def get_conversation_history(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation history for a given ID"""
        return self.conversations.get(conversation_id)