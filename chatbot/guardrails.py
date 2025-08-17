#!/usr/bin/env python3
"""
Guardrails System
Protects the chatbot from off-topic questions and inappropriate use.
"""

import re
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class GuardrailsEngine:
    """Implements guardrails to keep conversations focused on Sahibpreet Singh"""
    
    def __init__(self):
        self.allowed_topics = {
            'sahibpreet_personal': [
                'sahibpreet', 'singh', 'about you', 'who are you', 'tell me about yourself',
                'your background', 'your experience', 'your skills', 'your education'
            ],
            'professional': [
                'work', 'job', 'role', 'position', 'career', 'experience', 'employment',
                'projects', 'achievements', 'accomplishments', 'responsibilities'
            ],
            'technical': [
                'skills', 'technology', 'programming', 'ai', 'ml', 'machine learning',
                'genai', 'rag', 'llm', 'pytorch', 'python', 'azure', 'aws', 'cuda'
            ],
            'education': [
                'education', 'degree', 'university', 'college', 'study', 'learning',
                'academic', 'course', 'certification'
            ],
            'research': [
                'research', 'publication', 'paper', 'study', 'investigation',
                'tokenizer', 'optimization'
            ],
            'contact_info': [
                'contact', 'email', 'phone', 'linkedin', 'github', 'reach',
                'connect', 'hiring', 'available'
            ]
        }
        
        # Topics/questions to politely decline
        self.forbidden_patterns = [
            # Personal questions about others
            r'\b(who is|tell me about|what about)\s+(?!sahibpreet|singh)\w+',
            
            # Requests for other people's information
            r'\b(email|phone|contact).*\b(?!sahibpreet|singh)\w+\b',
            
            # Technical help unrelated to Sahibpreet
            r'\b(help me|how to|write|code|debug|fix)\b.*\b(?!sahibpreet|about sahibpreet)',
            
            # General knowledge questions
            r'\b(what is|define|explain)(?!.*sahibpreet).*\b(python|ai|ml|programming)',
            
            # Random personal questions
            r'\b(your|my|his|her)\s+(age|weight|height|address|salary|personal)',
            
            # Inappropriate content
            r'\b(hack|steal|illegal|inappropriate|offensive)\b',
            
            # Off-topic requests
            r'\b(weather|news|sports|politics|celebrity|movie|music)\b',
        ]
        
        self.polite_responses = [
            "I'm here to help you learn about Sahibpreet Singh's professional background and experience. Could you please ask something related to his work, skills, or career?",
            
            "I'd be happy to share information about Sahibpreet Singh's expertise, projects, or professional journey. What would you like to know about his background?",
            
            "My purpose is to provide information about Sahibpreet Singh's professional profile. Please feel free to ask about his experience, skills, education, or projects.",
            
            "I'm designed to discuss Sahibpreet Singh's career, technical expertise, and professional achievements. How can I help you learn more about his background?",
            
            "I focus on sharing details about Sahibpreet Singh's professional experience and capabilities. What specific aspect of his background interests you?"
        ]
    
    def check_query_appropriateness(self, query: str) -> Dict[str, Any]:
        """
        Check if a query is appropriate and on-topic
        
        Args:
            query: User's input query
            
        Returns:
            Dictionary with appropriateness check results
        """
        query_lower = query.lower().strip()
        
        # Check for empty or very short queries
        if len(query_lower) < 3:
            return {
                'is_appropriate': False,
                'reason': 'too_short',
                'response': "Could you please provide a more specific question about Sahibpreet Singh?"
            }
        
        # Check for forbidden patterns
        for pattern in self.forbidden_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return {
                    'is_appropriate': False,
                    'reason': 'off_topic',
                    'response': self._get_polite_decline_response()
                }
        
        # Check if query contains allowed topics
        topic_score = self._calculate_topic_relevance(query_lower)
        
        if topic_score < 0.1:  # Very low relevance to Sahibpreet
            return {
                'is_appropriate': False,
                'reason': 'unrelated',
                'response': self._get_polite_decline_response()
            }
        
        # Query seems appropriate
        return {
            'is_appropriate': True,
            'topic_score': topic_score,
            'detected_topics': self._detect_query_topics(query_lower)
        }
    
    def _calculate_topic_relevance(self, query: str) -> float:
        """Calculate how relevant the query is to allowed topics"""
        total_score = 0.0
        word_count = len(query.split())
        
        for topic_category, keywords in self.allowed_topics.items():
            for keyword in keywords:
                if keyword in query:
                    # Give higher score for exact matches
                    if keyword == 'sahibpreet' or keyword == 'singh':
                        total_score += 2.0
                    else:
                        total_score += 1.0
        
        # Normalize by query length
        if word_count > 0:
            return min(total_score / word_count, 1.0)
        return 0.0
    
    def _detect_query_topics(self, query: str) -> List[str]:
        """Detect which topics the query is asking about"""
        detected_topics = []
        
        for topic_category, keywords in self.allowed_topics.items():
            for keyword in keywords:
                if keyword in query:
                    if topic_category not in detected_topics:
                        detected_topics.append(topic_category)
                    break
        
        return detected_topics
    
    def _get_polite_decline_response(self) -> str:
        """Get a random polite decline response"""
        import random
        return random.choice(self.polite_responses)
    
    def enhance_response_with_context(self, response: str, query: str) -> str:
        """Add professional context to responses"""
        
        # If the response seems too generic, add context
        if len(response) < 50 or 'sahibpreet' not in response.lower():
            context_prefix = "Regarding Sahibpreet Singh: "
            response = context_prefix + response
        
        # Add a professional closing for longer responses
        if len(response) > 200:
            closing_options = [
                "\n\nIs there anything else you'd like to know about Sahibpreet's background?",
                "\n\nFeel free to ask more questions about his experience or projects.",
                "\n\nWould you like to know more about any specific aspect of his work?",
            ]
            import random
            response += random.choice(closing_options)
        
        return response
    
    def filter_sensitive_information(self, response: str) -> str:
        """Remove any potentially sensitive information from responses"""
        
        # Pattern to detect and mask potential sensitive data
        sensitive_patterns = [
            (r'\b\d{3}-\d{3}-\d{4}\b', '[Phone Number]'),  # Phone numbers
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[Email Address]'),  # Emails (though we want to keep his public email)
            (r'\b\d{1,5}\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr)\b', '[Address]'),  # Street addresses
        ]
        
        filtered_response = response
        for pattern, replacement in sensitive_patterns:
            # Don't filter his public professional email
            if 'ss9334931@gmail.com' not in response:
                filtered_response = re.sub(pattern, replacement, filtered_response, flags=re.IGNORECASE)
        
        return filtered_response
    
    def validate_response_quality(self, response: str, query: str) -> Dict[str, Any]:
        """Validate that the response meets quality standards"""
        
        issues = []
        
        # Check response length
        if len(response) < 20:
            issues.append("Response too short")
        elif len(response) > 1000:
            issues.append("Response too long")
        
        # Check if response is relevant to Sahibpreet
        if 'sahibpreet' not in response.lower() and 'singh' not in response.lower():
            issues.append("Response doesn't mention Sahibpreet Singh")
        
        # Check for placeholder text
        if '[' in response and ']' in response:
            issues.append("Response contains placeholder text")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'response': response
        }