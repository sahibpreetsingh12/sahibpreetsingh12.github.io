#!/usr/bin/env python3
"""
Simple Chatbot - Direct Groq LLM call with resume markdown
"""

import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Load resume content
with open('resume.md', 'r') as f:
    resume_content = f.read()

def check_guardrails(question):
    """Simple guardrails check"""
    question_lower = question.lower().strip()
    
    # Block off-topic questions
    blocked_keywords = [
        'weather', 'politics', 'religion', 'other people', 'someone else',
        'another person', 'family', 'relationship', 'dating', 'marriage',
        'illegal', 'harmful', 'dangerous', 'violence'
    ]
    
    if any(keyword in question_lower for keyword in blocked_keywords):
        return False, "I can only answer questions about Sahibpreet Singh's professional background, skills, and experience."
    
    return True, None

def ask_chatbot(question):
    """Main function to ask chatbot a question"""
    
    # Step 1: Check guardrails first
    is_valid, blocked_message = check_guardrails(question)
    if not is_valid:
        return blocked_message
    
    # Step 2: Call Groq LLM with resume
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": f"""You are an AI assistant representing Sahibpreet Singh. Answer questions about his professional background based on his resume below. Be professional and concise.

RESUME:
{resume_content}

Only answer questions about Sahibpreet Singh based on this resume information."""
                },
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Sorry, I'm having trouble processing your question right now. Error: {e}"

# Interactive mode
if __name__ == "__main__":
    print("🤖 Sahibpreet Singh Chatbot")
    print("Ask me anything about Sahibpreet's professional background!")
    print("Type 'quit' to exit\n")
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break
        
        if not question:
            continue
            
        answer = ask_chatbot(question)
        print(f"\nChatbot: {answer}\n")