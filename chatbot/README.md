# Sahibpreet Singh - Simple Resume Chatbot

A simple AI-powered chatbot that answers questions about Sahibpreet Singh using Groq's Llama-3.3-70B model.

## Features

- **Groq LLM Integration**: Uses Llama-3.3-70B for dynamic responses  
- **Simple Guardrails**: Prevents off-topic questions and maintains professional responses
- **Markdown Resume**: Reads resume data from simple markdown file
- **Professional Responses**: Maintains formal, helpful tone
- **Secure API Key Management**: Environment-based configuration

## Setup

1. **Install Dependencies**
```bash
cd chatbot
pip install -r requirements.txt
```

2. **Configure Groq API (Required for Dynamic Responses)**
```bash
# Copy the environment template
cp .env.example .env

# Edit .env and add your Groq API key
GROQ_API_KEY=your_actual_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

3. **Run the Chatbot API**
```bash
python app.py
```

The API will be available at: `http://localhost:8000`

**Note**: A Groq API key is required for the chatbot to work. Without it, the chatbot will return error messages.

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status  
- `POST /chat` - Main chat endpoint

## Example Usage

```bash
# Test the chatbot
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is Sahibpreet'\''s experience with GenAI?"}'
```

## Guardrails

Simple guardrails automatically filter:
- Off-topic questions unrelated to professional background
- Requests for other people's information
- Inappropriate content (weather, politics, personal life)

All responses are professional and focused on Sahibpreet's background.

## Architecture

- **FastAPI**: REST API framework
- **Groq LLM**: Llama-3.3-70B for dynamic response generation
- **Markdown Resume**: Simple resume.md file with all information
- **Simple Guardrails**: Basic keyword-based filtering

## Integration

Add the chatbot widget to your website by including the JavaScript widget (see `widget.js`).