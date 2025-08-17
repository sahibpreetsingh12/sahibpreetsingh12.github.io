# Sahibpreet Singh - Resume Chatbot

A professional AI-powered chatbot that answers questions about Sahibpreet Singh using RAG (Retrieval-Augmented Generation) on his resume.

## Features

- **Semantic Search**: Uses sentence transformers for intelligent content retrieval
- **Guardrails**: Prevents off-topic questions and maintains professional responses
- **Section-based RAG**: Intelligent chunking based on resume sections
- **Professional Responses**: Maintains formal, helpful tone
- **Conversation Memory**: Tracks conversation context
- **Fallback Support**: Works even without ML libraries installed

## Setup

1. **Install Dependencies**
```bash
cd chatbot
pip install -r requirements.txt
```

2. **Run the Chatbot API**
```bash
python app.py
```

The API will be available at: `http://localhost:8000`

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /chat` - Main chat endpoint
- `GET /resume/summary` - Resume summary
- `GET /chat/suggestions` - Suggested questions

## Example Usage

```bash
# Test the chatbot
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "What is Sahibpreet's experience with GenAI?"}'
```

## Guardrails

The chatbot automatically filters:
- Off-topic questions unrelated to Sahibpreet Singh
- Requests for other people's information
- Inappropriate or personal questions
- General technical help requests

All responses are professional and focused on Sahibpreet's background.

## Architecture

- **FastAPI**: REST API framework
- **Resume Processor**: Extracts and chunks resume content
- **Vector Store**: Semantic search using sentence transformers
- **Guardrails Engine**: Content filtering and response validation
- **Chatbot Engine**: Main conversation logic with RAG

## Integration

Add the chatbot widget to your website by including the JavaScript widget (see `widget.js`).