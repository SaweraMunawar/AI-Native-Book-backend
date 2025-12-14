# Physical AI Textbook - RAG Backend

FastAPI backend providing RAG (Retrieval-Augmented Generation) capabilities for the Physical AI textbook chatbot.

## Features

- **Vector Search**: Qdrant-powered semantic search over textbook content
- **LLM Generation**: Groq Llama 3 for response generation
- **Confidence Scoring**: High/Medium/Low confidence based on retrieval scores
- **Rate Limiting**: 100 requests/hour per IP
- **Health Checks**: Dependency monitoring endpoint

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### 2. External Services

You need accounts for:

1. **Qdrant Cloud** - https://cloud.qdrant.io
2. **Groq** - https://console.groq.com
3. **Neon PostgreSQL** (optional) - https://neon.tech

### 3. Initialize Services

```bash
# Create Qdrant collection
python scripts/setup_qdrant.py

# Ingest textbook content
python scripts/ingest.py --docs-path ../docs
```

### 4. Run Server

```bash
uvicorn src.main:app --reload --port 8000
```

## API Endpoints

### Health Check

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "dependencies": {
    "qdrant": "up",
    "groq": "up",
    "neon": "up"
  }
}
```

### Send Chat Message

```http
POST /chat
Content-Type: application/json

{
  "message": "What is ROS 2?",
  "session_id": "optional-uuid"
}
```

Response:
```json
{
  "message_id": "uuid",
  "session_id": "uuid",
  "answer": "ROS 2 (Robot Operating System 2) is...",
  "confidence": "high",
  "sources": [
    {
      "chapter_slug": "ros2-fundamentals",
      "chapter_title": "ROS 2 Fundamentals",
      "excerpt": "...",
      "score": 0.85
    }
  ],
  "disclaimer": null
}
```

### Contextual Chat (with selected text)

```http
POST /chat/context
Content-Type: application/json

{
  "message": "Explain this in simpler terms",
  "selected_text": "ROS 2 nodes communicate via topics...",
  "chapter_slug": "ros2-fundamentals",
  "session_id": "optional-uuid"
}
```

## Project Structure

```
backend/
├── src/
│   ├── api/
│   │   ├── chat.py        # Chat endpoints
│   │   └── health.py      # Health check
│   ├── services/
│   │   ├── embeddings.py  # MiniLM embeddings
│   │   ├── retrieval.py   # Qdrant search
│   │   └── generation.py  # Groq LLM
│   ├── models/
│   │   └── schemas.py     # Pydantic models
│   ├── config.py          # Settings
│   └── main.py            # FastAPI app
├── scripts/
│   ├── ingest.py          # Content ingestion
│   ├── setup_qdrant.py    # Qdrant setup
│   └── setup_neon.sql     # PostgreSQL schema
├── tests/
├── requirements.txt
└── .env.example
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| QDRANT_URL | Qdrant Cloud URL | - |
| QDRANT_API_KEY | Qdrant API key | - |
| GROQ_API_KEY | Groq API key | - |
| DATABASE_URL | Neon PostgreSQL URL | - |
| CORS_ORIGINS | Allowed origins | http://localhost:3000 |
| RATE_LIMIT_REQUESTS | Requests per window | 100 |
| RATE_LIMIT_WINDOW_SECONDS | Rate limit window | 3600 |

## Architecture

```
Request → Rate Limiter → Chat Endpoint
                              │
                    ┌─────────┴─────────┐
                    │                   │
              Embed Query          Get Context
                    │                   │
                    ▼                   ▼
               Qdrant Search      Selected Text
                    │                   │
                    └─────────┬─────────┘
                              │
                              ▼
                    Confidence Scoring
                              │
                    ┌─────────┴─────────┐
                    │                   │
                 High/Med            Low
                    │                   │
                    ▼                   ▼
              Groq Generate     Return "Not Found"
                    │
                    ▼
              Format Response
                    │
                    ▼
              Return with Sources
```

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Type Checking

```bash
mypy src/
```

### Linting

```bash
ruff check src/
ruff format src/
```

## Deployment

### Railway

1. Connect GitHub repository
2. Add environment variables
3. Deploy

### Vercel (Serverless)

Create `vercel.json`:
```json
{
  "builds": [
    { "src": "src/main.py", "use": "@vercel/python" }
  ],
  "routes": [
    { "src": "/(.*)", "dest": "src/main.py" }
  ]
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Qdrant connection failed | Check QDRANT_URL and API key |
| Empty responses | Run ingest.py to populate vectors |
| Rate limit errors | Wait for window reset (1 hour) |
| Slow responses | Check Groq rate limits |
