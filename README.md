
## Architecture

- **FastAPI** web service with `/health` and `/chat` endpoints  
- **Claude claude-sonnet-4-20250514** (Anthropic) as the LLM backbone  
- **TF-IDF retrieval** (no GPU required) for RAG context injection  
- **80-assessment catalog** scraped and curated from the SHL product catalog  

## API

### `GET /health`
```json
{"status": "ok"}
```

### `POST /chat`
Request:
```json
{
  "messages": [
    {"role": "user", "content": "I need to hire a Java developer, about 4 years experience."},
    {"role": "assistant", "content": "Happy to help! What seniority level and what skills should we test?"},
    {"role": "user", "content": "Mid-level, needs to work with stakeholders too."}
  ]
}
```

Response:
```json
{
  "reply": "Here are 5 assessments for a mid-level Java developer who works with stakeholders.",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

## Local Development

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn main:app --reload
```

## Deploy to Render

1. Fork/push this repo to GitHub  
2. Create a new Render **Web Service**, connect the repo  
3. Set environment variable `ANTHROPIC_API_KEY`  
4. Render picks up `render.yaml` automatically  

## Deploy to Railway

```bash
railway up
railway variables set ANTHROPIC_API_KEY=sk-ant-...
```

## Run Evaluations

```bash
python evaluate.py --url https://your-deployed-url.onrender.com
```
