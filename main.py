"""
SHL Assessment Recommender - FastAPI Service
Conversational agent that recommends SHL Individual Test Solutions
using RAG + Claude claude-sonnet-4-20250514.
"""

import json
import os
import re
import numpy as np
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import anthropic

# ── Config ──────────────────────────────────────────────────────────────────
CATALOG_PATH = Path(__file__).parent / "catalog.json"
MODEL = "claude-sonnet-4-20250514"
MAX_RECOMMENDATIONS = 10
TOP_K_RETRIEVAL = 15   # retrieve more, let LLM re-rank down to ≤10

# ── Load catalog ─────────────────────────────────────────────────────────────
with open(CATALOG_PATH) as f:
    CATALOG: list[dict] = json.load(f)

CATALOG_BY_NAME: dict[str, dict] = {a["name"].lower(): a for a in CATALOG}
VALID_URLS: set[str] = {a["url"] for a in CATALOG}

# ── Simple TF-style retrieval (no GPU / no heavy deps needed for Render) ────
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9#+.]+", text.lower())

def _build_index(catalog: list[dict]) -> tuple[list[list[str]], dict[str, int]]:
    docs = []
    idf_counts: dict[str, int] = {}
    for item in catalog:
        tokens = _tokenize(
            f"{item['name']} {item['description']} "
            f"{item['test_type_label']} {' '.join(item['job_levels'])} "
            f"{' '.join(item['languages'])}"
        )
        docs.append(tokens)
        for t in set(tokens):
            idf_counts[t] = idf_counts.get(t, 0) + 1
    return docs, idf_counts

DOC_TOKENS, IDF_COUNTS = _build_index(CATALOG)
N_DOCS = len(CATALOG)

def _score(query: str) -> list[tuple[int, float]]:
    q_tokens = _tokenize(query)
    scores = []
    for idx, doc_tokens in enumerate(DOC_TOKENS):
        doc_freq: dict[str, int] = {}
        for t in doc_tokens:
            doc_freq[t] = doc_freq.get(t, 0) + 1
        score = 0.0
        for qt in q_tokens:
            if qt in doc_freq:
                tf = doc_freq[qt] / len(doc_tokens)
                idf = np.log((N_DOCS + 1) / (IDF_COUNTS.get(qt, 0) + 1)) + 1
                score += tf * idf
        scores.append((idx, score))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def retrieve(query: str, k: int = TOP_K_RETRIEVAL) -> list[dict]:
    scored = _score(query)
    return [CATALOG[idx] for idx, score in scored[:k] if score > 0]

# ── Format catalog for context ───────────────────────────────────────────────
def format_assessment(a: dict) -> str:
    return (
        f"Name: {a['name']}\n"
        f"URL: {a['url']}\n"
        f"Test Type: {a['test_type']} ({a['test_type_label']})\n"
        f"Remote Testing: {'Yes' if a['remote_testing'] else 'No'}\n"
        f"Adaptive/IRT: {'Yes' if a['adaptive_irt'] else 'No'}\n"
        f"Duration: {a.get('duration_minutes', 'N/A')} minutes\n"
        f"Job Levels: {', '.join(a['job_levels'])}\n"
        f"Languages: {', '.join(a['languages'][:5])}{'...' if len(a['languages']) > 5 else ''}\n"
        f"Description: {a['description']}"
    )

FULL_CATALOG_SUMMARY = "\n\n---\n\n".join(format_assessment(a) for a in CATALOG)

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are an SHL Assessment Recommender assistant. Your job is to help hiring managers and recruiters find the right SHL Individual Test Solutions from the official SHL catalog.

## Your responsibilities
1. **Clarify** vague queries before recommending. "I need an assessment" is not enough — ask about role, job level, skills needed.
2. **Recommend** 1–10 assessments once you have enough context (role, seniority, what to measure). Always include assessment name and catalog URL.
3. **Refine** recommendations when the user adds constraints mid-conversation without starting over.
4. **Compare** assessments using only the catalog data provided — never use prior knowledge to invent details.

## Strict rules
- ONLY recommend assessments that appear in the catalog below. Every URL you return MUST be from this catalog.
- Do NOT discuss general hiring advice, legal questions, compensation, or anything unrelated to SHL assessments.
- Do NOT recommend Pre-packaged Job Solutions — Individual Test Solutions only.
- If asked to do something off-topic, politely decline and redirect.
- Do NOT hallucinate assessment names, durations, or features.

## Test type codes
- A = Ability & Aptitude (cognitive, reasoning)
- B = Biodata & Situational Judgement
- C = Competencies
- D = Development & 360
- E = Assessment Exercises
- K = Knowledge & Skills (technical/domain knowledge)
- P = Personality & Behavior
- S = Simulations (coding, work simulations)

## Response format instructions
When you are ready to provide recommendations, you MUST end your reply with a JSON block (and ONLY one such block) in this exact format:

```json
{{
  "recommendations": [
    {{"name": "Assessment Name", "url": "https://www.shl.com/...", "test_type": "K"}},
    ...
  ],
  "end_of_conversation": false
}}
```

Set `end_of_conversation` to `true` only when the user has confirmed they are satisfied with the shortlist.
When still gathering context or refusing, output:
```json
{{
  "recommendations": [],
  "end_of_conversation": false
}}
```

## Full SHL Individual Test Solutions Catalog

{FULL_CATALOG_SUMMARY}
"""

# ── Pydantic models ──────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v

class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def non_empty(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages cannot be empty")
        if len(v) > 20:
            raise ValueError("Too many messages — conversation capped at 20 turns")
        return v

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

# ── Claude client ─────────────────────────────────────────────────────────────
_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def _extract_json_block(text: str) -> Optional[dict]:
    """Extract the last ```json ... ``` block from the model reply."""
    pattern = r"```json\s*(\{.*?\})\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return None

def _validate_recommendations(recs: list[dict]) -> list[Recommendation]:
    """Filter out any recommendations not in the catalog."""
    valid = []
    for r in recs:
        url = r.get("url", "")
        name = r.get("name", "")
        test_type = r.get("test_type", "")
        if url in VALID_URLS:
            valid.append(Recommendation(name=name, url=url, test_type=test_type))
    return valid[:MAX_RECOMMENDATIONS]

def _build_rag_context(messages: list[Message]) -> str:
    """Build a retrieval-augmented context from conversation history."""
    # Collect all user text for query
    user_text = " ".join(m.content for m in messages if m.role == "user")
    retrieved = retrieve(user_text, k=TOP_K_RETRIEVAL)
    if not retrieved:
        return ""
    context = "\n\n---\n\n".join(format_assessment(a) for a in retrieved)
    return f"\n\n## Retrieved assessments most relevant to this conversation\n\n{context}"

def call_claude(messages: list[Message]) -> tuple[str, list[Recommendation], bool]:
    """Call Claude claude-sonnet-4-20250514 with RAG-augmented system prompt."""
    rag_context = _build_rag_context(messages)
    system = SYSTEM_PROMPT + rag_context

    api_messages = [{"role": m.role, "content": m.content} for m in messages]

    response = _client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=system,
        messages=api_messages,
    )

    full_text: str = response.content[0].text

    # Strip the JSON block from the visible reply
    reply_text = re.sub(r"```json\s*\{.*?\}\s*```", "", full_text, flags=re.DOTALL).strip()

    # Parse structured output
    parsed = _extract_json_block(full_text)
    recs: list[Recommendation] = []
    end_of_conv = False

    if parsed:
        raw_recs = parsed.get("recommendations", [])
        recs = _validate_recommendations(raw_recs)
        end_of_conv = bool(parsed.get("end_of_conversation", False))

    return reply_text, recs, end_of_conv

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL Individual Test Solutions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        reply, recommendations, end_of_conversation = call_claude(request.messages)
        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=end_of_conversation,
        )
    except anthropic.APIConnectionError as e:
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please retry.")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e.message))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}: {e}")
