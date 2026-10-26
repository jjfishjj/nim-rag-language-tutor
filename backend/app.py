"""Rate-limited Vercel backend for the public LinguaRAG demo.

The NVIDIA API key is read only from the server environment. The browser sends
questions to this service and never receives the credential.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from threading import Lock

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = os.environ.get("NIM_EMBED_MODEL", "nvidia/nemotron-3.5-embed-1b")
CHAT_MODEL = os.environ.get("NIM_CHAT_MODEL", "nvidia/nemotron-3-super-120b-a12b")
ALLOWED_ORIGINS = [
    item.strip()
    for item in os.environ.get(
        "ALLOWED_ORIGINS",
        "https://junbin666-nim-rag-language-tutor.static.hf.space,https://huggingface.co",
    ).split(",")
    if item.strip()
]

CORPUS = [
    {"topic": "英文現在完成式", "text": "現在完成式 (have/has + p.p.) 表達過去發生且與現在有關的事件，常搭配 already、yet、since、for；過去簡單式只陳述已完成的過去事實。"},
    {"topic": "英文假設語氣", "text": "與現在事實相反用 If + 過去式, would + 原形；與過去事實相反用 If + had p.p., would have p.p.。正式語體常用 If I were you。"},
    {"topic": "日文て形", "text": "て形可連接動作、提出請求、構成進行式與表示許可；五段動詞依字尾分組變化。"},
    {"topic": "日文助詞は與が", "text": "は標記主題或已知資訊；が標記主語、新資訊或強調。疑問詞作主語時使用が。"},
    {"topic": "英文冠詞用法", "text": "a/an 用於首次提及的可數單數；the 用於特定、獨一無二、序數或最高級；不可數與複數泛指通常不加冠詞。"},
    {"topic": "西班牙文動詞變位基礎", "text": "規則動詞分 -ar、-er、-ir；動詞字尾表達人稱，所以主詞代名詞通常可省略。"},
    {"topic": "語言學習方法：間隔重複", "text": "在即將遺忘時複習並逐步拉長間隔，搭配主動回想，通常比反覆閱讀更有效。"},
    {"topic": "英文關係子句", "text": "who 指人、which 指物、that 可指人或物；非限定用法加逗號且不可使用 that。"},
    {"topic": "日文敬語基礎", "text": "敬語分尊敬語、謙讓語與丁寧語；商務場合對外提及己方人員時使用謙讓表達。"},
    {"topic": "英文分詞構句", "text": "分詞構句省略連接詞與重複主詞；分詞的邏輯主詞必須和主句主詞一致，以避免懸垂分詞。"},
]

SYSTEM_PROMPT = """你是一位親切的繁體中文／English 雙語語言家教。
只能依據提供的筆記回答，並用 [來源 N] 標示依據。若筆記不足，必須明確拒絕猜測。
回答最後提供一題簡短練習。"""

app = FastAPI(title="LinguaRAG NIM Backend", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class TutorRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


_requests: dict[str, deque[float]] = defaultdict(deque)
_rate_lock = Lock()
_passage_vectors: np.ndarray | None = None
_vector_lock = Lock()


def _client() -> OpenAI:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key or os.environ.get("NIM_LIVE_ENABLED") != "1":
        raise HTTPException(status_code=503, detail="Live NIM is not enabled.")
    return OpenAI(base_url=BASE_URL, api_key=key, timeout=45.0)


def _check_rate_limit(request: Request) -> None:
    forwarded = request.headers.get("x-forwarded-for", "")
    identity = forwarded.split(",", 1)[0].strip() or (request.client.host if request.client else "unknown")
    now = time.time()
    with _rate_lock:
        recent = _requests[identity]
        while recent and recent[0] < now - 60:
            recent.popleft()
        if len(recent) >= 5:
            raise HTTPException(status_code=429, detail="Rate limit reached. Try again in one minute.")
        recent.append(now)


def _embed(client: OpenAI, texts: list[str], input_type: str) -> np.ndarray:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    return np.asarray([item.embedding for item in response.data], dtype=np.float32)


def _passages(client: OpenAI) -> np.ndarray:
    global _passage_vectors
    if _passage_vectors is None:
        with _vector_lock:
            if _passage_vectors is None:
                _passage_vectors = _embed(client, [item["text"] for item in CORPUS], "passage")
    return _passage_vectors


def _retrieve(client: OpenAI, question: str) -> list[dict]:
    passages = _passages(client)
    query = _embed(client, [question], "query")[0]
    denominator = np.linalg.norm(passages, axis=1) * max(np.linalg.norm(query), 1e-12)
    scores = passages @ query / np.maximum(denominator, 1e-12)
    indices = np.argsort(scores)[::-1][:3]
    return [{**CORPUS[index], "score": round(float(scores[index]), 4)} for index in indices]


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "live": bool(os.environ.get("NVIDIA_API_KEY")) and os.environ.get("NIM_LIVE_ENABLED") == "1",
        "embedding_model": EMBED_MODEL,
        "chat_model": CHAT_MODEL,
    }


@app.post("/api/tutor")
def tutor(payload: TutorRequest, request: Request) -> dict:
    _check_rate_limit(request)
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be empty.")
    client = _client()
    started = time.perf_counter()
    try:
        retrieval_started = time.perf_counter()
        notes = _retrieve(client, question)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        context = "\n\n".join(
            f"[來源 {index}] {note['topic']}: {note['text']}"
            for index, note in enumerate(notes, 1)
        )
        generation_started = time.perf_counter()
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"參考筆記：\n{context}\n\n學習者問題：{question}"},
            ],
            temperature=0.2,
            max_tokens=450,
        )
        generation_ms = (time.perf_counter() - generation_started) * 1000
    except HTTPException:
        raise
    except Exception as exc:
        print(f"NIM provider error: {type(exc).__name__}")
        raise HTTPException(status_code=502, detail="NVIDIA NIM request failed.") from exc

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    input_rate = float(os.environ.get("NIM_INPUT_USD_PER_MTOK", "0") or 0)
    output_rate = float(os.environ.get("NIM_OUTPUT_USD_PER_MTOK", "0") or 0)
    cost = None
    if input_rate or output_rate:
        cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    return {
        "mode": "live_nim",
        "answer": response.choices[0].message.content,
        "notes": notes,
        "metrics": {
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(generation_ms, 1),
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(cost, 6) if cost is not None else None,
            "cost_note": "Configured estimate" if cost is not None else "N/A — no pricing assumption configured",
            "embedding_model": EMBED_MODEL,
            "chat_model": CHAT_MODEL,
        },
    }
