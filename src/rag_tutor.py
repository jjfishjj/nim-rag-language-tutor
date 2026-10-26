"""LinguaRAG — a retrieval-augmented language tutor built on NVIDIA NIM APIs.

Pipeline: user question -> NV-EmbedQA embedding -> cosine top-k over the
grammar-note corpus -> Nemotron chat completion grounded on retrieved notes.

Requires NVIDIA_API_KEY (free at https://build.nvidia.com).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = os.environ.get("NIM_EMBED_MODEL", "nvidia/nemotron-3.5-embed-1b")
CHAT_MODEL = os.environ.get("NIM_CHAT_MODEL", "nvidia/nemotron-3-super-120b-a12b")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_PATH = DATA_DIR / "index.npz"

SYSTEM_PROMPT = """You are a friendly bilingual (繁體中文/English) language tutor.
Answer the learner's question using ONLY the reference notes provided.
If the notes don't cover the question, say so honestly and suggest what to study.
Always end with one short practice exercise related to the question."""


def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("Set NVIDIA_API_KEY first — get a free key at https://build.nvidia.com")
    return OpenAI(base_url=BASE_URL, api_key=api_key, timeout=45.0)


def load_corpus() -> list[dict]:
    with open(DATA_DIR / "corpus.json", encoding="utf-8") as f:
        return json.load(f)


def embed(client: OpenAI, texts: list[str], input_type: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    return np.array([d.embedding for d in resp.data])


def build_index(client: OpenAI) -> None:
    corpus = load_corpus()
    vecs = embed(client, [doc["text"] for doc in corpus], input_type="passage")
    np.savez(INDEX_PATH, vectors=vecs, model=np.array(EMBED_MODEL))
    print(f"Indexed {len(corpus)} notes -> {INDEX_PATH}")


def retrieve(client: OpenAI, query: str, k: int = 3) -> list[dict]:
    corpus = load_corpus()
    payload = np.load(INDEX_PATH)
    stored_model = str(payload["model"]) if "model" in payload else "legacy"
    if stored_model != EMBED_MODEL:
        build_index(client)
        payload = np.load(INDEX_PATH)
    vectors = payload["vectors"]
    q = embed(client, [query], input_type="query")[0]
    sims = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    top = np.argsort(sims)[::-1][:k]
    return [{**corpus[i], "score": float(sims[i])} for i in top]


def answer_with_metrics(client: OpenAI, query: str) -> dict:
    query = query.strip()
    if not query:
        raise ValueError("Question cannot be empty.")
    if len(query) > 500:
        raise ValueError("Question must be 500 characters or fewer.")
    started = time.perf_counter()
    retrieval_started = time.perf_counter()
    notes = retrieve(client, query)
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
    context = "\n\n".join(f"[{n['topic']}] {n['text']}" for n in notes)
    generation_started = time.perf_counter()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Reference notes:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    generation_ms = (time.perf_counter() - generation_started) * 1000
    usage = getattr(resp, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    input_rate = float(os.environ.get("NIM_INPUT_USD_PER_MTOK", "0") or 0)
    output_rate = float(os.environ.get("NIM_OUTPUT_USD_PER_MTOK", "0") or 0)
    estimated_cost = None
    if input_rate or output_rate:
        estimated_cost = input_tokens * input_rate / 1_000_000 + output_tokens * output_rate / 1_000_000
    return {
        "answer": resp.choices[0].message.content,
        "notes": notes,
        "metrics": {
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(generation_ms, 1),
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(estimated_cost, 6) if estimated_cost is not None else None,
            "cost_note": "Configured estimate" if estimated_cost is not None else "N/A — pricing variables not configured",
            "embedding_model": EMBED_MODEL,
            "chat_model": CHAT_MODEL,
        },
    }


def answer(client: OpenAI, query: str) -> str:
    result = answer_with_metrics(client, query)
    print("--- retrieved notes ---")
    for note in result["notes"]:
        print(f"  {note['score']:.3f}  {note['topic']}")
    print("-----------------------")
    return result["answer"]


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG language tutor on NVIDIA NIM")
    parser.add_argument("--build-index", action="store_true", help="(re)build the embedding index")
    parser.add_argument("question", nargs="?", help="question to ask the tutor")
    args = parser.parse_args()

    client = get_client()
    if args.build_index:
        build_index(client)
        return
    if not args.question:
        parser.error("provide a question, or --build-index")
    if not INDEX_PATH.exists():
        build_index(client)
    print(answer(client, args.question))


if __name__ == "__main__":
    main()
