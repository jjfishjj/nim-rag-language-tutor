"""LinguaRAG — a retrieval-augmented language tutor built on NVIDIA NIM APIs.

Pipeline: user question -> NV-EmbedQA embedding -> cosine top-k over the
grammar-note corpus -> Nemotron chat completion grounded on retrieved notes.

Requires NVIDIA_API_KEY (free at https://build.nvidia.com).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

BASE_URL = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"
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
    return OpenAI(base_url=BASE_URL, api_key=api_key)


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
    np.savez(INDEX_PATH, vectors=vecs)
    print(f"Indexed {len(corpus)} notes -> {INDEX_PATH}")


def retrieve(client: OpenAI, query: str, k: int = 3) -> list[dict]:
    corpus = load_corpus()
    vectors = np.load(INDEX_PATH)["vectors"]
    q = embed(client, [query], input_type="query")[0]
    sims = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q))
    top = np.argsort(sims)[::-1][:k]
    return [{**corpus[i], "score": float(sims[i])} for i in top]


def answer(client: OpenAI, query: str) -> str:
    notes = retrieve(client, query)
    context = "\n\n".join(f"[{n['topic']}] {n['text']}" for n in notes)
    print("--- retrieved notes ---")
    for n in notes:
        print(f"  {n['score']:.3f}  {n['topic']}")
    print("-----------------------")
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Reference notes:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.5,
        max_tokens=800,
    )
    return resp.choices[0].message.content


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
