"""Hugging Face Space UI for the NVIDIA NIM-backed language tutor."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rag_tutor import answer_with_metrics, get_client, load_corpus  # noqa: E402

OFFLINE_KEYWORDS = {
    "英文現在完成式": ("完成式", "過去式", "have", "has", "since", "for"),
    "英文假設語氣": ("假設", "if", "would", "were"),
    "日文て形": ("て形", "動詞變化", "てください"),
    "日文助詞は與が": ("助詞", "は", "が", "主題", "主語"),
    "英文冠詞用法": ("冠詞", "a ", "an ", "the "),
    "西班牙文動詞變位基礎": ("西班牙", "變位", "hablar", "comer", "vivir"),
    "語言學習方法：間隔重複": ("間隔重複", "記憶", "單字", "複習"),
    "英文關係子句": ("關係子句", "who", "which", "that"),
    "日文敬語基礎": ("敬語", "尊敬語", "謙讓語", "丁寧語"),
    "英文分詞構句": ("分詞", "懸垂", "participle"),
}


def offline_answer(question: str) -> tuple[str, list[dict]]:
    question = question.strip()
    if not question:
        raise gr.Error("請先輸入問題。")
    if len(question) > 500:
        raise gr.Error("問題請限制在 500 字以內。")
    corpus = load_corpus()
    normalized = question.lower()
    terms = {char for char in normalized if char.strip()}
    def local_score(item: dict) -> int:
        keyword_hits = sum(4 for keyword in OFFLINE_KEYWORDS[item["topic"]] if keyword.lower() in normalized)
        character_hits = len(terms.intersection(set((item["topic"] + item["text"]).lower())))
        return keyword_hits + character_hits
    ranked = sorted(
        corpus,
        key=local_score,
        reverse=True,
    )[:3]
    notes = [{**item, "score": round(max(0.05, 0.72 - index * 0.17), 2)} for index, item in enumerate(ranked)]
    best = notes[0]
    answer = (
        f"### {best['topic']}\n\n{best['text']}\n\n"
        "> Offline preview：此回答用本地 deterministic retrieval，未呼叫 NVIDIA NIM。"
    )
    return answer, notes


def format_sources(notes: list[dict]) -> str:
    return "\n\n".join(
        f"**{index}. {note['topic']}** · score `{note['score']:.3f}`\n\n{note['text']}"
        for index, note in enumerate(notes, 1)
    )


def tutor(question: str):
    if os.environ.get("NVIDIA_API_KEY"):
        try:
            result = answer_with_metrics(get_client(), question)
            return (
                result["answer"],
                format_sources(result["notes"]),
                result["metrics"],
                "🟢 **LIVE NIM** · API key stays in the Hugging Face Space secret store.",
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        except Exception:
            raise gr.Error("NVIDIA NIM request failed. The secret, endpoint quota, or model availability may need review.")
    answer, notes = offline_answer(question)
    metrics = {
        "mode": "offline_demo",
        "latency": "not measured",
        "tokens": "not available",
        "estimated_cost_usd": None,
        "cost_note": "No NIM request; add NVIDIA_API_KEY as a Space secret",
    }
    return answer, format_sources(notes), metrics, "🟠 **OFFLINE DEMO** · Owner has not configured the `NVIDIA_API_KEY` Space secret."


CSS = """
.gradio-container {max-width: 1180px !important; margin: auto !important;}
.hero {border: 1px solid #30392f; border-radius: 18px; padding: 24px; background: linear-gradient(145deg,#111810,#090b09);}
.hero h1 {font-size: clamp(34px, 6vw, 64px) !important; letter-spacing: -.04em; margin: 0 0 8px;}
.evidence {border-left: 3px solid #76b900 !important;}
footer {display:none !important;}
"""

with gr.Blocks(title="NIM RAG Language Tutor", css=CSS, theme=gr.themes.Base(primary_hue="lime", neutral_hue="slate")) as demo:
    gr.HTML("""
    <section class="hero">
      <div style="color:#9ddb43;font-weight:800;letter-spacing:.12em;font-size:12px">NVIDIA NIM · LIVE-BACKEND READY</div>
      <h1>Ask a language question.<br>Inspect the evidence.</h1>
      <p>Server-side NIM RAG with visible sources, latency, token usage, and explicit cost assumptions.</p>
    </section>
    """)
    mode = gr.Markdown("Checking backend mode…")
    with gr.Row():
        with gr.Column(scale=5):
            question = gr.Textbox(label="你的語言問題", value="現在完成式跟過去式差在哪？", lines=2, max_lines=4)
            with gr.Row():
                submit = gr.Button("檢索並回答", variant="primary")
                clear = gr.ClearButton([question])
            gr.Examples(
                examples=[["日文的は和が怎麼分？"], ["英文假設語氣怎麼用？"], ["西班牙文規則動詞如何變位？"]],
                inputs=question,
            )
        with gr.Column(scale=7):
            answer_box = gr.Markdown("答案會顯示在這裡。", elem_classes="evidence")
    with gr.Row():
        sources = gr.Markdown("檢索證據會顯示在這裡。")
        metrics = gr.JSON(label="Latency · tokens · cost contract")
    submit.click(tutor, question, [answer_box, sources, metrics, mode])
    question.submit(tutor, question, [answer_box, sources, metrics, mode])
    demo.load(
        lambda: "🟢 **LIVE NIM** · Secret configured." if os.environ.get("NVIDIA_API_KEY") else "🟠 **OFFLINE DEMO** · Add `NVIDIA_API_KEY` in Space Settings → Variables and secrets.",
        outputs=mode,
    )
    gr.Markdown("""
    ---
    **Security boundary:** the browser never receives the NVIDIA API key. The Space backend reads it from an environment secret. Questions are capped at 500 characters; provider errors are sanitized. Monetary cost is shown only when the owner configures current per-token rates via `NIM_INPUT_USD_PER_MTOK` and `NIM_OUTPUT_USD_PER_MTOK`.

    [GitHub source](https://github.com/jjfishjj/nim-rag-language-tutor) · [NVIDIA API Catalog](https://build.nvidia.com/)
    """)

if __name__ == "__main__":
    demo.launch()
