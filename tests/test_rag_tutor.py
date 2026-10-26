import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import app


def test_offline_answer_returns_ranked_notes():
    answer, notes = app.offline_answer("日文的は和が怎麼分？")
    assert "Offline preview" in answer
    assert len(notes) == 3
    assert notes[0]["topic"] == "日文助詞は與が"


def test_offline_answer_rejects_empty_question():
    with pytest.raises(Exception):
        app.offline_answer("   ")


def test_tutor_uses_offline_mode_without_secret(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    _, sources, metrics, mode = app.tutor("英文冠詞怎麼用？")
    assert "英文冠詞用法" in sources
    assert metrics["mode"] == "offline_demo"
    assert "OFFLINE DEMO" in mode
