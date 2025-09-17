from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import pdfplumber
import fitz  # PyMuPDF

from ..models import UploadedDocument


def extract_text_from_file(path: Path) -> str:
    text = ""
    try:
        if path.suffix.lower() == '.pdf':
            # First try pdfplumber
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages[:5]:
                    text += page.extract_text() or ""
            if not text.strip():
                # Fallback to PyMuPDF
                with fitz.open(path) as doc:
                    for page in doc[:5]:
                        text += page.get_text()
        else:
            text = path.read_text(encoding='utf-8', errors='ignore')[:5000]
    except Exception:
        text = ""
    return (text or "").strip()


def process_documents(documents: List[UploadedDocument], prompt: str) -> Tuple[list, list, list]:
    enriched = []
    for d in documents:
        path = Path(d.file.path)
        text = extract_text_from_file(path)[:2000]
        d.text_excerpt = text[:400]
        d.save(update_fields=["text_excerpt"])
        enriched.append({
            "id": d.id,
            "name": d.original_name,
            "length": len(text),
            "has_deadline": "deadline" in text.lower(),
            "has_exam": "exam" in text.lower() or "quiz" in text.lower(),
        })

    enriched.sort(key=lambda x: (not x["has_exam"], not x["has_deadline"], -x["length"]))

    ranking = [{"document_id": e["id"], "priority": i + 1} for i, e in enumerate(enriched)]
    summaries = [{"document_id": e["id"], "summary": f"Auto-summary for {e['name']} ({e['length']} chars)."} for e in enriched]

    quiz = []
    for e in enriched[:3]:
        quiz.append({
            "document_id": e["id"],
            "questions": [
                {
                    "type": "mcq",
                    "question": f"What is a key topic in {e['name']}?",
                    "options": ["Definition", "Example", "Deadline", "All of the above"],
                    "answer": 3,
                },
                {
                    "type": "short",
                    "question": f"Summarize {e['name']} in one sentence.",
                },
            ],
        })

    return ranking, summaries, quiz

