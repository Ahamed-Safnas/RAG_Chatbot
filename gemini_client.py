import os
import google.generativeai as genai
from typing import List

MODEL_NAME = os.getenv("GENERATION_MODEL", "gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in .env")

genai.configure(api_key=GEMINI_API_KEY)
_model = genai.GenerativeModel(MODEL_NAME)

SYSTEM_INSTRUCTIONS = "You are a retrieval-augmented assistant. Answer strictly using provided context."

def build_prompt(query: str, context_snippets: List[str]) -> List[dict]:
    context = "\n\n---\n\n".join(context_snippets[:20])
    user_msg = f"Question:\n{query}\n\nContext:\n{context}"
    return [
        {"role": "user", "parts": [{"text": SYSTEM_INSTRUCTIONS}]},
        {"role": "user", "parts": [{"text": user_msg}]}
    ]

def answer_with_context(query: str, context_snippets: List[str]) -> str:
    msgs = build_prompt(query, context_snippets)
    resp = _model.generate_content(msgs)
    try:
        return resp.text.strip()
    except Exception:
        return "Sorry, I couldn't generate a response."
