import os
from dotenv import load_dotenv

load_dotenv()

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL, MOCK_MODE


def build_prompt(query: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        f"نظر {i+1}:\n{c.get('body', '')}"
        for i, c in enumerate(contexts)
    )
    return (
        f"بر اساس نظرات زیر به سوال کاربر پاسخ بده.\n\n"
        f"نظرات:\n{context_text}\n\n"
        f"سوال: {query}\n\n"
        f"پاسخ:"
    )


def generate_answer(query: str, contexts: list[dict]) -> str:
    if MOCK_MODE:
        first_body = contexts[0].get("body", "")[:100] if contexts else "خالی"
        return f"[MOCK] {len(contexts)} نظر یافت شد. اولین نظر: {first_body}"

    from openai import OpenAI

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL or None,
    )
    prompt = build_prompt(query, contexts)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
