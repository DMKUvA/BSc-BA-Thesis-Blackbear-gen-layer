from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

print("OPENAI key loaded:", bool(os.getenv("OPENAI_API_KEY")))


def generate_if_configured(messages: List[Dict[str, str]]) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package is not installed. Run: pip install openai"
        ) from exc

    model = os.getenv("OPENAI_MODEL", "gpt-4.1")
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
    )

    content = response.choices[0].message.content
    if not content or not isinstance(content, str):
        raise RuntimeError("Model returned empty content.")
    return content