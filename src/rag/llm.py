from __future__ import annotations

import json
from typing import Any

from rag.config import settings


def _generate_anthropic(system_prompt: str, user_message: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def _generate_openai(system_prompt: str, user_message: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content or ""


def generate(system_prompt: str, user_message: str) -> str:
    if settings.llm_provider == "openai":
        return _generate_openai(system_prompt, user_message)
    return _generate_anthropic(system_prompt, user_message)


def generate_json(system_prompt: str, user_message: str) -> dict[str, Any]:
    """Best-effort JSON generation.

    For OpenAI: uses JSON response_format.
    For Anthropic: asks for JSON and attempts to parse.
    """

    if settings.llm_provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model=settings.openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)

    # Anthropic fallback
    text = _generate_anthropic(
        system_prompt + "\n\nReturn ONLY valid JSON.",
        user_message,
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"answer": text, "citations": []}
