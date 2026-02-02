from functools import lru_cache

import anthropic

from rag.config import settings


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def generate(system_prompt: str, user_message: str) -> str:
    client = get_client()
    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text
