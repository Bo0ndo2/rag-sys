"""Gemini provider — registered as 'gemini' in LLMFactory."""

from __future__ import annotations

from typing import List

import httpx

from src.helpers.config import get_settings
from src.stores.llm import BaseLLMClient, LLMFactory

settings = get_settings()

GEMINI_GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
)

# Gemini embedding model is always text-embedding-004 — never use OpenAI model names here
GEMINI_EMBED_MODEL = "gemini-embedding-001"
GEMINI_EMBED_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_EMBED_MODEL}:embedContent?key={{key}}"
)


@LLMFactory.register("gemini")
class GeminiClient(BaseLLMClient):
    def provider_name(self) -> str:
        return "gemini"

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        url = GEMINI_GENERATE_URL.format(
            model=settings.GEMINI_MODEL,
            key=settings.GEMINI_API_KEY,
        )
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_message}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return data["candidates"][0]["content"]["parts"][0]["text"]

    async def embed(self, text: str) -> List[float]:
        url = GEMINI_EMBED_URL.format(key=settings.GEMINI_API_KEY)
        payload = {
            "content": {
                "parts": [{"text": text}]
            }
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise ValueError(
                    f"Gemini embedding failed {resp.status_code}: {resp.text}"
                )
            data = resp.json()

        embedding = data.get("embedding", {})
        if isinstance(embedding, dict) and "values" in embedding:
            return embedding["values"]

        raise ValueError(f"Unexpected Gemini embedding response: {data}")