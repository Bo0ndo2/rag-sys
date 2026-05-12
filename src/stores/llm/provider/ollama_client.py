"""Ollama provider (local models like Mistral) — registered as 'ollama'."""

import httpx
from src.stores.llm import LLMFactory, BaseLLMClient
from src.helpers.config import get_settings

settings = get_settings()


@LLMFactory.register("ollama")
class OllamaClient(BaseLLMClient):

    def provider_name(self) -> str:
        return "ollama"

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        url = f"{settings.OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": settings.OLLAMA_MODEL,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "messages": [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_message},
            ],
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["message"]["content"]

    async def embed(self, text: str) -> list[float]:
        """Ollama embed endpoint (available since Ollama 0.1.26)."""
        url = f"{settings.OLLAMA_BASE_URL}/api/embed"
        payload = {"model": settings.OLLAMA_MODEL, "input": text}
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        # Ollama returns {"embeddings": [[...]]}
        return data["embeddings"][0]
