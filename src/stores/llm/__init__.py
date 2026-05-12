"""
LLM Factory — Bonus Phase (+5%)
=================================
Implements the Factory Design Pattern to allow the system to switch between
different LLM providers (OpenAI, Gemini, Ollama) through a single config change.

Pattern:
  LLMFactory.get_client("openai")  → OpenAIClient
  LLMFactory.get_client("gemini")  → GeminiClient
  LLMFactory.get_client("ollama")  → OllamaClient

All clients implement BaseLLMClient so the RAG controller is provider-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Literal


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """Common interface that every LLM provider must implement."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """Return the model's reply as a plain string."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for the given text."""
        ...

    @abstractmethod
    def provider_name(self) -> str: ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

LLMProvider = Literal["openai", "gemini", "ollama"]


class LLMFactory:
    _registry: dict[str, type[BaseLLMClient]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a provider class."""
        def decorator(klass: type[BaseLLMClient]):
            cls._registry[name] = klass
            return klass
        return decorator

    @classmethod
    def get_client(cls, provider: str) -> BaseLLMClient:
        if provider not in cls._registry:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[provider]()


# ---------------------------------------------------------------------------
# Import providers so they self-register via @LLMFactory.register(...)
# ---------------------------------------------------------------------------
from src.stores.llm.provider import openai_client, gemini_client, ollama_client  # noqa: E402, F401
