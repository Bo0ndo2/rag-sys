"""OpenAI provider — registered as 'openai' in LLMFactory."""

from openai import AsyncOpenAI
from src.stores.llm import LLMFactory, BaseLLMClient
from src.helpers.config import get_settings

settings = get_settings()


@LLMFactory.register("openai")
class OpenAIClient(BaseLLMClient):

    def __init__(self):
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    def provider_name(self) -> str:
        return "openai"

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        response = await self._client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding
