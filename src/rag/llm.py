import os
from typing import Iterator

import ollama


class OllamaLLM:
    """
    Thin wrapper around a locally-running Ollama model.

    The Ollama host defaults to localhost:11434 but can be overridden via
    the OLLAMA_HOST env var (useful when running inside Docker).
    """

    def __init__(
        self,
        model: str = "mistral",
        host: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client = ollama.Client(host=host)

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt and return the model's text response."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature},
        )
        return response["message"]["content"].strip()

    def generate_stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """Yield response tokens one at a time as the model generates them."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for chunk in self._client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature},
            stream=True,
        ):
            yield chunk["message"]["content"]
