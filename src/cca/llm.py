"""Minimal, provider-agnostic LLM interface: one `generate(system, prompt) -> str` call.

Keeping the surface this small means the agent logic (planning, writing, revising)
is identical across providers and fully testable with a scripted mock.

    CCA_LLM_PROVIDER = gemini | anthropic | offline   (default: auto-detect from API keys)
    GEMINI_API_KEY / GOOGLE_API_KEY, GEMINI_MODEL       (default model below; override freely)
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


class LLM:
    name = "base"

    def generate(self, system: str, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class GeminiLLM(LLM):
    model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    name: str = "gemini"

    def __post_init__(self):
        from google import genai

        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=key)

    def generate(self, system: str, prompt: str) -> str:
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self.model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.0),
        )
        return resp.text or ""


@dataclass
class AnthropicLLM(LLM):
    model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    name: str = "anthropic"

    def __post_init__(self):
        import anthropic

        self._client = anthropic.Anthropic()

    def generate(self, system: str, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self.model, max_tokens=2000, temperature=0.0, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


@dataclass
class ScriptedLLM(LLM):
    """Returns canned responses in order -- used by tests to exercise the agent loop offline."""
    responses: list[str] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)
    name: str = "scripted"

    def generate(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        return self.responses.pop(0) if self.responses else "{}"


def get_llm(provider: str | None = None) -> LLM | None:
    """Return the configured LLM, or None for the deterministic offline mode."""
    provider = (provider or os.getenv("CCA_LLM_PROVIDER") or "").lower()
    if not provider:
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            provider = "offline"
    if provider == "gemini":
        return GeminiLLM()
    if provider == "anthropic":
        return AnthropicLLM()
    return None
