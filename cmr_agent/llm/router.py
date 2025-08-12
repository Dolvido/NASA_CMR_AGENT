from __future__ import annotations
from typing import Optional
import logging
from cmr_agent.config import settings

# Lazy-optional imports to avoid hard dependency issues
try:
    from langchain_openai import ChatOpenAI  # type: ignore
except Exception:
    ChatOpenAI = None  # type: ignore

try:
    from langchain_anthropic import ChatAnthropic  # type: ignore
except Exception:
    ChatAnthropic = None  # type: ignore

logger = logging.getLogger(__name__)


class LLMRouter:
    def __init__(self):
        self.primary: Optional[object] = None
        self.secondary: Optional[object] = None
        self.primary_failed = False
        self.secondary_failed = False
        if settings.openai_api_key and ChatOpenAI is not None:
            self.primary = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=settings.openai_api_key)
        if settings.anthropic_api_key and ChatAnthropic is not None:
            anthropic = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.2, api_key=settings.anthropic_api_key)
            if self.primary is None:
                self.primary = anthropic
            else:
                self.secondary = anthropic

    def get(self) -> object:
        """Return an available LLM client, preferring the primary."""
        if not self.primary_failed and self.primary is not None:
            return self.primary
        if not self.secondary_failed and self.secondary is not None:
            return self.secondary
        raise RuntimeError("No LLM providers configured")

    def fallback(self) -> Optional[object]:
        """Return the secondary provider if it has not failed."""
        if self.secondary_failed:
            return None
        return self.secondary

    def record_failure(self, provider: object) -> None:
        """Mark a provider as failed so subsequent calls skip it."""
        if provider is self.primary:
            logger.warning("Primary LLM provider failed; will use fallback if available")
            self.primary_failed = True
        elif provider is self.secondary:
            logger.warning("Secondary LLM provider failed")
            self.secondary_failed = True
