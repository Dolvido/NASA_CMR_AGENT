from __future__ import annotations
from typing import Any, List
import logging


logger = logging.getLogger(__name__)


class SynthesisAgent:
    def __init__(self):
        try:
            from cmr_agent.llm.router import LLMRouter
            self.router = LLMRouter()
            try:
                self.llm = self.router.get()
            except Exception:
                self.llm = None
        except Exception:
            self.router = None
            self.llm = None

    async def _fallback(self, query: str, analysis: dict, history: List[str]) -> str:
        parts = [
            f"Query: {query}",
            f"Total collections: {analysis.get('total_collections', 0)}",
            f"Total granules: {analysis.get('total_granules', 0)}",
            f"Total variables: {analysis.get('total_variables', 0)}",
        ]
        for q in analysis.get('queries', [])[:3]:
            line = (
                f"- '{q.get('query')}' -> collections={q.get('collections_found')}, "
                f"granules={q.get('granules_found')}, providers={','.join(q.get('providers', []))}"
            )
            tc = q.get('temporal_coverage')
            if tc:
                line += f", coverage={tc.get('start')} to {tc.get('end')}"
            parts.append(line)
        parts.append(f"Session memory: {len(history)-1} previous queries")
        parts.append(
            "Recommendations: refine temporal/spatial filters and select collections with consistent coverage."
        )
        return "\n".join(parts)

    async def run(self, query: str, analysis: dict, history: List[str]) -> str:
        if self.llm is None:
            return await self._fallback(query, analysis, history)
        prompt = (
            "You are an Earth science data expert. Given a user's query and analysis metadata (counts, examples), "
            "write a concise, structured recommendation: 1) Summary 2) Datasets to consider 3) Gaps & trade-offs 4) Next steps."
        )
        text = f"{prompt}\nUser query: {query}\nAnalysis JSON: {analysis}"
        try:
            msg = await self.llm.ainvoke(text)
        except Exception as e:
            logger.warning("Synthesis LLM failed: %s", e)
            if self.router is not None:
                try:
                    self.router.record_failure(self.llm)
                    self.llm = self.router.get()
                    msg = await self.llm.ainvoke(text)
                except Exception as e2:
                    logger.error("Synthesis fallback LLM failed: %s", e2)
                    self.llm = None
                    return await self._fallback(query, analysis, history)
            else:
                self.llm = None
                return await self._fallback(query, analysis, history)
        return getattr(msg, "content", str(msg))
