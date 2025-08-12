import pytest

from cmr_agent.agents.intent_agent import IntentAgent
from cmr_agent.agents.synthesis_agent import SynthesisAgent


class FailingLLM:
    async def ainvoke(self, _):
        raise RuntimeError("boom")


class DummyRouter:
    def record_failure(self, provider):
        pass

    def get(self):
        raise RuntimeError("no provider")


@pytest.mark.asyncio
async def test_intent_agent_falls_back(monkeypatch):
    agent = IntentAgent()
    agent.llm = FailingLLM()
    agent.router = DummyRouter()
    intent, subqueries = await agent.run("find rainfall data")
    assert intent == "specific"
    assert subqueries == ["find rainfall data"]


@pytest.mark.asyncio
async def test_synthesis_agent_falls_back(monkeypatch):
    agent = SynthesisAgent()
    agent.llm = FailingLLM()
    agent.router = DummyRouter()
    text = await agent.run("query", {"total_collections": 1, "total_granules": 2, "total_variables": 3, "queries": []}, ["q1"])
    assert "Total collections" in text
