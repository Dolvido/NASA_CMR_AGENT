import asyncio
import pytest
import cmr_agent.graph.pipeline as pipeline
from cmr_agent.graph.pipeline import build_graph

@pytest.mark.asyncio
async def test_graph_runs(monkeypatch):
  class DummyStore:
    def similarity_search(self, query, k=5):
      return []

  class DummyRetrievalAgent:
    def __init__(self, *args, **kwargs):
      self.store = DummyStore()

    async def run(self, query: str, k: int = 5):
      return []

  monkeypatch.setattr(pipeline, 'RetrievalAgent', DummyRetrievalAgent)
  graph = build_graph()
  res = await graph.ainvoke({'user_query': 'social security'})
  assert 'synthesis' in res
  # ensure plan exists in state
  assert 'plan' in res
