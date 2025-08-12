import pytest
from cmr_agent.agents.cmr_agent import CMRAgent

class DummyClient:
    def __init__(self):
        self.id_calls = 0
    async def search_variables(self, params):
        return {
            "items": [
                {
                    "meta": {"concept-id": "V1"},
                    "associations": {"collections": [{"concept_id": "C1"}]},
                }
            ]
        }
    async def search_collections(self, params):
        if "concept_id" in params:
            self.id_calls += 1
            ids = params.get("concept_id", [])
            items = [{"meta": {"concept-id": cid}} for cid in ids]
        else:
            items = []
        return {"items": items}
    async def search_granules(self, params):
        return {"items": []}
    async def close(self):
        return

@pytest.mark.asyncio
async def test_collection_id_cache(monkeypatch):
    agent = CMRAgent()
    agent.client = DummyClient()
    plan = {
        "stages": [
            {
                "name": "stage",
                "type": "search",
                "query": "rain",
                "variable_terms": ["rain_var"],
            }
        ]
    }
    await agent.run("rain", plan)
    await agent.run("rain", plan)
    assert agent.client.id_calls == 1
