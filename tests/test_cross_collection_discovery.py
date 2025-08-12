import pytest
from cmr_agent.agents.analysis_agent import AnalysisAgent

@pytest.mark.asyncio
async def test_cross_collection_discovery_maps_ids():
    cmr_results = {
        "searches": [
            {
                "query": "q1",
                "collections": {"items": [{"meta": {"concept-id": "C1"}, "umm": {}}]},
                "granules": {"items": []},
                "variables": {"items": []},
            },
            {
                "query": "q2",
                "collections": {"items": [{"meta": {"concept-id": "C1"}, "umm": {}}]},
                "granules": {"items": []},
                "variables": {"items": []},
            },
        ]
    }
    agent = AnalysisAgent()
    summary = await agent.run(cmr_results)
    assert summary["cross_collection_map"] == {"C1": [0, 1]}
