 
import pytest
from fastapi.testclient import TestClient
import cmr_agent.graph.pipeline as pipeline
from cmr_agent.utils import infer_temporal, infer_bbox
import server.main as m


def test_infer_temporal_and_bbox():
    start, end = infer_temporal("rainfall 2010-2012 over sub-saharan africa")
    assert start == "2010-01-01T00:00:00Z"
    assert end == "2012-12-31T23:59:59Z"
    bbox = infer_bbox("Datasets for Sub-Saharan Africa")
    assert bbox == (-20.0, -35.0, 52.0, 20.0)


@pytest.mark.asyncio
async def test_cmr_agent_uses_variables(monkeypatch):
    calls = {"variables": 0}

    class DummyClient:
        async def search_collections(self, params):
            return {"items": []}

        async def search_granules(self, params):
            return {"items": []}

        async def search_variables(self, params):
            calls["variables"] += 1
            return {"items": []}

        async def close(self):
            pass

    agent = pipeline.CMRAgent()
    agent.client = DummyClient()
    await agent.run("rain", [])
    await agent.close()
    assert calls["variables"] == 1


@pytest.mark.asyncio
async def test_analysis_temporal_coverage():
    from cmr_agent.agents.analysis_agent import AnalysisAgent

    cmr_results = {
        "searches": [
            {
                "query": "test",
                "collections": {"items": []},
                "granules": {
                    "items": [
                        {
                            "umm": {
                                "TemporalExtent": {
                                    "RangeDateTime": {
                                        "BeginningDateTime": "2020-01-01T00:00:00Z",
                                        "EndingDateTime": "2020-01-10T00:00:00Z",
                                    }
                                }
                            }
                        },
                        {
                            "umm": {
                                "TemporalExtent": {
                                    "RangeDateTime": {
                                        "BeginningDateTime": "2020-01-05T00:00:00Z",
                                        "EndingDateTime": "2020-01-20T00:00:00Z",
                                    }
                                }
                            }
                        },
                    ]
                },
                "variables": {"items": []},
            }
        ]
    }
    agent = AnalysisAgent()
    summary = await agent.run(cmr_results)
    coverage = summary["queries"][0]["temporal_coverage"]
    assert coverage["start"] == "2020-01-01"
    assert coverage["end"] == "2020-01-20"
    chart = summary["queries"][0]["coverage_chart"]
    assert isinstance(chart, str) and len(chart) > 50


@pytest.mark.asyncio
async def test_analysis_spatial_extent():
    from cmr_agent.agents.analysis_agent import AnalysisAgent

    cmr_results = {
        "searches": [
            {
                "query": "test",
                "collections": {"items": []},
                "granules": {
                    "items": [
                        {
                            "umm": {
                                "SpatialExtent": {
                                    "HorizontalSpatialDomain": {
                                        "Geometry": {
                                            "BoundingBox": {
                                                "WestBoundingCoordinate": -10,
                                                "SouthBoundingCoordinate": -5,
                                                "EastBoundingCoordinate": 5,
                                                "NorthBoundingCoordinate": 10,
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        {
                            "umm": {
                                "SpatialExtent": {
                                    "HorizontalSpatialDomain": {
                                        "Geometry": {
                                            "BoundingBox": {
                                                "WestBoundingCoordinate": -15,
                                                "SouthBoundingCoordinate": 0,
                                                "EastBoundingCoordinate": 10,
                                                "NorthBoundingCoordinate": 12,
                                            }
                                        }
                                    }
                                }
                            }
                        },
                    ]
                },
                "variables": {"items": []},
            }
        ]
    }
    agent = AnalysisAgent()
    summary = await agent.run(cmr_results)
    bbox = summary["queries"][0]["spatial_extent"]["bbox"]
    assert bbox == [-15.0, -5.0, 10.0, 12.0]


@pytest.mark.asyncio
async def test_analysis_handles_empty_results():
    from cmr_agent.agents.analysis_agent import AnalysisAgent

    cmr_results = {
        "searches": [
            {
                "query": "test",
                "collections": {"items": []},
                "granules": {"items": []},
                "variables": {"items": []},
            }
        ]
    }
    agent = AnalysisAgent()
    summary = await agent.run(cmr_results)
    q = summary["queries"][0]["quality"]
    assert q["temporal_res"] is None
    assert q["coverage"]["temporal_pct"] == 0.0
    assert q["completeness_score"] == 0.0
    assert q["tradeoffs"] == []


@pytest.mark.asyncio
async def test_dataset_relationship_recommendations():
    from cmr_agent.agents.analysis_agent import AnalysisAgent

    cmr_results = {
        "searches": [
            {
                "query": "test",
                "collections": {
                    "items": [
                        {"meta": {"concept-id": "C1"}, "umm": {}},
                        {"meta": {"concept-id": "C2"}, "umm": {}},
                    ]
                },
                "granules": {"items": []},
                "variables": {
                    "items": [
                        {
                            "umm": {"Name": "VarA"},
                            "associations": {
                                "collections": [
                                    {"concept_id": "C1"},
                                    {"concept_id": "C2"},
                                ]
                            },
                        }
                    ]
                },
            }
        ]
    }
    agent = AnalysisAgent()
    summary = await agent.run(cmr_results)
    rels = summary.get("dataset_relationships", [])
    assert any(r["variable"] == "VarA" and set(r["collections"]) == {"C1", "C2"} for r in rels)


def test_session_memory_persists(monkeypatch):
    class DummyRetrievalAgent:
        def __init__(self, *args, **kwargs):
            self.store = type("S", (), {"similarity_search": lambda self, q, k=5: []})()

        async def run(self, query: str, k: int = 5):
            return []

    class DummyCMR:
        async def run(self, query: str, subqueries):
            return {"searches": []}

        async def close(self):
            pass

    monkeypatch.setattr(pipeline, "RetrievalAgent", DummyRetrievalAgent)
    monkeypatch.setattr(pipeline, "CMRAgent", DummyCMR)

    with TestClient(m.app) as client:
        r1 = client.get("/query", params={"query": "first", "session_id": "abc"})
        assert r1.json()["history"] == ["first"]
        r2 = client.get("/query", params={"query": "second", "session_id": "abc"})
        assert r2.json()["history"] == ["first", "second"]


@pytest.mark.asyncio
async def test_planning_expands_synonyms():
    from cmr_agent.agents.planning_agent import PlanningAgent

    agent = PlanningAgent()
    plan = await agent.run("rainfall over africa", ["rainfall over africa"])
    terms = plan.get("expanded_terms", [])
    if getattr(agent, "llm", None) is None:
        assert "rainfall" in terms
    else:
        assert any(t in terms for t in ["precipitation", "imerg", "trmm"])
