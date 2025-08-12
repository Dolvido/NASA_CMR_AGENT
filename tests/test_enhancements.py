import os
from fastapi.testclient import TestClient
import chromadb
import server.main as m
from cmr_agent.config import Settings
from cmr_agent.vectordb import ChromaStore
import cmr_agent.graph.pipeline as pipeline


def test_default_provider_all(monkeypatch):
    monkeypatch.delenv('CMR_PROVIDER', raising=False)
    s = Settings()
    assert s.cmr_provider == 'ALL'


def test_query_reuses_app_graph(monkeypatch):
    calls = {'count': 0}
    orig_build = m.build_graph

    def counting_build_graph():
        calls['count'] += 1
        return orig_build()

    monkeypatch.setattr(m, 'build_graph', counting_build_graph)

    class DummyStore:
        def similarity_search(self, query, k=5):
            return []

    class DummyRetrievalAgent:
        def __init__(self, *args, **kwargs):
            self.store = DummyStore()

        async def run(self, query: str, k: int = 5):
            return []

    monkeypatch.setattr(pipeline, 'RetrievalAgent', DummyRetrievalAgent)

    with TestClient(m.app) as client:
        assert calls['count'] == 1
        resp = client.get('/query', params={'query': 'social security'})
        assert resp.status_code == 200
        assert calls['count'] == 1


def test_chroma_telemetry_disabled(monkeypatch):
    captured = {}

    class DummyClient:
        def __init__(self, settings):
            captured['settings'] = settings

        def get_or_create_collection(self, name):
            class DummyCollection:
                def add(self, ids, documents, metadatas=None):
                    pass

                def query(self, query_texts, n_results):
                    return {'ids': [[]], 'documents': [[]], 'metadatas': [[]]}

            return DummyCollection()

    monkeypatch.setattr(chromadb, 'Client', DummyClient)
    store = ChromaStore('test')
    assert captured['settings'].anonymized_telemetry is False


def test_posthog_capture_does_not_error(monkeypatch):
    """Ensure PostHog telemetry hooks are harmless no-ops."""

    # Prevent real chroma client from initialising
    class DummyClient:
        def __init__(self, settings):
            pass

        def get_or_create_collection(self, name):
            class DummyCollection:
                def add(self, ids, documents, metadatas=None):
                    pass

                def query(self, query_texts, n_results):
                    return {'ids': [[]], 'documents': [[]], 'metadatas': [[]]}

            return DummyCollection()

    monkeypatch.setattr(chromadb, 'Client', DummyClient)

    # Import posthog after monkeypatching so the module is available
    import posthog

    store = ChromaStore('dummy')

    # The patched capture should accept arbitrary args without raising
    posthog.capture('id', 'event', {'foo': 'bar'})


def test_planning_stage_present_in_graph():
    graph = pipeline.build_graph()
    # compiled graph has agraph attribute; ensure node exists in config string
    text = str(graph)
    assert 'planning_step' in text
