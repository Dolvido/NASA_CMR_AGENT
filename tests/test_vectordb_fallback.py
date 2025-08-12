import chromadb
from cmr_agent.vectordb import ChromaStore


def test_similarity_search_handles_failure(monkeypatch):
    class DummyCollection:
        def add(self, ids, documents, metadatas=None):
            pass

        def query(self, query_texts, n_results):
            raise RuntimeError("boom")

    class DummyClient:
        def __init__(self, settings):
            pass

        def get_or_create_collection(self, name):
            return DummyCollection()

    monkeypatch.setattr(chromadb, "Client", DummyClient)

    store = ChromaStore("test")
    assert store.similarity_search("foo") == []
