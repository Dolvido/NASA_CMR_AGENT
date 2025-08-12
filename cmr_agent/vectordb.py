from __future__ import annotations
from typing import Iterable, List, Dict, Any
import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from cmr_agent.config import settings

# Chroma uses the `posthog` package for anonymised telemetry. The execution
# environment ships with a minimal stub whose ``capture`` function accepts a
# single argument, whereas Chroma calls it with three. This results in noisy
# ``TypeError`` messages during client initialisation. To keep output clean and
# avoid network calls, we patch ``posthog.capture`` and ``posthog.identify`` to
# permissive no-ops that swallow all arguments.
try:  # pragma: no cover - best effort; if posthog isn't installed, nothing to do
    import posthog

    posthog.capture = lambda *args, **kwargs: None
    posthog.identify = lambda *args, **kwargs: None
except Exception:  # pragma: no cover - if import fails we simply skip patching
    pass

class ChromaStore:
    def __init__(self, collection_name: str = 'nasa_docs'):
        persist_dir = settings.vector_db_dir
        os.makedirs(persist_dir, exist_ok=True)
        try:
            self.client = chromadb.Client(
                ChromaSettings(persist_directory=persist_dir, anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(collection_name)
        except Exception:
            # Fallback to no-op store when chroma initialisation fails
            self.client = None
            self.collection = None

    def add_texts(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]] | None = None):
        if not self.collection:
            return
        try:
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
        except Exception:
            pass

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.collection:
            return []
        try:
            res = self.collection.query(query_texts=[query], n_results=k)
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        for ids, docs, metas in zip(res.get('ids', [[]])[0], res.get('documents', [[]])[0], res.get('metadatas', [[]])[0]):
            out.append({'id': ids, 'text': docs, 'metadata': metas})
        return out

# Simple ingestion helper

def ingest_docs(docs: Iterable[Dict[str, Any]], id_key: str = 'id', text_key: str = 'text', meta_keys: List[str] | None = None, collection: str = 'nasa_docs'):
    store = ChromaStore(collection)
    ids, texts, metas = [], [], []
    for d in docs:
        ids.append(str(d[id_key]))
        texts.append(str(d[text_key]))
        if meta_keys:
            metas.append({k: d.get(k) for k in meta_keys})
        else:
            metas.append({})
    if ids:
        store.add_texts(ids, texts, metas)
    return len(ids)

