from __future__ import annotations
from typing import List, Dict, Any
from cmr_agent.vectordb import ChromaStore

class RetrievalAgent:
    def __init__(self, collection: str = 'nasa_docs'):
        self.store = ChromaStore(collection)

    async def run(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return self.store.similarity_search(query, k=k)
