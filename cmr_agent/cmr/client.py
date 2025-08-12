import httpx
from typing import Any, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from cmr_agent.config import settings
from cmr_agent.cmr.circuit import CircuitBreaker

class AsyncCMRClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or settings.cmr_base_url
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        self.circuit = CircuitBreaker()

    async def close(self):
        await self._client.aclose()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=3))
    async def _safe_get(self, path: str, params: dict):
        if not self.circuit.allow():
            raise RuntimeError('CMR circuit open; temporarily rejecting requests')
        try:
            resp = await self._client.get(path, params=params)
            resp.raise_for_status()
            self.circuit.record_success()
            return resp
        except Exception:
            self.circuit.record_failure()
            raise

    async def search_collections(self, params: Dict[str, Any]) -> Dict[str, Any]:
        resp = await self._safe_get('/search/collections.umm_json', params=params)
        return resp.json()

    async def search_granules(self, params: Dict[str, Any]) -> Dict[str, Any]:
        resp = await self._safe_get('/search/granules.umm_json', params=params)
        return resp.json()

    async def search_variables(self, params: Dict[str, Any]) -> Dict[str, Any]:
        resp = await self._safe_get('/search/variables.umm_json', params=params)
        return resp.json()
