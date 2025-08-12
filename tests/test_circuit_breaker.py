import pytest
from tenacity import RetryError
from cmr_agent.cmr.client import AsyncCMRClient

@pytest.mark.asyncio
async def test_cmr_circuit_breaker_trips():
    client = AsyncCMRClient(base_url="http://localhost:9")
    client.circuit.failure_threshold = 1
    with pytest.raises(RetryError):
        await client.search_collections({"keyword": "x"})
    assert not client.circuit.allow()
    with pytest.raises(RetryError):
        await client.search_collections({"keyword": "x"})
    await client.close()
