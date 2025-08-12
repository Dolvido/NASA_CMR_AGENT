import json
import pytest
from server import main

class DummyGraph:
    async def astream(self, state):
        for i in range(2):
            yield {"step": i}

@pytest.mark.asyncio
async def test_stream_emits_sse():
    main.APP_GRAPH = DummyGraph()
    gen = main.run_query_stream("test", None)
    first = await gen.__anext__()
    text = first.decode()
    assert text.startswith("event: update\ndata: ")
    # ensure data is JSON
    payload = text.split("data: ", 1)[1].strip()
    json.loads(payload)
