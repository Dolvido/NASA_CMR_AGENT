from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import asyncio
from cmr_agent.graph.pipeline import build_graph
from cmr_agent.agents.intent_agent import IntentAgent
from cmr_agent.agents.analysis_agent import AnalysisAgent
from cmr_agent.agents.validation_agent import ValidationAgent
from cmr_agent.agents.synthesis_agent import SynthesisAgent
from cmr_agent.cmr.client import AsyncCMRClient
from cmr_agent.config import settings
from cmr_agent.utils import infer_temporal, infer_bbox


SESSIONS: dict[str, list[str]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global APP_GRAPH
    APP_GRAPH = build_graph()
    yield


app = FastAPI(title='NASA CMR AI Agent', lifespan=lifespan)

# Serve the simple UI
_ROOT = Path(__file__).resolve().parents[1]
_UI_DIR = _ROOT / 'ui'
if _UI_DIR.exists():
    app.mount('/ui', StaticFiles(directory=str(_UI_DIR), html=True), name='ui')

@app.get('/')
async def root():
    # Redirect to the UI if available, otherwise show a helpful message
    if _UI_DIR.exists():
        return RedirectResponse(url='/ui/')
    return {"message": "UI not found. Ensure the 'ui/' directory exists.", "endpoints": ["/query", "/stream"]}

async def run_query_stream(user_query: str, session_id: str | None):
    """Stream incremental updates to reduce perceived latency during CMR calls.

    This orchestrates key steps inline and emits SSE 'update' events as soon as
    partial results are available (intent → per-subquery collections/variables →
    analysis → synthesis). It avoids waiting for all CMR requests to complete
    before sending the first bytes to the client.
    """
    history = SESSIONS.get(session_id, []) if session_id else []
    history = list(history)
    history.append(user_query)

    async def emit(obj: dict):
        try:
            data = json.dumps(obj)
        except Exception:
            data = str(obj)
        yield_bytes = f"event: update\ndata: {data}\n\n".encode('utf-8')
        return yield_bytes

    # 1) Intent first (fast), then begin CMR in parallel and stream as each completes
    try:
        # Intent classification
        intent_agent = IntentAgent()
        intent, subqueries = await intent_agent.run(user_query)
        first = {
            'phase': 'intent',
            'user_query': user_query,
            'intent': intent,
            'subqueries': subqueries,
            'history': history,
        }
        yield await emit(first)

        # Lightweight validation so the UI can tailor messaging during stream
        try:
            validation_agent = ValidationAgent()
            validation = await validation_agent.run(user_query, subqueries or [])
            yield await emit({'phase': 'validation', 'validation': validation, 'validated': bool(validation.get('feasible', False))})
        except Exception as e:
            yield await emit({'phase': 'validation', 'error': str(e), 'validated': False})

        # 2) CMR streaming per subquery
        cmr_client = AsyncCMRClient(settings.cmr_base_url)
        cmr_searches: list[dict] = []
        try:
            for q in (subqueries or [user_query]):
                temporal = infer_temporal(q)
                bbox = infer_bbox(q)
                params: dict = {'page_size': 25, 'keyword': q}
                provider = getattr(settings, 'cmr_provider', None)
                if provider and provider not in ('', 'ALL', 'CMR_ALL'):
                    params['provider'] = provider
                if temporal[0] and temporal[1]:
                    params['temporal'] = f"{temporal[0]},{temporal[1]}"
                if bbox:
                    w, s, e, n = bbox
                    params['bounding_box'] = f"{w},{s},{e},{n}"

                # Fire collections and variables concurrently; stream as each completes
                collections_task = asyncio.create_task(cmr_client.search_collections(params))
                variables_task = asyncio.create_task(cmr_client.search_variables({'keyword': q, 'page_size': 25}))

                partial = {'query': q, 'collections': {'items': []}, 'granules': {'items': []}, 'variables': {'items': []}}

                for task in asyncio.as_completed([collections_task, variables_task]):
                    try:
                        res = await task
                        if task is collections_task:
                            partial['collections'] = res if isinstance(res, dict) else {'items': []}
                            yield await emit({'phase': 'cmr', 'type': 'collections', 'query': q, 'data': partial['collections']})
                        else:
                            partial['variables'] = res if isinstance(res, dict) else {'items': []}
                            yield await emit({'phase': 'cmr', 'type': 'variables', 'query': q, 'data': partial['variables']})
                    except Exception as e:
                        kind = 'collections' if task is collections_task else 'variables'
                        yield await emit({'phase': 'cmr', 'type': kind, 'query': q, 'error': str(e)})

                cmr_searches.append(partial)

            cmr_results = {'searches': cmr_searches}
            yield await emit({'phase': 'cmr', 'type': 'done', 'results': cmr_results})
        finally:
            await cmr_client.close()

        # 3) Analysis on the aggregated (possibly partial) CMR results
        analysis_agent = AnalysisAgent()
        analysis = await analysis_agent.run(cmr_results)
        yield await emit({'phase': 'analysis', 'analysis': analysis})

        # 4) Synthesis (fast fallback if LLM not configured)
        synthesis_agent = SynthesisAgent()
        text = await synthesis_agent.run(user_query, analysis, history)
        final = {'phase': 'synthesis', 'synthesis': text}
        yield await emit(final)

        # 5) Recommendations summary compatible with UI expectations during streaming
        try:
            def append_unique(target: list[dict], value: str, why: str):
                normalized_existing = {str(t.get('collection', '')).strip().lower() for t in target}
                normalized_value = str(value or '').strip().lower()
                if not normalized_value or normalized_value in normalized_existing:
                    return
                target.append({'collection': value, 'rank': len(target) + 1, 'why': why})

            recs: list[dict] = []
            for q in analysis.get('queries', [])[:5]:
                for name in (q.get('example_collections') or [])[:3]:
                    append_unique(recs, str(name), 'coverage + relevance (example)')
                if len(recs) >= 5:
                    break
            if len(recs) < 5:
                for rc in analysis.get('related_collections', [])[:10]:
                    cid = (rc or {}).get('concept_id') or ''
                    if cid:
                        append_unique(recs, cid, 'related collection (variables)')
                    if len(recs) >= 5:
                        break
            if len(recs) < 5:
                for q in analysis.get('queries', [])[:5]:
                    for cid in (q.get('related_collections') or [])[:5]:
                        append_unique(recs, str(cid), 'related collection (per-query)')
                        if len(recs) >= 5:
                            break
                    if len(recs) >= 5:
                        break
            if len(recs) < 3:
                uq = (user_query or '').lower()
                suggestions: list[str] = []
                if ('precipitation' in uq) or ('rain' in uq) or ('rainfall' in uq):
                    suggestions.extend(['GPM IMERG', 'TRMM 3B42', 'CHIRPS', 'GPCC', 'ERA5'])
                if 'aerosol' in uq:
                    suggestions.extend(['MODIS Aerosol', 'MERRA-2 Aerosol'])
                seen: set[str] = set()
                for s in suggestions:
                    if s not in seen:
                        append_unique(recs, s, 'keyword-based suggestion')
                        seen.add(s)
                    if len(recs) >= 5:
                        break

            comparison = {'criteria': ['resolution', 'latency', 'record_length', 'validation_status'], 'ranked_recommendations': recs[:5]}
            yield await emit({'phase': 'comparison', 'comparison': comparison})
        except Exception as e:
            yield await emit({'phase': 'comparison', 'error': str(e)})
        # Emit a terminal event so the client can distinguish normal completion
        yield b"event: end\ndata: {}\n\n"

    except Exception as e:
        err = json.dumps({'error': str(e)})
        yield f"event: error\ndata: {err}\n\n".encode('utf-8')
        # Also emit an end event so the client can clean up gracefully
        yield b"event: end\ndata: {}\n\n"
    finally:
        if session_id is not None:
            SESSIONS[session_id] = history

@app.get('/stream')
async def stream(query: str, session_id: str | None = None):
    headers = {
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        # Helpful when running behind proxies like Nginx
        'X-Accel-Buffering': 'no',
    }
    return StreamingResponse(
        run_query_stream(query, session_id),
        media_type='text/event-stream',
        headers=headers,
    )

@app.get('/query')
async def query(query: str, session_id: str | None = None):
    history = SESSIONS.get(session_id, []) if session_id else []
    result = await APP_GRAPH.ainvoke({'user_query': query, 'history': history})
    if session_id is not None:
        SESSIONS[session_id] = result.get('history', history)
    return result
