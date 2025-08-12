from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

from cmr_agent.cmr.client import AsyncCMRClient
from cmr_agent.config import settings
from cmr_agent.utils import infer_temporal, infer_bbox


class CMRAgent:
    def __init__(self):
        self.client = AsyncCMRClient(settings.cmr_base_url)
        self.query_log: List[Dict[str, Any]] = []
        # cache collection records keyed by concept-id to avoid repeat lookups
        self.collection_cache: Dict[str, Dict[str, Any]] = {}
        self.circuit_tripped = False

    def _log(self, endpoint: str, params: Dict[str, Any], result: Dict[str, Any]):
        try:
            items = result.get("items") or []
            self.query_log.append(
                {
                    "endpoint": endpoint,
                    "params": {k: v for k, v in params.items() if k != 'password'},
                    "page_size": params.get("page_size"),
                    "total_hits": len(items),
                }
            )
        except Exception:
            pass

    async def run(self, query: str, plan_or_subqueries: Any) -> dict:
        async def search_for(q: str) -> dict:
            temporal = infer_temporal(q)
            bbox = infer_bbox(q)
            params: Dict[str, Any] = {
                "page_size": 25,
                "keyword": q,
            }
            provider = getattr(settings, "cmr_provider", None)
            if provider and provider not in ("", "ALL", "CMR_ALL"):
                params["provider"] = provider
            if temporal[0] and temporal[1]:
                params["temporal"] = f"{temporal[0]},{temporal[1]}"
            if bbox:
                w, s, e, n = bbox
                params["bounding_box"] = f"{w},{s},{e},{n}"

            collections_task = self.client.search_collections(params)

            async def granules(params: Dict[str, Any]) -> dict:
                try:
                    cols = await collections_task
                    items = (cols or {}).get("items", [])
                    if items:
                        concept_ids = [i.get("meta", {}).get("concept-id") for i in items if i.get("meta")]
                        gid = concept_ids[0]
                        gparams = {k: v for k, v in params.items() if k != "page_size"}
                        if gid:
                            gparams["collection_concept_id"] = gid
                        gparams["page_size"] = 50
                        return await self.client.search_granules(gparams)
                except Exception:
                    pass
                return {"items": []}

            variables_task = self.client.search_variables({"keyword": q, "page_size": 25})

            results = await asyncio.gather(
                collections_task, granules(params), variables_task, return_exceptions=True
            )
            collections, granules_res, variables_res = results

            def handle(res, log_params, endpoint: str):
                if isinstance(res, dict):
                    self._log(endpoint, log_params, res)
                    return res
                if isinstance(res, RuntimeError) and 'circuit' in str(res).lower():
                    self.circuit_tripped = True
                return {"items": [], "error": str(res)}

            collections = handle(collections, params, 'collections')
            gparams = {k: v for k, v in params.items() if k != "page_size"}
            gparams["page_size"] = 50
            granules_res = handle(granules_res, gparams, 'granules')
            variables_res = handle(variables_res, {"keyword": q, "page_size": 25}, 'variables')

            return {
                "query": q,
                "collections": collections,
                "granules": granules_res,
                "variables": variables_res,
            }

        # If a planner provided stages, follow variable->collections->granules per stage
        if isinstance(plan_or_subqueries, dict) and plan_or_subqueries.get("stages"):
            stages = plan_or_subqueries["stages"]

            async def run_stage(stage: Dict[str, Any]) -> Dict[str, Any]:
                q = stage.get("query", query)
                temporal = infer_temporal(q)
                bbox = infer_bbox(q)
                base_params: Dict[str, Any] = {"page_size": 25, "keyword": q}
                provider = getattr(settings, "cmr_provider", None)
                if provider and provider not in ("", "ALL", "CMR_ALL"):
                    base_params["provider"] = provider
                if temporal[0] and temporal[1]:
                    base_params["temporal"] = f"{temporal[0]},{temporal[1]}"
                if bbox:
                    w, s, e, n = bbox
                    base_params["bounding_box"] = f"{w},{s},{e},{n}"

                variable_terms: List[str] = stage.get("variable_terms", []) or [q]
                # Query variables concurrently for each term
                var_params = [{"keyword": term, "page_size": 25} for term in variable_terms]
                var_tasks = [self.client.search_variables(p) for p in var_params]
                raw_var_results = await asyncio.gather(*var_tasks, return_exceptions=True)
                var_results: List[Dict[str, Any]] = []
                for res, term, params_var in zip(raw_var_results, variable_terms, var_params):
                    if isinstance(res, Exception):
                        if isinstance(res, RuntimeError) and 'circuit' in str(res).lower():
                            self.circuit_tripped = True
                        res = {"error": str(res), "items": []}
                    res["_term"] = term
                    var_results.append(res)
                    if isinstance(res, dict):
                        self._log('variables', params_var, res)

                # Collect related collection concept ids from variable associations
                related_collection_ids: List[str] = []
                for vres in var_results:
                    for v in (vres.get("items") or []):
                        assocs = (v.get("associations") or {}).get("collections", [])
                        for a in assocs:
                            cid = a.get("concept_id") or a.get("concept-id")
                            if cid:
                                related_collection_ids.append(cid)
                # de-duplicate
                seen = set()
                related_collection_ids_unique = []
                for cid in related_collection_ids:
                    if cid not in seen:
                        related_collection_ids_unique.append(cid)
                        seen.add(cid)

                # Prepare collection searches: by keyword plus by short_name, science_keywords_h, and concept_id batches if available
                collections_keyword_task = self.client.search_collections(base_params)
                short_name_tasks = []
                science_kw_tasks = []
                for term in (variable_terms or [])[:3]:
                    p_short = {**base_params}
                    p_short.pop("keyword", None)
                    p_short["short_name"] = term
                    short_name_tasks.append(self.client.search_collections(p_short))

                    p_science = {**base_params}
                    p_science.pop("keyword", None)
                    p_science["science_keywords_h"] = term
                    science_kw_tasks.append(self.client.search_collections(p_science))
                # fetch collections for ids not already cached
                collections_by_id: Dict[str, Any] = {"items": []}
                cached_items: List[Dict[str, Any]] = []
                missing_ids: List[str] = []
                for cid in related_collection_ids_unique:
                    item = self.collection_cache.get(cid)
                    if item:
                        cached_items.append(item)
                    else:
                        missing_ids.append(cid)
                if missing_ids:
                    try:
                        id_params = {**base_params}
                        id_params.pop("keyword", None)
                        for cid in missing_ids[:50]:
                            id_params.setdefault("concept_id", []).append(cid)
                        collections_by_id = await self.client.search_collections(id_params)
                        for c in (collections_by_id or {}).get("items", []):
                            cid = (c.get("meta") or {}).get("concept-id")
                            if cid:
                                self.collection_cache[cid] = c
                    except Exception as e:
                        if isinstance(e, RuntimeError) and 'circuit' in str(e).lower():
                            self.circuit_tripped = True
                        collections_by_id = {"items": []}

                # Merge collection results (unique by concept-id)
                def extract_items(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
                    return (obj or {}).get("items", [])

                merged_items: List[Dict[str, Any]] = []
                merge_seen = set()
                results_list: List[Dict[str, Any]] = []
                kw_cols = await collections_keyword_task
                if isinstance(kw_cols, Exception):
                    if isinstance(kw_cols, RuntimeError) and 'circuit' in str(kw_cols).lower():
                        self.circuit_tripped = True
                    kw_cols = {"items": [], "error": str(kw_cols)}
                else:
                    self._log('collections', base_params, kw_cols)
                results_list.append(kw_cols)

                extra_results = await asyncio.gather(*short_name_tasks, *science_kw_tasks, return_exceptions=True)
                for er in extra_results:
                    if isinstance(er, Exception):
                        if isinstance(er, RuntimeError) and 'circuit' in str(er).lower():
                            self.circuit_tripped = True
                        continue
                    results_list.append(er)
                    self._log('collections', base_params, er)
                results_list.append(collections_by_id)
                if isinstance(collections_by_id, dict) and missing_ids:
                    id_params = {k: v for k, v in base_params.items() if k != "keyword"}
                    for cid in missing_ids[:50]:
                        id_params.setdefault("concept_id", []).append(cid)
                    self._log('collections', id_params, collections_by_id)
                if cached_items:
                    results_list.append({"items": cached_items})

                for c in sum((extract_items(r) for r in results_list), []):
                    cid = (c.get("meta") or {}).get("concept-id")
                    if cid and cid not in merge_seen:
                        merged_items.append(c)
                        merge_seen.add(cid)

                # Choose a few collections to fetch granules for
                async def fetch_granules_for_collection(collection: Dict[str, Any]) -> Dict[str, Any]:
                    gid = (collection.get("meta") or {}).get("concept-id")
                    gparams = {k: v for k, v in base_params.items() if k != "page_size"}
                    if gid:
                        gparams["collection_concept_id"] = gid
                    gparams["page_size"] = 50
                    res = await self.client.search_granules(gparams)
                    if isinstance(res, dict):
                        self._log('granules', gparams, res)
                    return res

                granules_results: List[Dict[str, Any]] = []
                # Limit to first 3 collections to avoid heavy load
                granules_results = await asyncio.gather(
                    *(fetch_granules_for_collection(c) for c in merged_items[:3]), return_exceptions=True
                )

                # Aggregate granules into a single view for the stage
                combined_granules_items: List[Dict[str, Any]] = []
                for gr in granules_results:
                    if isinstance(gr, Exception):
                        if isinstance(gr, RuntimeError) and 'circuit' in str(gr).lower():
                            self.circuit_tripped = True
                        continue
                    combined_granules_items.extend((gr or {}).get("items", []))

                return {
                    "query": q,
                    "variables": {"items": sum((vres.get("items", []) for vres in var_results), [])},
                    "collections": {"items": merged_items},
                    "granules": {"items": combined_granules_items},
                    "related_collection_ids": related_collection_ids_unique,
                }

            searches = await asyncio.gather(*(run_stage(st) for st in stages))
            return {"searches": searches}

        subqueries = plan_or_subqueries or [query]
        searches = await asyncio.gather(*(search_for(q) for q in (subqueries or [query])))
        cb_open = False
        circuit = getattr(self.client, 'circuit', None)
        if circuit is not None:
            try:
                cb_open = not circuit.allow()
            except Exception:
                cb_open = False
        return {
            "searches": searches,
            "query_log": self.query_log,
            "circuit_breaker_tripped": self.circuit_tripped or cb_open,
        }

    async def close(self):
        await self.client.close()
