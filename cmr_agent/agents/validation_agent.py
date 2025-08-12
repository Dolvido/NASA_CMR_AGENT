from __future__ import annotations
from typing import Dict, List
import re
from cmr_agent.utils import infer_bbox


REGIONS = {
    "sub-saharan africa": [-20.0, -35.0, 52.0, 20.0],
    "subsaharan africa": [-20.0, -35.0, 52.0, 20.0],
    "sub saharan africa": [-20.0, -35.0, 52.0, 20.0],
    "ssa": [-20.0, -35.0, 52.0, 20.0],
    "global": [-180, -90, 180, 90],
}


class ValidationAgent:
    async def run(self, query: str, subqueries: list[str]) -> Dict:
        """Validate the user query and return structured feedback."""
        reasons: List[str] = []
        feasible = True

        if not query or not query.strip():
            feasible = False
            reasons.append("Empty query")

        if len(query) < 8:
            reasons.append("Very short query; may be ambiguous")

        lowered = query.lower()

        # simple out-of-scope detection
        banned = ["medical records", "social security", "bank account"]
        if any(b in lowered for b in banned):
            feasible = False
            reasons.append("Out-of-scope content detected")

        # Temporal bounds check (capture full 4-digit years)
        years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", query)]
        if not years:
            reasons.append("No temporal bounds detected")
        elif len(years) >= 2 and years[0] > years[-1]:
            feasible = False
            reasons.append("Start year after end year")

        # Region check (tolerant to common spellings or if bbox can be inferred)
        has_region = any(r in lowered for r in REGIONS.keys()) or infer_bbox(query) is not None
        if not has_region:
            feasible = False
            reasons.append("Region not recognized")

        if len(subqueries) > 5:
            reasons.append("High complexity; will decompose into steps")

        alternatives: List[str] = []
        if "Region not recognized" in reasons:
            alternatives.extend(list(REGIONS.keys()))

        return {
            "feasible": feasible,
            "reasons": reasons,
            "suggested_alternatives": alternatives,
        }
