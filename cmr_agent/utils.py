"""Utility helpers for the CMR agent system."""

from __future__ import annotations

import re
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Callable, TypeVar, Any, Tuple

T = TypeVar('T')


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
def with_retry(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Execute ``fn`` with basic retry semantics."""
    return fn(*args, **kwargs)


# Simple geographic shorthands to bounding boxes (W,S,E,N)
REGION_TO_BBOX = {
    'sub-saharan africa': (-20.0, -35.0, 52.0, 20.0),
    'subsaharan africa': (-20.0, -35.0, 52.0, 20.0),
    'sub saharan africa': (-20.0, -35.0, 52.0, 20.0),
    'ssa': (-20.0, -35.0, 52.0, 20.0),
    'global': (-180.0, -90.0, 180.0, 90.0),
}


def infer_temporal(text: str) -> Tuple[str | None, str | None]:
    """Extract begin/end timestamps from free-form text.

    Returns a tuple of ISO8601 strings or ``(None, None)`` if no year range is
    detected.
    """

    years = [int(y) for y in re.findall(r"((?:19|20)\d{2})", text)]
    if len(years) >= 2:
        years.sort()
        start = datetime(years[0], 1, 1).strftime('%Y-%m-%dT%H:%M:%SZ')
        end = datetime(years[-1], 12, 31, 23, 59, 59).strftime('%Y-%m-%dT%H:%M:%SZ')
        return start, end
    return None, None


def infer_bbox(text: str) -> Tuple[float, float, float, float] | None:
    """Return a bounding box for known regions inside ``text``."""

    lowered = text.lower()
    for key, bbox in REGION_TO_BBOX.items():
        if key in lowered:
            return bbox
    return None

