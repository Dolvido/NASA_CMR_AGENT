from typing import Literal, TypedDict, Any, Optional

IntentType = Literal["exploratory", "specific", "analytical"]

class QueryState(TypedDict, total=False):
    user_query: str
    intent: IntentType
    subqueries: list[str]
    plan: dict
    validated: bool
    validation_notes: str
    cmr_results: dict
    analysis: dict
    synthesis: str
    context: dict
    temporal: tuple[str, str]
    bbox: tuple[float, float, float, float]
    history: list[str]

class AgentResult(TypedDict, total=False):
    name: str
    data: Any
    error: Optional[str]
