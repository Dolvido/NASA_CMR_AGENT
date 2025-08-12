import pytest
from cmr_agent.agents.planning_agent import PlanningAgent

@pytest.mark.asyncio
async def test_planner_produces_variable_terms_without_stopwords():
    agent = PlanningAgent()
    query = "rainfall 2010-2012 over sub-saharan africa"
    plan = await agent.run(query, [query])
    vars_terms = plan.get("variable_terms", [])
    # ensure synonyms included
    assert "rainfall" in vars_terms
    assert "precipitation" in vars_terms
    # ensure stop word removed
    assert "over" not in vars_terms
    # numeric ranges should be filtered
    assert all(not t.replace('-', '').isdigit() for t in vars_terms)
