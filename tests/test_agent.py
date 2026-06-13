"""Basic tests for the agentic core."""
import pytest
import asyncio
from ai_core.agents.field_agent import run_agent_turn

@pytest.mark.asyncio
async def test_agent_runs_in_mock_mode():
    """Test that the agent can execute a full turn (mocked)."""
    result = await run_agent_turn(
        session_id="test-001",
        user_input="The pump is leaking from the seal and vibrating.",
        image_b64=None,  # no image
        technician_id="test-tech"
    )
    
    assert result["success"] is True
    assert "response" in result
    assert "immediate" in result["response"]
    assert result["response"]["confidence"] >= 0.0

def test_schemas_import():
    from ai_core.models.schemas import AgentState, VisionResult, GuidancePlan
    assert AgentState is not None
    assert VisionResult is not None
