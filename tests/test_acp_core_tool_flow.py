"""Test ACP app-to-app flow through the core tool-call execution path."""

import asyncio
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acp import ACPAgent
from api import execute_tool_calls
from tools import set_acp_agent


async def test_acp_cross_app_tool_flow() -> None:
    """Core tool calls should discover peers and deliver messages via ACP."""
    print("Test: ACP core tool flow across apps")
    print("-" * 60)

    core_agent = ACPAgent(
        agent_id="core_agent",
        agent_name="Core Agent",
        host="127.0.0.1",
        port=8780,
        protocol="tcp",
    )
    weather_app = ACPAgent(
        agent_id="weather_app",
        agent_name="Weather App",
        host="127.0.0.1",
        port=8781,
        protocol="tcp",
    )

    try:
        await core_agent.initialize()
        await weather_app.initialize()
        await core_agent.register_capabilities(["core", "tool_execution"])
        await weather_app.register_capabilities(["weather_lookup", "forecast"])
        set_acp_agent(core_agent)

        tool_message = {
            "tool_calls": [
                {
                    "id": "call_discover",
                    "type": "function",
                    "function": {
                        "name": "acp_discover_peers",
                        "arguments": json.dumps(
                            {"query_type": "capability", "query_value": "weather_lookup"}
                        ),
                    },
                },
                {
                    "id": "call_send",
                    "type": "function",
                    "function": {
                        "name": "acp_send_message",
                        "arguments": json.dumps(
                            {
                                "recipient_id": "weather_app",
                                "message": "Fetch weather for Boston",
                                "secure": True,
                            }
                        ),
                    },
                },
                {
                    "id": "call_registry",
                    "type": "function",
                    "function": {
                        "name": "acp_get_registry",
                        "arguments": json.dumps({}),
                    },
                },
            ]
        }

        results = await execute_tool_calls(tool_message)
        assert len(results) == 3

        discover_result = json.loads(results[0]["content"])
        assert discover_result["success"] is True
        assert discover_result["peers_found"] >= 1
        discovered_ids = {peer["agent_id"] for peer in discover_result["peers"]}
        assert "weather_app" in discovered_ids

        send_result = json.loads(results[1]["content"])
        assert send_result["success"] is True

        assert weather_app.received_messages, "Weather app did not receive ACP message"
        last_message = weather_app.received_messages[-1]["message"]
        assert last_message["sender_id"] == "core_agent"
        assert last_message["payload"]["text"] == "Fetch weather for Boston"

        registry_result = json.loads(results[2]["content"])
        assert registry_result["success"] is True
        peer_ids = {peer["agent_id"] for peer in registry_result["peers"]}
        assert "weather_app" in peer_ids

        print("✓ Core ACP tool calls discovered and messaged another app")
        print()
    finally:
        set_acp_agent(None)
        await core_agent.shutdown()
        await weather_app.shutdown()


async def run_all_tests() -> int:
    """Run ACP core tool flow tests."""
    try:
        await test_acp_cross_app_tool_flow()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    print("All ACP core tool flow tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_all_tests()))
