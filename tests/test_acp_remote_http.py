"""Test ACP remote HTTP peer discovery and messaging."""

import asyncio
import json
import sys
import os

from aiohttp import web

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acp import ACPAgent


async def test_remote_http_peer_flow() -> None:
    """ACP should discover and message a remote HTTP ACP peer."""
    print("Test: ACP remote HTTP peer flow")
    print("-" * 60)

    received_run_requests = []

    async def agents_handler(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "agents": [
                    {
                        "name": "remote_agent",
                        "description": "Mock ACP peer",
                        "metadata": {
                            "tags": ["coding", "headless"],
                        },
                        "input_content_types": ["*/*"],
                        "output_content_types": ["text/plain"],
                    }
                ]
            }
        )

    async def runs_handler(request: web.Request) -> web.Response:
        body = await request.json()
        received_run_requests.append(body)
        return web.json_response(
            {
                "run_id": "mock-run-id",
                "agent_name": body.get("agent_name", ""),
                "session_id": "mock-session-id",
                "status": "completed",
                "output": [
                    {
                        "role": "agent/remote_agent",
                        "parts": [
                            {
                                "content_type": "text/plain",
                                "content": "pong",
                                "content_encoding": "plain",
                            }
                        ],
                    }
                ],
            }
        )

    app = web.Application()
    app.router.add_get("/agents", agents_handler)
    app.router.add_post("/runs", runs_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    sockets = site._server.sockets  # type: ignore[attr-defined]
    assert sockets, "Expected listening socket for mock ACP peer"
    port = sockets[0].getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"

    agent = ACPAgent(
        agent_id="core_agent",
        agent_name="Core Agent",
        host="127.0.0.1",
        port=8788,
        protocol="tcp",
        remote_endpoints=[endpoint],
    )

    try:
        await agent.initialize()
        await agent.register_capabilities(["core"])

        peers = await agent.discover_peers(query_type="all")
        peer_ids = {peer.agent_id for peer in peers}
        assert "remote_agent" in peer_ids

        success = await agent.send_message(
            recipient_id="remote_agent",
            payload={"text": "Reply with exactly one word: pong"},
            secure=False,
        )
        assert success is True

        assert received_run_requests, "No /runs request sent to remote ACP peer"
        run_request = received_run_requests[-1]
        assert run_request["agent_name"] == "remote_agent"
        assert run_request["mode"] == "sync"
        assert run_request["input"][-1]["role"] == "user"
        assert (
            run_request["input"][-1]["parts"][-1]["content"]
            == "Reply with exactly one word: pong"
        )

        assert agent.remote_run_history, "Expected recorded remote run history"
        latest = agent.remote_run_history[-1]
        assert latest["response"]["run_id"] == "mock-run-id"
        print("✓ Remote HTTP ACP discovery and messaging works")
        print()
    finally:
        await agent.shutdown()
        await runner.cleanup()


async def run_all_tests() -> int:
    """Run ACP remote HTTP tests."""
    try:
        await test_remote_http_peer_flow()
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    print("All ACP remote HTTP tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_all_tests()))
