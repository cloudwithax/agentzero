"""Test ACP (Agent Communication Protocol) functionality."""

import asyncio
import sys
import os
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acp import (
    ACPAgent,
    CapabilityProfile,
    CRYPTOGRAPHY_AVAILABLE,
    ServiceRegistry,
    ServiceRegistryEntry,
    SecurityPlugin,
    PluginRegistry,
    PluginManager,
    PluginConfig,
)


async def test_acp_basic():
    """Test basic ACP agent initialization and operations."""
    print("Test 1: Basic ACP Agent Initialization")
    print("-" * 60)

    agent = ACPAgent(
        agent_id="test_agent_1",
        agent_name="Test Agent 1",
        host="127.0.0.1",
        port=8766,
        protocol="tcp",
    )

    assert agent.agent_id == "test_agent_1"
    assert agent.agent_name == "Test Agent 1"
    assert agent._port == 8766
    assert agent._protocol == "tcp"

    print("✓ Agent initialized with correct properties")

    await agent.initialize()
    print("✓ Agent initialized successfully")

    await agent.shutdown()
    print("✓ Agent shutdown successfully")
    print()


async def test_capability_profile():
    """Test CapabilityProfile serialization/deserialization."""
    print("Test 2: CapabilityProfile Serialization")
    print("-" * 60)

    profile = CapabilityProfile(
        agent_id="test_agent_1",
        agent_name="Test Agent 1",
        capabilities=["memory_access", "web_search", "tool_execution"],
        supported_protocols=["tcp", "websocket"],
        endpoints=["127.0.0.1:8766"],
        version="1.0",
        metadata={"version": "1.0", "stable": True},
    )

    profile_dict = profile.to_dict()
    assert profile_dict["agent_id"] == "test_agent_1"
    assert profile_dict["agent_name"] == "Test Agent 1"
    assert "memory_access" in profile_dict["capabilities"]

    restored_profile = CapabilityProfile.from_dict(profile_dict)
    assert restored_profile.agent_id == "test_agent_1"
    assert restored_profile.agent_name == "Test Agent 1"

    print("✓ Profile serialization/deserialization works")
    print()


async def test_service_registry():
    """Test ServiceRegistry operations."""
    print("Test 3: Service Registry Operations")
    print("-" * 60)

    registry = ServiceRegistry("test_agent_1")

    profile = CapabilityProfile(
        agent_id="test_agent_2",
        agent_name="Test Agent 2",
        capabilities=["memory_access"],
    )

    entry = ServiceRegistryEntry(profile=profile, last_seen=1234567890.0)

    await registry.add_entry(entry)
    assert await registry.get_entry("test_agent_2") is not None

    all_entries = await registry.get_all_entries()
    assert len(all_entries) == 1

    peers = await registry.get_peers()
    assert len(peers) == 1
    assert peers[0].agent_id == "test_agent_2"

    await registry.remove_entry("test_agent_2")
    assert await registry.get_entry("test_agent_2") is None

    print("✓ Service registry operations work correctly")
    print()


async def test_security_plugin():
    """Test SecurityPlugin for message signing/verification."""
    print("Test 4: Security Plugin")
    print("-" * 60)

    plugin = SecurityPlugin()
    plugin.initialize(PluginConfig(enabled=True, config={}))

    message = b"Test message content"
    signature = plugin.sign_message(message)

    assert signature is not None
    assert len(signature) > 0

    is_valid = plugin.verify_signature(message, signature, "test_agent")
    assert is_valid is True

    is_invalid = plugin.verify_signature(message, "invalid_signature", "test_agent")
    assert is_invalid is False

    print("✓ Security plugin signing/verification works")
    print()


async def test_plugin_system():
    """Test PluginRegistry and PluginManager."""
    print("Test 5: Plugin System")
    print("-" * 60)

    class TestPlugin(PluginRegistry):
        def __init__(self, name: str):
            self.name = name

        @property
        def plugin_name(self) -> str:
            return self.name

        def initialize(self, config: PluginConfig) -> None:
            pass

        async def process_inbound(self, data: bytes) -> Optional[bytes]:
            return data

        async def process_outbound(self, data: bytes) -> bytes:
            return data

    plugin_manager = PluginManager()
    test_plugin = TestPlugin("test_plugin")
    plugin_manager.register_plugin(test_plugin)

    assert plugin_manager.get_plugin("test_plugin") is test_plugin
    assert "test_plugin" in plugin_manager.list_plugins()

    print("✓ Plugin system works correctly")
    print()


async def test_acp_registration():
    """Test registering capabilities with ACP."""
    print("Test 6: ACP Capability Registration")
    print("-" * 60)

    agent = ACPAgent(
        agent_id="test_agent_3",
        agent_name="Test Agent 3",
        host="127.0.0.1",
        port=8767,
        protocol="tcp",
    )

    await agent.initialize()

    capabilities = ["memory_access", "web_search", "tool_execution"]
    await agent.register_capabilities(capabilities)

    registry_status = await agent.get_registry_status()
    assert registry_status["local_agent_id"] == "test_agent_3"
    assert len(registry_status["peers"]) == 0

    print("✓ Capability registration works")
    await agent.shutdown()
    print()


async def test_discovery_protocol():
    """Test discovery request/response protocol."""
    print("Test 7: Discovery Protocol")
    print("-" * 60)

    agent1 = ACPAgent(
        agent_id="agent1",
        agent_name="Agent 1",
        host="127.0.0.1",
        port=8768,
        protocol="tcp",
    )

    agent2 = ACPAgent(
        agent_id="agent2",
        agent_name="Agent 2",
        host="127.0.0.1",
        port=8769,
        protocol="tcp",
    )

    await agent1.initialize()
    await agent2.initialize()

    await agent1.register_capabilities(["memory_access"])
    await agent2.register_capabilities(["web_search"])

    # Agent 1 should discover Agent 2
    peers = await agent1.discover_peers()
    assert len(peers) == 1
    assert peers[0].agent_id == "agent2"

    print("✓ Discovery protocol works")
    await agent1.shutdown()
    await agent2.shutdown()
    print()


async def test_message_sending():
    """Test secure message sending between agents."""
    print("Test 8: Secure Message Sending")
    print("-" * 60)

    agent1 = ACPAgent(
        agent_id="sender",
        agent_name="Sender Agent",
        host="127.0.0.1",
        port=8770,
        protocol="tcp",
    )

    agent2 = ACPAgent(
        agent_id="receiver",
        agent_name="Receiver Agent",
        host="127.0.0.1",
        port=8771,
        protocol="tcp",
    )

    await agent1.initialize()
    await agent2.initialize()

    await agent1.register_capabilities(["test"])

    payload = {
        "text": "Hello from Agent 1!",
        "metadata": {"timestamp": "test"},
    }

    success = await agent1.send_message("receiver", payload, secure=True)
    assert success is True

    print("✓ Secure message sending works")
    await agent1.shutdown()
    await agent2.shutdown()
    print()


async def test_connection_info():
    """Test connection information retrieval."""
    print("Test 9: Connection Information")
    print("-" * 60)

    agent = ACPAgent(
        agent_id="test_agent",
        agent_name="Test Agent",
        host="0.0.0.0",
        port=8772,
        protocol="tcp",
    )

    info = agent.connection_info
    assert info.host == "0.0.0.0"
    assert info.port == 8772
    assert info.protocol == "tcp"

    print("✓ Connection information retrieval works")
    print()


async def test_identity():
    """Test agent identity generation."""
    print("Test 10: Agent Identity")
    print("-" * 60)

    agent = ACPAgent(
        agent_id="test_agent",
        agent_name="Test Agent",
        host="127.0.0.1",
        port=8773,
        protocol="tcp",
    )

    identity = agent.identity
    assert identity.agent_id == "test_agent"
    if CRYPTOGRAPHY_AVAILABLE:
        assert identity.identity_key is not None
    else:
        assert identity.identity_key is None
    assert identity.secret_key is not None

    print("✓ Agent identity generation works")
    print()


async def run_all_tests():
    """Run all ACP tests."""
    print("=" * 60)
    print("ACAgent Communication Protocol (ACP) Tests")
    print("=" * 60)
    print()

    tests = [
        test_acp_basic,
        test_capability_profile,
        test_service_registry,
        test_security_plugin,
        test_plugin_system,
        test_acp_registration,
        test_discovery_protocol,
        test_message_sending,
        test_connection_info,
        test_identity,
    ]

    failed_tests = []
    for test in tests:
        try:
            await test()
        except Exception as e:
            print(f"✗ Test failed: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed_tests.append(test.__name__)
        print()

    print("=" * 60)
    if failed_tests:
        print(f"Tests failed: {', '.join(failed_tests)}")
        return 1
    else:
        print("All tests passed!")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)
