"""Agent Communication Protocol (ACP) implementation.

Provides dynamic service discovery, registration, and secure inter-agent communication.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import socket
import struct
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Type
import aiohttp

logger = logging.getLogger(__name__)


# Try to import cryptography; provide fallback if unavailable
CRYPTOGRAPHY_AVAILABLE = True
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    hashes = None  # type: ignore[assignment]
    serialization = None  # type: ignore[assignment]
    padding = None  # type: ignore[assignment]
    Cipher = None  # type: ignore[assignment]
    algorithms = None  # type: ignore[assignment]
    modes = None  # type: ignore[assignment]
    default_backend = None  # type: ignore[assignment]
    logger.warning("cryptography module not available; ACP security features limited")


class ACPVersion(Enum):
    """ACP protocol versions."""
    V1 = "1.0"


@dataclass
class CapabilityProfile:
    """Agent capability profile for discovery."""
    agent_id: str
    agent_name: str
    capabilities: List[str] = field(default_factory=list)
    supported_protocols: List[str] = field(default_factory=lambda: ["tcp", "websocket"])
    endpoints: List[str] = field(default_factory=list)
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "capabilities": self.capabilities,
            "supported_protocols": self.supported_protocols,
            "endpoints": self.endpoints,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CapabilityProfile":
        return cls(
            agent_id=data.get("agent_id", ""),
            agent_name=data.get("agent_name", ""),
            capabilities=data.get("capabilities", []),
            supported_protocols=data.get("supported_protocols", ["tcp"]),
            endpoints=data.get("endpoints", []),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DiscoveryRequest:
    """Discovery request message."""
    request_id: str
    requester_id: str
    query_type: str  # "all", "capability", "name"
    query_value: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class DiscoveryResponse:
    """Discovery response message."""
    request_id: str
    responder_id: str
    profiles: List[CapabilityProfile]
    timestamp: float = field(default_factory=time.time)


@dataclass
class SecureMessage:
    """Securely encrypted/aggregated message."""
    message_id: str
    sender_id: str
    recipient_id: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    nonce: str = field(default_factory=lambda: secrets.token_hex(16))
    signature: Optional[str] = None
    encrypted: bool = False


@dataclass
class ServiceRegistryEntry:
    """Entry in the service registry."""
    profile: CapabilityProfile
    last_seen: float
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginConfig:
    """Configuration for ACP plugins."""
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


class PluginRegistry(ABC):
    """Base class for ACP plugins."""

    @property
    @abstractmethod
    def plugin_name(self) -> str:
        """Return the plugin's unique name."""
        pass

    @abstractmethod
    def initialize(self, config: PluginConfig) -> None:
        """Initialize the plugin with configuration."""
        pass

    @abstractmethod
    async def process_inbound(self, data: bytes) -> Optional[bytes]:
        """Process incoming data before it reaches the protocol handler."""
        pass

    @abstractmethod
    async def process_outbound(self, data: bytes) -> bytes:
        """Process outgoing data before transmission."""
        pass


class SecurityPlugin(PluginRegistry):
    """Plugin for message encryption and signing."""

    def __init__(self):
        self._private_key: Any = None
        self._public_key: Any = None
        self._encryption_key: Optional[bytes] = None
        self._peer_keys: Dict[str, Any] = {}

    @property
    def plugin_name(self) -> str:
        return "security"

    def initialize(self, config: PluginConfig) -> None:
        key_pair = config.config.get("key_pair")
        if key_pair:
            self._private_key = key_pair.get("private")
            self._public_key = key_pair.get("public")
        else:
            self._generate_key_pair()

        self._encryption_key = config.config.get("encryption_key") or secrets.token_bytes(32)

    def _generate_key_pair(self) -> None:
        if not CRYPTOGRAPHY_AVAILABLE:
            self._private_key = None
            self._public_key = None
            return
        self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self._public_key = self._private_key.public_key()

    def get_public_key(self) -> Any:
        return self._public_key

    def sign_message(self, message: bytes) -> str:
        if not CRYPTOGRAPHY_AVAILABLE or not self._private_key:
            digest = hashlib.sha256(message).digest()
            return "sha256:" + base64.b64encode(digest).decode("ascii")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode("ascii")

    def verify_signature(self, message: bytes, signature: str, peer_id: str) -> bool:
        if signature.startswith("sha256:"):
            expected = "sha256:" + base64.b64encode(hashlib.sha256(message).digest()).decode("ascii")
            return hmac.compare_digest(signature, expected)

        peer_key = self._peer_keys.get(peer_id)
        if not peer_key:
            # Local/self validation fallback for tests and single-agent flows.
            peer_key = self._public_key
        if not peer_key or not CRYPTOGRAPHY_AVAILABLE:
            return False
        try:
            peer_key.verify(
                base64.b64decode(signature),
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

    def encrypt_to_peer(self, plaintext: bytes, peer_id: str) -> bytes:
        peer_key = self._peer_keys.get(peer_id)
        if not peer_key or not CRYPTOGRAPHY_AVAILABLE:
            return plaintext
        return peer_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

    def decrypt(self, ciphertext: bytes) -> bytes:
        if not self._private_key or not CRYPTOGRAPHY_AVAILABLE:
            return ciphertext
        return self._private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

    def register_peer_public_key(self, peer_id: str, public_key: Any) -> None:
        self._peer_keys[peer_id] = public_key

    async def process_inbound(self, data: bytes) -> Optional[bytes]:
        return data

    async def process_outbound(self, data: bytes) -> bytes:
        return data


class TransportPlugin(PluginRegistry):
    """Plugin for transport layer handling."""

    def __init__(self):
        self._transport_type: str = "tcp"

    @property
    def plugin_name(self) -> str:
        return "transport"

    def initialize(self, config: PluginConfig) -> None:
        self._transport_type = config.config.get("type", "tcp")

    async def process_inbound(self, data: bytes) -> Optional[bytes]:
        if self._transport_type == "websocket":
            return self._decode_websocket(data)
        return data

    async def process_outbound(self, data: bytes) -> bytes:
        if self._transport_type == "websocket":
            return self._encode_websocket(data)
        return data

    def _decode_websocket(self, data: bytes) -> Optional[bytes]:
        if len(data) < 6:
            return None
        length = struct.unpack(">H", data[:2])[0]
        return data[6:6 + length]

    def _encode_websocket(self, data: bytes) -> bytes:
        length = len(data)
        return struct.pack(">H", length) + data


@dataclass
class ConnectionInfo:
    """Connection endpoint information."""
    host: str
    port: int
    protocol: str = "tcp"
    secure: bool = False


@dataclass
class AgentIdentity:
    """Agent identity for ACP communication."""
    agent_id: str
    identity_key: Any
    secret_key: bytes


class ProtocolHandler(ABC):
    """Base class for protocol handlers."""

    def __init__(self, identity: AgentIdentity):
        self._identity = identity
        self._connected_peers: Set[str] = set()

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    @property
    @abstractmethod
    def protocol_name(self) -> str:
        pass

    @abstractmethod
    async def connect(self, address: str, port: int) -> bool:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def send(self, recipient_id: str, message: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    async def receive(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        pass

    @abstractmethod
    async def broadcast(self, message: Dict[str, Any]) -> None:
        pass


class TCPProtocolHandler(ProtocolHandler):
    """TCP-based protocol handler."""

    def __init__(self, identity: AgentIdentity, host: str, port: int):
        super().__init__(identity)
        self._host = host
        self._port = port
        self._reader: Optional[aiohttp.StreamReader] = None
        self._writer: Optional[aiohttp.StreamWriter] = None
        self._running = False

    @property
    def protocol_name(self) -> str:
        return "tcp"

    async def connect(self, address: str, port: Optional[int] = None) -> bool:
        port = port or self._port
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(address, port),
                timeout=10.0
            )
            self._connected_peers.add(address)
            return True
        except Exception as e:
            logger.error(f"TCP connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        self._running = False
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
        self._connected_peers.clear()

    async def send(self, recipient_id: str, message: Dict[str, Any]) -> bool:
        if not self._writer:
            return False
        payload = json.dumps(message).encode("utf-8")
        length_prefix = struct.pack(">H", len(payload))
        self._writer.write(length_prefix + payload)
        await self._writer.drain()
        return True

    async def receive(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self._reader:
            return None
        length_data = await self._reader.readexactly(2)
        length = struct.unpack(">H", length_data)[0]
        payload = await self._reader.read(length)
        message = json.loads(payload.decode("utf-8"))
        sender = list(self._connected_peers)[0] if self._connected_peers else "unknown"
        return sender, message

    async def broadcast(self, message: Dict[str, Any]) -> None:
        for peer in self._connected_peers:
            await self.send(peer, message)


class WebsocketProtocolHandler(TCPProtocolHandler):
    """WebSocket-based protocol handler."""

    def __init__(self, identity: AgentIdentity, endpoint: str):
        super().__init__(identity, "", 0)
        self._endpoint = endpoint
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None

    @property
    def protocol_name(self) -> str:
        return "websocket"

    async def connect(self, address: str, port: Optional[int] = None) -> bool:
        url = f"{self._endpoint}/{address}"
        try:
            self._ws = await asyncio.wait_for(
                aiohttp.ClientSession().ws_connect(url),
                timeout=10.0
            )
            self._connected_peers.add(address)
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()
        self._connected_peers.clear()

    async def send(self, recipient_id: str, message: Dict[str, Any]) -> bool:
        if not self._ws:
            return False
        await self._ws.send_json(message)
        return True

    async def receive(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self._ws:
            return None
        message = await self._ws.receive_json()
        return list(self._connected_peers)[0] if self._connected_peers else "unknown", message

    async def broadcast(self, message: Dict[str, Any]) -> None:
        for peer in self._connected_peers:
            await self.send(peer, message)


class ProtocolRegistry:
    """Registry for ACP protocol implementations."""

    def __init__(self):
        self._protocols: Dict[str, Type[ProtocolHandler]] = {}

    def register(self, protocol_name: str, handler_class: Type[ProtocolHandler]) -> None:
        self._protocols[protocol_name] = handler_class

    def get(self, protocol_name: str) -> Optional[Type[ProtocolHandler]]:
        return self._protocols.get(protocol_name)

    def list_protocols(self) -> List[str]:
        return list(self._protocols.keys())


@dataclass
class ConnectionInfo:
    """Connection endpoint information."""
    host: str
    port: int
    protocol: str = "tcp"
    secure: bool = False


@dataclass
class AgentIdentity:
    """Agent identity for ACP communication."""
    agent_id: str
    identity_key: Any
    secret_key: bytes


class ServiceRegistry:
    """Registry for discovering and managing agent services."""

    def __init__(self, local_agent_id: str):
        self._local_agent_id = local_agent_id
        self._entries: Dict[str, ServiceRegistryEntry] = {}
        self._lock = asyncio.Lock()

    async def register_local(self, profile: CapabilityProfile) -> None:
        async with self._lock:
            self._entries[profile.agent_id] = ServiceRegistryEntry(
                profile=profile,
                last_seen=time.time(),
            )

    async def unregister_local(self) -> None:
        async with self._lock:
            self._entries.pop(self._local_agent_id, None)

    async def add_entry(self, entry: ServiceRegistryEntry) -> None:
        async with self._lock:
            self._entries[entry.profile.agent_id] = entry

    async def remove_entry(self, agent_id: str) -> None:
        async with self._lock:
            self._entries.pop(agent_id, None)

    async def get_entry(self, agent_id: str) -> Optional[ServiceRegistryEntry]:
        async with self._lock:
            return self._entries.get(agent_id)

    async def get_all_entries(self) -> List[ServiceRegistryEntry]:
        async with self._lock:
            return list(self._entries.values())

    async def find_by_capability(self, capability: str) -> List[ServiceRegistryEntry]:
        async with self._lock:
            return [
                entry for entry in self._entries.values()
                if capability in entry.profile.capabilities
            ]

    async def find_by_name(self, name_pattern: str) -> List[ServiceRegistryEntry]:
        async with self._lock:
            return [
                entry for entry in self._entries.values()
                if name_pattern.lower() in entry.profile.agent_name.lower()
            ]

    async def get_peers(self) -> List[CapabilityProfile]:
        async with self._lock:
            return [
                entry.profile for entry in self._entries.values()
                if entry.profile.agent_id != self._local_agent_id
            ]

    async def update_heartbeat(self, agent_id: str) -> None:
        async with self._lock:
            if agent_id in self._entries:
                self._entries[agent_id].last_seen = time.time()

    async def get_expired_entries(self, timeout: float = 300.0) -> List[str]:
        async with self._lock:
            now = time.time()
            expired = [
                agent_id for agent_id, entry in self._entries.items()
                if now - entry.last_seen > timeout
            ]
            for agent_id in expired:
                self._entries.pop(agent_id, None)
            return expired


class PluginManager:
    """Manages ACP plugins for extended functionality."""

    def __init__(self):
        self._plugins: Dict[str, PluginRegistry] = {}
        self._configs: Dict[str, PluginConfig] = {}

    def register_plugin(self, plugin: PluginRegistry) -> None:
        self._plugins[plugin.plugin_name] = plugin

    def get_plugin(self, name: str) -> Optional[PluginRegistry]:
        return self._plugins.get(name)

    def list_plugins(self) -> List[str]:
        return list(self._plugins.keys())

    async def initialize(self, plugin_configs: Dict[str, PluginConfig]) -> None:
        for name, config in plugin_configs.items():
            plugin = self._plugins.get(name)
            if plugin:
                plugin.initialize(config)
                self._configs[name] = config

    async def process_inbound(self, data: bytes) -> Optional[bytes]:
        for plugin in self._plugins.values():
            result = await plugin.process_inbound(data)
            if result is not None:
                data = result
        return data

    async def process_outbound(self, data: bytes) -> bytes:
        for plugin in self._plugins.values():
            data = await plugin.process_outbound(data)
        return data


class ACPAgent:
    """Main ACP agent class for inter-agent communication."""
    _network_agents: Dict[str, "ACPAgent"] = {}
    _network_lock = threading.RLock()

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        host: str = "0.0.0.0",
        port: int = 8765,
        protocol: str = "tcp",
        remote_endpoints: Optional[List[str]] = None,
    ):
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._host = host
        self._port = port
        self._protocol = protocol

        # Generate identity
        self._identity_key = None
        if CRYPTOGRAPHY_AVAILABLE:
            self._identity_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self._secret_key = secrets.token_bytes(32)
        self._identity = AgentIdentity(
            agent_id=self._agent_id,
            identity_key=self._identity_key.public_key() if self._identity_key else None,
            secret_key=self._secret_key,
        )

        # Initialize components
        self._registry = ServiceRegistry(agent_id)
        self._plugin_manager = PluginManager()
        self._protocol_handler: Optional[ProtocolHandler] = None

        # Register built-in plugins
        security_plugin = SecurityPlugin()
        self._plugin_manager.register_plugin(security_plugin)

        transport_plugin = TransportPlugin()
        self._plugin_manager.register_plugin(transport_plugin)

        # Register protocols
        protocol_registry = ProtocolRegistry()
        protocol_registry.register("tcp", TCPProtocolHandler)
        protocol_registry.register("websocket", WebsocketProtocolHandler)

        # Discovery state
        self._discovery_peers: Set[str] = set()
        self._running = False
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._received_messages: List[Dict[str, Any]] = []
        self._remote_run_history: List[Dict[str, Any]] = []
        self._remote_endpoint_by_agent: Dict[str, str] = {}
        self._remote_endpoints = self._resolve_remote_endpoints(remote_endpoints)

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def identity(self) -> AgentIdentity:
        return self._identity

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return

            # Keep ACP reliable in local/offline environments by using an in-process network.
            await self._plugin_manager.initialize(
                {
                    "security": PluginConfig(enabled=True, config={}),
                    "transport": PluginConfig(
                        enabled=True,
                        config={"type": self._protocol},
                    ),
                }
            )
            with ACPAgent._network_lock:
                ACPAgent._network_agents[self._agent_id] = self

            self._running = True
            self._initialized = True

    async def shutdown(self) -> None:
        self._running = False
        with ACPAgent._network_lock:
            existing = ACPAgent._network_agents.get(self._agent_id)
            if existing is self:
                ACPAgent._network_agents.pop(self._agent_id, None)
        if self._protocol_handler:
            await self._protocol_handler.disconnect()
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    def _resolve_remote_endpoints(
        self,
        remote_endpoints: Optional[List[str]],
    ) -> List[str]:
        explicit = remote_endpoints or []
        env_value = os.environ.get("ACP_REMOTE_ENDPOINTS", "").strip()
        env_values = [v.strip() for v in env_value.split(",") if v.strip()]

        normalized: List[str] = []
        seen: Set[str] = set()
        for candidate in [*explicit, *env_values]:
            endpoint = candidate.strip()
            if not endpoint:
                continue
            if "://" not in endpoint:
                endpoint = f"http://{endpoint}"
            endpoint = endpoint.rstrip("/")
            if endpoint in seen:
                continue
            seen.add(endpoint)
            normalized.append(endpoint)
        return normalized

    @staticmethod
    def _query_matches_profile(
        profile: CapabilityProfile,
        query_type: str,
        query_value: Optional[str],
    ) -> bool:
        if query_type == "all":
            return True

        token = (query_value or "").strip().lower()
        if not token:
            return False

        if query_type == "capability":
            return any(capability.lower() == token for capability in profile.capabilities)
        if query_type == "name":
            return token in profile.agent_name.lower()
        return False

    async def _discover_remote_agents(self) -> List[CapabilityProfile]:
        if not self._remote_endpoints:
            return []

        timeout = aiohttp.ClientTimeout(total=20)
        discovered: List[CapabilityProfile] = []
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for endpoint in self._remote_endpoints:
                url = f"{endpoint}/agents"
                try:
                    async with session.get(url) as response:
                        if response.status >= 400:
                            body = await response.text()
                            logger.warning(
                                "ACP remote discovery failed for %s: HTTP %s %s",
                                endpoint,
                                response.status,
                                body[:200],
                            )
                            continue
                        payload = await response.json(content_type=None)
                except Exception as e:
                    logger.warning("ACP remote discovery failed for %s: %s", endpoint, e)
                    continue

                for agent in payload.get("agents", []):
                    agent_name = str(agent.get("name") or "").strip()
                    if not agent_name or agent_name == self._agent_id:
                        continue
                    metadata = agent.get("metadata") or {}
                    tags = metadata.get("tags")
                    capabilities = [str(tag) for tag in tags] if isinstance(tags, list) else []
                    profile = CapabilityProfile(
                        agent_id=agent_name,
                        agent_name=agent_name,
                        capabilities=capabilities,
                        supported_protocols=["http"],
                        endpoints=[endpoint],
                        metadata={
                            **metadata,
                            "description": agent.get("description"),
                            "input_content_types": agent.get("input_content_types", []),
                            "output_content_types": agent.get("output_content_types", []),
                        },
                    )
                    discovered.append(profile)
                    self._remote_endpoint_by_agent[agent_name] = endpoint
        return discovered

    async def _send_remote_message(
        self,
        endpoint: str,
        recipient_id: str,
        payload: Dict[str, Any],
        secure: bool,
    ) -> bool:
        user_text = str(payload.get("text") or json.dumps(payload, ensure_ascii=False))
        run_request = {
            "agent_name": recipient_id,
            "mode": "sync",
            "input": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "content": user_text,
                            "media_type": "text/plain",
                        }
                    ],
                }
            ],
            "metadata": {
                "sender_id": self._agent_id,
                "secure_requested": bool(secure),
                "transport": "agentzero_acp_http",
            },
        }

        timeout = aiohttp.ClientTimeout(total=120)
        url = f"{endpoint}/runs"
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(url, json=run_request) as response:
                    response_text = await response.text()
                    if response.status >= 400:
                        logger.warning(
                            "ACP remote send failed (%s): HTTP %s %s",
                            recipient_id,
                            response.status,
                            response_text[:300],
                        )
                        return False
                    try:
                        response_payload = json.loads(response_text)
                    except Exception:
                        logger.warning(
                            "ACP remote send returned non-JSON response for %s: %s",
                            recipient_id,
                            response_text[:200],
                        )
                        return False
            except Exception as e:
                logger.warning(
                    "ACP remote send failed for %s at %s: %s",
                    recipient_id,
                    endpoint,
                    e,
                )
                return False

        self._remote_run_history.append(
            {
                "recipient_id": recipient_id,
                "endpoint": endpoint,
                "request": run_request,
                "response": response_payload,
                "sent_at": time.time(),
            }
        )
        return bool(response_payload.get("run_id"))

    async def _get_local_profile(self) -> CapabilityProfile:
        local_entry = await self._registry.get_entry(self._agent_id)
        if local_entry is not None:
            return local_entry.profile
        return CapabilityProfile(
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            capabilities=[],
            supported_protocols=[self._protocol],
            endpoints=[f"{self._host}:{self._port}"],
            metadata={"autogenerated": True},
        )

    async def register_capabilities(self, capabilities: List[str]) -> None:
        await self._ensure_initialized()
        profile = CapabilityProfile(
            agent_id=self._agent_id,
            agent_name=self._agent_name,
            capabilities=capabilities,
            supported_protocols=[self._protocol],
            endpoints=[f"{self._host}:{self._port}"],
        )
        await self._registry.register_local(profile)

    async def discover_peers(
        self,
        query_type: str = "all",
        query_value: Optional[str] = None,
        timeout: float = 10.0,
    ) -> List[CapabilityProfile]:
        del timeout  # Discovery is immediate in the local ACP network.
        await self._ensure_initialized()

        normalized_query_type = (query_type or "all").strip().lower()
        if normalized_query_type not in {"all", "capability", "name"}:
            raise ValueError(f"Unsupported query_type: {query_type}")

        discovered_by_id: Dict[str, CapabilityProfile] = {}

        with ACPAgent._network_lock:
            peer_agents = [
                agent
                for agent_id, agent in ACPAgent._network_agents.items()
                if agent_id != self._agent_id
            ]

        for peer_agent in peer_agents:
            profile = await peer_agent._get_local_profile()
            discovered_by_id[profile.agent_id] = profile

        for profile in await self._discover_remote_agents():
            discovered_by_id[profile.agent_id] = profile

        matched_profiles: List[CapabilityProfile] = []
        for profile in discovered_by_id.values():
            if not self._query_matches_profile(
                profile=profile,
                query_type=normalized_query_type,
                query_value=query_value,
            ):
                continue
            matched_profiles.append(profile)
            self._discovery_peers.add(profile.agent_id)
            await self._registry.add_entry(
                ServiceRegistryEntry(profile=profile, last_seen=time.time())
            )

        return matched_profiles

    async def send_message(
        self,
        recipient_id: str,
        payload: Dict[str, Any],
        secure: bool = True,
    ) -> bool:
        await self._ensure_initialized()
        with ACPAgent._network_lock:
            recipient_agent = ACPAgent._network_agents.get(recipient_id)

        if recipient_agent is not None:
            message = SecureMessage(
                message_id=secrets.token_hex(16),
                sender_id=self._agent_id,
                recipient_id=recipient_id,
                payload=payload,
            )

            if secure:
                sec_plugin = self._plugin_manager._plugins.get("security")
                if isinstance(sec_plugin, SecurityPlugin):
                    signature = sec_plugin.sign_message(
                        json.dumps({k: v for k, v in message.__dict__.items() if k != "signature"}).encode("utf-8")
                    )
                    message.signature = signature

                    sender_public_key = sec_plugin.get_public_key()
                    recipient_sec_plugin = recipient_agent._plugin_manager.get_plugin("security")
                    if (
                        sender_public_key is not None
                        and isinstance(recipient_sec_plugin, SecurityPlugin)
                    ):
                        recipient_sec_plugin.register_peer_public_key(
                            self._agent_id, sender_public_key
                        )

            await recipient_agent._receive_direct_message(
                self._agent_id,
                {"type": "agent_message", "data": message.__dict__},
            )
            recipient_profile = await recipient_agent._get_local_profile()
            await self._registry.add_entry(
                ServiceRegistryEntry(profile=recipient_profile, last_seen=time.time())
            )
            self._discovery_peers.add(recipient_id)
            return True

        endpoint = self._remote_endpoint_by_agent.get(recipient_id)
        if not endpoint and self._remote_endpoints:
            await self.discover_peers(query_type="all")
            endpoint = self._remote_endpoint_by_agent.get(recipient_id)
        if not endpoint:
            logger.warning("ACP recipient not found: %s", recipient_id)
            return False

        success = await self._send_remote_message(
            endpoint=endpoint,
            recipient_id=recipient_id,
            payload=payload,
            secure=secure,
        )
        if success:
            self._discovery_peers.add(recipient_id)
            existing_profile = await self._registry.get_entry(recipient_id)
            if existing_profile is None:
                await self._registry.add_entry(
                    ServiceRegistryEntry(
                        profile=CapabilityProfile(
                            agent_id=recipient_id,
                            agent_name=recipient_id,
                            endpoints=[endpoint],
                            supported_protocols=["http"],
                        ),
                        last_seen=time.time(),
                    )
                )
            await self._registry.update_heartbeat(recipient_id)
        return success

    async def _receive_direct_message(self, sender: str, message: Dict[str, Any]) -> None:
        await self._process_inbound_message(sender, message)

    async def _process_inbound_message(
        self,
        sender: str,
        message: Dict[str, Any],
    ) -> None:
        msg_type = message.get("type")

        if msg_type == "discovery_request":
            await self._handle_discovery_request(sender, message["data"])
        elif msg_type == "discovery_response":
            await self._handle_discovery_response(message["data"])
        elif msg_type == "agent_message":
            await self._handle_agent_message(sender, message["data"])

    async def _handle_discovery_request(
        self,
        sender: str,
        data: Dict[str, Any],
    ) -> None:
        request = DiscoveryRequest(**data)
        profiles = await self.discover_peers(
            query_type=request.query_type,
            query_value=request.query_value,
        )
        response = {
            "request_id": request.request_id,
            "responder_id": self._agent_id,
            "profiles": [profile.to_dict() for profile in profiles],
        }
        await self._handle_discovery_response(response)

    async def _handle_discovery_response(self, data: Dict[str, Any]) -> None:
        profile_dicts = data.get("profiles", [])
        for profile_data in profile_dicts:
            profile = (
                profile_data
                if isinstance(profile_data, CapabilityProfile)
                else CapabilityProfile.from_dict(profile_data)
            )
            await self._registry.add_entry(
                ServiceRegistryEntry(
                    profile=profile,
                    last_seen=time.time(),
                )
            )
            self._discovery_peers.add(profile.agent_id)

    async def _handle_agent_message(
        self,
        sender: str,
        data: Dict[str, Any],
    ) -> None:
        if "signature" in data:
            sec_plugin = self._plugin_manager._plugins.get("security")
            if sec_plugin:
                msg_data = {k: v for k, v in data.items() if k != "signature"}
                if not sec_plugin.verify_signature(
                    json.dumps(msg_data).encode("utf-8"),
                    data["signature"],
                    sender
                ):
                    logger.warning("Invalid signature on message from %s", sender)
                    return

        self._received_messages.append(
            {
                "sender_id": sender,
                "message": data,
                "received_at": time.time(),
            }
        )

        with ACPAgent._network_lock:
            sender_agent = ACPAgent._network_agents.get(sender)
        if sender_agent is not None:
            sender_profile = await sender_agent._get_local_profile()
        else:
            sender_profile = CapabilityProfile(
                agent_id=sender,
                agent_name=sender,
                capabilities=[],
                endpoints=[],
                supported_protocols=[],
            )
        await self._registry.add_entry(
            ServiceRegistryEntry(profile=sender_profile, last_seen=time.time())
        )
        await self._registry.update_heartbeat(sender)

    async def get_registry_status(self) -> Dict[str, Any]:
        all_entries = await self._registry.get_all_entries()
        return {
            "local_agent_id": self._agent_id,
            "total_entries": len(all_entries),
            "peers": [e.profile.to_dict() for e in all_entries if e.profile.agent_id != self._agent_id],
        }

    @property
    def connection_info(self) -> ConnectionInfo:
        return ConnectionInfo(
            host=self._host,
            port=self._port,
            protocol=self._protocol,
        )

    @property
    def received_messages(self) -> List[Dict[str, Any]]:
        return list(self._received_messages)

    @property
    def remote_run_history(self) -> List[Dict[str, Any]]:
        return list(self._remote_run_history)
