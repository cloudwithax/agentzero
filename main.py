"""Main entry point for the agent framework."""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from memory import EnhancedMemoryStore  # noqa: E402
from capabilities import Capability, CapabilityProfile, AdaptiveFormatter  # noqa: E402
from examples import ExampleBank, AdaptiveFewShotManager  # noqa: E402
from planning import TaskPlanner, TaskAnalyzer  # noqa: E402
from skills import SkillRegistry  # noqa: E402
from tools import set_memory_store, set_skill_registry, set_acp_agent  # noqa: E402
from handler import AgentHandler  # noqa: E402
from integrations import run_telegram_bot_async, start_sendblue_bot  # noqa: E402
from openai_compat_server import start_openai_compatible_server  # noqa: E402
from acp import ACPAgent  # noqa: E402

# Setup logging
requested_log_level = os.environ.get("LOG_LEVEL", "INFO").strip().upper()
resolved_log_level = getattr(logging, requested_log_level, None)
if not isinstance(resolved_log_level, int):
    resolved_log_level = logging.INFO

logging.basicConfig(level=resolved_log_level)
logger = logging.getLogger(__name__)

if requested_log_level and not isinstance(
    getattr(logging, requested_log_level, None), int
):
    logger.warning("Invalid LOG_LEVEL=%r, defaulting to INFO", requested_log_level)

# Configuration - API key and model ID from environment variables with fallbacks
API_KEY = os.environ.get("NVIDIA_API_KEY", "")
MODEL_ID = os.environ.get("MODEL_ID", "moonshotai/kimi-k2-instruct-0905")
PID_FILE = "agentzero.pid"


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse boolean-like environment values."""
    fallback = "1" if default else "0"
    raw_value = os.environ.get(name, fallback)
    return str(raw_value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_port(name: str, default: int) -> int:
    """Parse a TCP port from environment with fallback logging."""
    raw_value = os.environ.get(name, str(default)).strip()
    try:
        parsed = int(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%r, defaulting to %s", name, raw_value, default)
        return default

    if parsed <= 0 or parsed > 65535:
        logger.warning("Out-of-range %s=%r, defaulting to %s", name, raw_value, default)
        return default

    return parsed


def daemonize() -> None:
    """Detach the current process and continue execution in the background."""
    if os.name != "posix":
        raise RuntimeError("Daemon mode is only supported on POSIX systems")

    # First fork: parent exits, child continues.
    if os.fork() > 0:
        os._exit(0)

    os.setsid()

    # Second fork: prevent reacquiring a controlling terminal.
    if os.fork() > 0:
        os._exit(0)

    sys.stdout.flush()
    sys.stderr.flush()

    with open(os.devnull, "r", encoding="utf-8") as stdin_handle, open(
        "agentzero.out.log", "a", encoding="utf-8"
    ) as stdout_handle, open(
        "agentzero.err.log", "a", encoding="utf-8"
    ) as stderr_handle:
        os.dup2(stdin_handle.fileno(), sys.stdin.fileno())
        os.dup2(stdout_handle.fileno(), sys.stdout.fileno())
        os.dup2(stderr_handle.fileno(), sys.stderr.fileno())


def _pid_is_running(pid: int) -> bool:
    """Check whether a process ID exists."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid_file() -> int | None:
    """Read PID from PID file if it exists and is valid."""
    if not os.path.exists(PID_FILE):
        return None

    try:
        with open(PID_FILE, "r", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (ValueError, OSError):
        return None


def write_pid_file() -> None:
    """Write current daemon PID to disk, rejecting duplicate live daemon."""
    existing_pid = _read_pid_file()
    if existing_pid and _pid_is_running(existing_pid):
        raise RuntimeError(
            f"AgentZero daemon already appears to be running with PID {existing_pid}."
        )

    # Remove stale or invalid PID file before writing current PID.
    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
        except OSError:
            pass

    with open(PID_FILE, "w", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))


def _find_orphan_daemon_pids() -> list[int]:
    """Find orphaned AgentZero daemon PIDs by inspecting /proc cmdlines."""
    if os.name != "posix":
        return []

    matches: list[int] = []
    proc_dir = "/proc"
    if not os.path.isdir(proc_dir):
        return []

    for entry in os.listdir(proc_dir):
        if not entry.isdigit():
            continue

        pid = int(entry)
        if pid == os.getpid():
            continue

        cmdline_path = os.path.join(proc_dir, entry, "cmdline")
        try:
            with open(cmdline_path, "rb") as handle:
                raw_cmdline = handle.read()
        except OSError:
            continue

        if not raw_cmdline:
            continue

        argv = [
            part.decode("utf-8", errors="ignore")
            for part in raw_cmdline.split(b"\0")
            if part
        ]
        cmdline = " ".join(argv)

        if "main.py" in cmdline and "--daemon" in cmdline:
            matches.append(pid)

    return matches


def _terminate_pid(pid: int) -> bool:
    """Terminate a PID gracefully, escalating to SIGKILL if needed."""
    if not _pid_is_running(pid):
        return True

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        print(f"Failed to stop daemon PID {pid}: {exc}")
        return False

    # Wait briefly for graceful shutdown.
    for _ in range(30):
        if not _pid_is_running(pid):
            return True
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except OSError as exc:
        print(f"Failed to force-stop daemon PID {pid}: {exc}")
        return False

    for _ in range(20):
        if not _pid_is_running(pid):
            return True
        time.sleep(0.1)

    print(f"Daemon PID {pid} did not terminate after SIGKILL.")
    return False


def stop_daemon() -> int:
    """Stop the daemon process referenced by the PID file."""
    pid = _read_pid_file()
    if pid is None:
        orphan_pids = _find_orphan_daemon_pids()
        if not orphan_pids:
            print("No PID file found. Daemon does not appear to be running.")
            return 1

        failures = []
        for orphan_pid in orphan_pids:
            if _terminate_pid(orphan_pid):
                print(f"Stopped orphaned daemon PID {orphan_pid}.")
            else:
                failures.append(orphan_pid)

        if failures:
            return 1

        return 0

    if not _pid_is_running(pid):
        print(f"Stale PID file found for PID {pid}. Removing it.")
        try:
            os.remove(PID_FILE)
        except OSError:
            pass
        orphan_pids = _find_orphan_daemon_pids()
        if not orphan_pids:
            return 1

        failures = []
        for orphan_pid in orphan_pids:
            if _terminate_pid(orphan_pid):
                print(f"Stopped orphaned daemon PID {orphan_pid}.")
            else:
                failures.append(orphan_pid)

        if failures:
            return 1

        return 0

    if not _terminate_pid(pid):
        return 1

    if os.path.exists(PID_FILE):
        try:
            os.remove(PID_FILE)
        except OSError:
            pass

    print(f"Stopped daemon PID {pid}.")
    return 0


def initialize_agent() -> tuple[AgentHandler, ACPAgent]:
    """Initialize all agent components and return the handler and ACP agent."""
    # Initialize memory store
    memory_store = EnhancedMemoryStore(
        db_path="agent_memory.db",
        api_key=API_KEY,
    )

    # Set memory store for tools
    set_memory_store(memory_store)

    # Initialize Agent Skills registry
    skill_registry = SkillRegistry(project_root=os.getcwd())
    set_skill_registry(skill_registry)

    # Initialize capability profile for the model
    # Using known profile for Kimi K2 which supports all major capabilities
    capability_profile = CapabilityProfile(
        capabilities={
            Capability.JSON_OUTPUT,
            Capability.TOOL_USE,
            Capability.CHAIN_OF_THOUGHT,
            Capability.REASONING,
            Capability.LONG_CONTEXT,
            Capability.FEW_SHOT,
            Capability.SELF_CORRECTION,
            Capability.STRUCTURED_OUTPUT,
        },
        model_name=MODEL_ID,
    )

    # Initialize task planner
    task_planner = TaskPlanner(capability_profile)
    task_analyzer = TaskAnalyzer()

    # Initialize adaptive formatter
    adaptive_formatter = AdaptiveFormatter(capability_profile)

    # Initialize example bank for few-shot learning
    example_bank = AdaptiveFewShotManager(ExampleBank(exploration_rate=0.1))

    # Load existing examples if available
    example_bank.bank.load_from_file("example_bank.json")

    # Create the agent handler
    handler = AgentHandler(
        memory_store=memory_store,
        capability_profile=capability_profile,
        example_bank=example_bank,
        task_planner=task_planner,
        task_analyzer=task_analyzer,
        adaptive_formatter=adaptive_formatter,
        skill_registry=skill_registry,
    )

    # Initialize ACP agent for inter-agent communication
    acp_agent = ACPAgent(
        agent_id=f"agent_{os.getpid()}",
        agent_name=f"AgentZero-{os.getpid()}",
        host="0.0.0.0",
        port=8765,
        protocol="tcp",
    )
    set_acp_agent(acp_agent)

    return handler, acp_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AgentZero integrations")
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run the process in the background (POSIX only).",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
        help="Stop a running daemon process.",
    )
    args = parser.parse_args()

    if args.daemon and args.stop:
        parser.error("--daemon and --stop cannot be used together")

    if args.stop:
        sys.exit(stop_daemon())

    if args.daemon:
        daemonize()
        write_pid_file()

    # Initialize the agent
    handler, acp_agent = initialize_agent()

    # Run integrations/server tasks simultaneously
    async def run_bots():
        """Run configured integrations and servers concurrently."""
        tasks = []

        try:
            await handler.start_reminder_scheduler()
            logging.info("Starting reminder scheduler...")
        except Exception as e:
            logging.error(f"Failed to start reminder scheduler: {e}")

        if _env_bool("OPENAI_COMPAT_ENABLED", default=False):
            openai_api_key = os.environ.get("OPENAI_COMPAT_API_KEY", "").strip()
            if not openai_api_key:
                logging.error(
                    "OPENAI_COMPAT_ENABLED is set but OPENAI_COMPAT_API_KEY is missing. "
                    "OpenAI-compatible server will not start."
                )
            else:
                openai_host = (
                    os.environ.get("OPENAI_COMPAT_HOST", "0.0.0.0").strip() or "0.0.0.0"
                )
                openai_port = _env_port("OPENAI_COMPAT_PORT", 8001)
                openai_model_alias = (
                    os.environ.get("OPENAI_COMPAT_MODEL", "agentzero-main").strip()
                    or "agentzero-main"
                )
                openai_task = asyncio.create_task(
                    start_openai_compatible_server(
                        handler,
                        host=openai_host,
                        port=openai_port,
                        api_key=openai_api_key,
                        model_alias=openai_model_alias,
                    )
                )
                tasks.append(openai_task)
                logging.info(
                    "Starting OpenAI-compatible server on %s:%s (model alias: %s)...",
                    openai_host,
                    openai_port,
                    openai_model_alias,
                )

        # Start Telegram bot if token is available
        try:
            telegram_task = asyncio.create_task(run_telegram_bot_async(handler))
            tasks.append(telegram_task)
            logging.info("Starting Telegram bot...")
        except Exception as e:
            logging.error(f"Failed to start Telegram bot: {e}")

        # Start Sendblue bot if credentials are available
        try:
            sendblue_task = asyncio.create_task(start_sendblue_bot(handler))
            tasks.append(sendblue_task)
            logging.info("Starting Sendblue bot...")
        except Exception as e:
            logging.error(f"Failed to start Sendblue bot: {e}")

        if not tasks:
            logging.error("No bots could be started. Check your credentials.")
            return

        # Run all bots concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run_bots())
