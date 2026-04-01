"""Main entry point for the agent framework."""

import asyncio
import logging
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from memory import EnhancedMemoryStore  # noqa: E402
from capabilities import Capability, CapabilityProfile, AdaptiveFormatter  # noqa: E402
from examples import ExampleBank, AdaptiveFewShotManager  # noqa: E402
from planning import TaskPlanner, TaskAnalyzer  # noqa: E402
from tools import set_memory_store  # noqa: E402
from handler import AgentHandler  # noqa: E402
from integrations import run_telegram_bot_async, start_sendblue_bot  # noqa: E402

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - API key and model ID from environment variables with fallbacks
API_KEY = os.environ.get("NVIDIA_API_KEY", "")
MODEL_ID = os.environ.get("MODEL_ID", "moonshotai/kimi-k2-instruct-0905")


def initialize_agent() -> AgentHandler:
    """Initialize all agent components and return the handler."""
    # Initialize memory store
    memory_store = EnhancedMemoryStore(
        db_path="agent_memory.db",
        api_key=API_KEY,
    )

    # Set memory store for tools
    set_memory_store(memory_store)

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
    )

    return handler


if __name__ == "__main__":
    # Initialize the agent
    handler = initialize_agent()

    # Run both bots simultaneously
    async def run_bots():
        """Run both Telegram and Sendblue bots concurrently."""
        tasks = []

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
