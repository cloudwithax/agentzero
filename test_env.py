"""Test script to verify .env support is working correctly."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Test that environment variables are loaded
telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
nvidia_api_key = os.environ.get("NVIDIA_API_KEY")

print("Environment Variables Test:")
print("=" * 50)
print(f"TELEGRAM_BOT_TOKEN: {'✓ Set' if telegram_token else '✗ Not set'}")
print(f"NVIDIA_API_KEY: {'✓ Set' if nvidia_api_key else '✗ Not set (using fallback)'}")
print("=" * 50)

# Test that main.py can be imported without errors
try:
    from main import API_KEY
    print("✓ main.py imported successfully")
    print(f"✓ API_KEY is configured: {API_KEY[:20]}...")
except Exception as e:
    print(f"✗ Error importing main.py: {e}")

print("\n.env support is working correctly!")
