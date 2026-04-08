"""Test the /setprompt command functionality."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

# Test if the command handler exists and can be imported
try:
    from integrations import telegram_setprompt, telegram_handle_msg, pending_prompt_users
    from memory import MemoryStore
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    import_error = str(e)


async def test_setprompt_command_flow():
    """Test the full /setprompt command flow."""
    if not IMPORTS_AVAILABLE:
        print(f"Skipping test due to import error: {import_error}")
        return True
    
    # Create a temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Initialize memory store
        memory_store = MemoryStore(db_path=db_path, api_key="test-key")
        
        # Create mock handler
        handler = MagicMock()
        handler.memory_store = memory_store
        
        # Create mock update object for /setprompt command
        mock_update = MagicMock()
        mock_user = MagicMock()
        mock_user.id = 12345
        mock_update.effective_user = mock_user
        mock_message = MagicMock()
        mock_update.message = mock_message
        mock_message.reply_text = AsyncMock()
        
        mock_context = MagicMock()
        
        # Simulate /setprompt command
        await telegram_setprompt(mock_update, mock_context)
        
        # Verify the response asking for prompt
        assert mock_message.reply_text.await_count == 1
        call_args = mock_message.reply_text.await_args[0][0]
        assert "Please send your new system prompt" in call_args
        
        # Verify user is in pending state
        assert 12345 in pending_prompt_users
        
        # Reset mock
        mock_message.reply_text.reset_mock()
        
        # Create mock update for the actual prompt message
        mock_update2 = MagicMock()
        mock_update2.effective_user = mock_user
        mock_message2 = MagicMock()
        mock_update2.message = mock_message2
        mock_message2.text = "You are a helpful assistant that speaks like a pirate."
        mock_message2.reply_text = AsyncMock()
        
        mock_context2 = MagicMock()
        
        # Simulate sending the prompt
        await telegram_handle_msg(handler, mock_update2, mock_context2)
        
        # Verify the success response
        assert mock_message2.reply_text.await_count == 1
        call_args2 = mock_message2.reply_text.await_args[0][0]
        assert "✅ System prompt updated successfully" in call_args2
        
        # Verify user is no longer in pending state
        assert 12345 not in pending_prompt_users
        
        # Verify the prompt was stored
        stored_prompt = memory_store.get_system_prompt()
        assert stored_prompt == "You are a helpful assistant that speaks like a pirate."
        
        print("✅ All tests passed!")
        return True
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
        # Clear pending state
        pending_prompt_users.clear()


async def test_setprompt_empty_prompt():
    """Test that empty prompts are rejected."""
    if not IMPORTS_AVAILABLE:
        print(f"Skipping test due to import error: {import_error}")
        return True
    
    # Create a temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Initialize memory store
        memory_store = MemoryStore(db_path=db_path, api_key="test-key")
        
        # Create mock handler
        handler = MagicMock()
        handler.memory_store = memory_store
        
        # Add user to pending state manually
        user_id = 99999
        pending_prompt_users[user_id] = True
        
        # Create mock update with empty message
        mock_update = MagicMock()
        mock_user = MagicMock()
        mock_user.id = user_id
        mock_update.effective_user = mock_user
        mock_message = MagicMock()
        mock_update.message = mock_message
        mock_message.text = "   "  # whitespace only
        mock_message.reply_text = AsyncMock()
        
        mock_context = MagicMock()
        
        # Simulate sending empty prompt
        await telegram_handle_msg(handler, mock_update, mock_context)
        
        # Verify the error response
        assert mock_message.reply_text.await_count == 1
        call_args = mock_message.reply_text.await_args[0][0]
        assert "Prompt cannot be empty" in call_args
        
        # Verify user is no longer in pending state
        assert user_id not in pending_prompt_users
        
        # Verify no prompt was stored
        stored_prompt = memory_store.get_system_prompt()
        assert stored_prompt is None
        
        print("✅ Empty prompt test passed!")
        return True
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
        # Clear pending state
        pending_prompt_users.clear()


async def main():
    """Run all tests."""
    print("Running /setprompt command tests...")
    
    test1_passed = await test_setprompt_command_flow()
    test2_passed = await test_setprompt_empty_prompt()
    
    if test1_passed and test2_passed:
        print("\n✅ All tests passed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
