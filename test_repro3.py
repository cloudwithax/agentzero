import asyncio
from main import initialize_agent

async def main():
    handler = initialize_agent()
    # Set a custom prompt that does not mention tools!
    handler.memory_store.set_system_prompt("You are a helpful assistant. You talk casually.")
    
    request = {
        "messages": [
            {"role": "user", "content": "I live in miami and the weather is 68. Please store this memory."}
        ]
    }
    response = await handler.handle(request, session_id="test")
    print("Response:", response)
    
    stats = handler.memory_store.get_memory_stats()
    print("Memory stats:", stats)
    
if __name__ == "__main__":
    asyncio.run(main())
