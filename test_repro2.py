import asyncio
from main import initialize_agent

async def main():
    handler = initialize_agent()
    request = {
        "messages": [
            {"role": "user", "content": "My name is John and I like pizza. Please store this memory."}
        ]
    }
    response = await handler.handle(request, session_id="test")
    print("Response:", response)
    stats = handler.memory_store.get_memory_stats()
    print("Memory stats:", stats)
    
if __name__ == "__main__":
    asyncio.run(main())
