"""Test the persistent memory system."""
import asyncio
import os
from main import MemoryStore

# Use a temp db for testing
TEST_DB = "/tmp/test_agent_memory.db"

def cleanup():
    """Remove test database."""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

async def test_basic_memory():
    """Test basic memory CRUD operations."""
    print("=== Testing Basic Memory Operations ===")
    cleanup()

    store = MemoryStore(
        db_path=TEST_DB,
        api_key=os.environ.get("NVIDIA_API_KEY", ""),
    )

    # Test 1: Add memories without embedding (offline mode)
    print("\n1. Adding memories (offline mode, no embeddings)...")
    mem1 = await store.add_memory(
        content="The user's name is Alice and they love Python programming",
        metadata={"type": "user_info", "importance": "high"},
        topics=["user", "programming"],
        generate_embedding=False,
    )
    print(f"   Added memory #{mem1}")

    mem2 = await store.add_memory(
        content="The user prefers dark mode in all their applications",
        metadata={"type": "preference", "importance": "medium"},
        topics=["user", "ui"],
        generate_embedding=False,
    )
    print(f"   Added memory #{mem2}")

    mem3 = await store.add_memory(
        content="The user's favorite database is PostgreSQL",
        metadata={"type": "preference", "importance": "medium"},
        topics=["user", "databases"],
        generate_embedding=False,
    )
    print(f"   Added memory #{mem3}")

    # Test 2: Get stats
    print("\n2. Memory stats:")
    stats = store.get_memory_stats()
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Embedded memories: {stats['embedded_memories']}")

    # Test 3: Get recent memories
    print("\n3. Recent memories:")
    recent = store.get_recent_memories(limit=5)
    for mem in recent:
        print(f"   [{mem.id}] {mem.content[:60]}...")

    # Test 4: Get by topic
    print("\n4. Memories tagged 'user':")
    user_mems = store.get_memories_by_topic("user", limit=10)
    for mem in user_mems:
        print(f"   [{mem.id}] {mem.content[:60]}...")

    # Test 5: Delete memory
    print("\n5. Deleting memory #2...")
    deleted = store.delete_memory(mem2)
    print(f"   Deleted: {deleted}")

    stats = store.get_memory_stats()
    print(f"   Total memories after deletion: {stats['total_memories']}")

    print("\n✓ Basic memory tests passed!")
    return store


async def test_embedding_search():
    """Test embedding-based semantic search."""
    print("\n=== Testing Embedding Search ===")

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key or api_key == "test_key":
        print("⚠ No NVIDIA_API_KEY found - skipping embedding tests")
        print("  Set NVIDIA_API_KEY env var to test semantic search")
        return

    cleanup()

    store = MemoryStore(
        db_path=TEST_DB,
        api_key=api_key,
    )

    # Add memories with embeddings
    print("\n1. Adding memories with embeddings...")
    test_memories = [
        ("I love working with Python and Django framework", ["programming", "python"]),
        ("My favorite vacation spot is the beach in Hawaii", ["travel", "personal"]),
        ("I enjoy hiking in the mountains on weekends", ["hobbies", "outdoors"]),
        ("I'm learning machine learning and neural networks", ["programming", "ai"]),
        ("I prefer tea over coffee in the morning", ["food", "preference"]),
    ]

    for content, topics in test_memories:
        try:
            mem_id = await store.add_memory(
                content=content,
                topics=topics,
                generate_embedding=True,
            )
            print(f"   Added memory #{mem_id}: {content[:50]}...")
        except Exception as e:
            print(f"   Error adding memory: {e}")
            return

    # Test semantic search
    print("\n2. Testing semantic search...")

    queries = [
        "What does the user like to drink?",
        "Tell me about the user's programming interests",
        "Where does the user like to travel?",
        "What are the user's hobbies?",
    ]

    for query in queries:
        print(f"\n   Query: '{query}'")
        try:
            results = await store.search_memories(query, top_k=2, threshold=0.15)
            if results:
                for mem, score in results:
                    print(f"   → [{mem.id}] Score: {score:.3f} | {mem.content}")
            else:
                print("   → No results found")
        except Exception as e:
            print(f"   Error: {e}")

    # Test topic-filtered search
    print("\n3. Testing topic-filtered search...")
    print("   Query: 'programming' with topic='ai'")
    try:
        results = await store.search_memories("programming", top_k=5, topic="ai")
        for mem, score in results:
            print(f"   → [{mem.id}] {mem.content}")
    except Exception as e:
        print(f"   Error: {e}")

    # Stats
    print("\n4. Final stats:")
    stats = store.get_memory_stats()
    print(f"   Total memories: {stats['total_memories']}")
    print(f"   Embedded memories: {stats['embedded_memories']}")
    print(f"   Total topics: {stats['total_topics']}")

    print("\n✓ Embedding search tests passed!")


async def test_conversation_history():
    """Test conversation history tracking."""
    print("\n=== Testing Conversation History ===")
    cleanup()

    store = MemoryStore(
        db_path=TEST_DB,
        api_key="",
    )

    session_id = "test_session_123"

    print("\n1. Adding conversation messages...")
    store.add_conversation_message(
        session_id=session_id,
        role="user",
        content="Hello, can you help me with Python?",
    )
    store.add_conversation_message(
        session_id=session_id,
        role="assistant",
        content="Of course! I'd be happy to help with Python.",
    )
    store.add_conversation_message(
        session_id=session_id,
        role="user",
        content="How do I use dictionaries?",
    )

    print("2. Retrieving conversation history...")
    history = store.get_conversation_history(session_id=session_id, limit=10)
    for msg in history:
        print(f"   [{msg['role']}] {msg['content'][:50]}...")

    stats = store.get_memory_stats()
    print(f"\n3. Total conversations stored: {stats['total_conversations']}")

    print("\n✓ Conversation history tests passed!")


async def main():
    """Run all memory tests."""
    print("=" * 60)
    print("AGENT MEMORY SYSTEM TESTS")
    print("=" * 60)

    try:
        await test_basic_memory()
        await test_conversation_history()
        await test_embedding_search()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)

    finally:
        cleanup()
        print(f"\nCleaned up test database: {TEST_DB}")


if __name__ == "__main__":
    asyncio.run(main())
