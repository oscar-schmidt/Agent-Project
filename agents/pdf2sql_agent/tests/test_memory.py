import asyncio
import sys
from pathlib import Path

# add parent dir to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.qdrant_store import QdrantStore
from src.memory.schemas import Episode, Semantic
from src.memory.memory_manager import ClaudeMemoryManager
from langchain_core.messages import HumanMessage, AIMessage


async def test_basic_memory():
    """test basic memory ops"""
    print("=== testing basic memory ===\n")

    store = QdrantStore(collection_name="ReviewAgent")

    # store an episode
    print("1. storing episode...")
    episode = Episode(
        observation="user asked to classify: 'app crashes when trying to checkout with apple pay'",
        thoughts="keywords 'crashes' and 'checkout' indicate critical severity",
        action="classify_review_criticality() -> Critical, analyze_sentiment() -> Frustrated",
        result="user confirmed, logged to notion. learned: payment + crash = critical"
    )
    ep_id = store.put(episode)
    print(f"   stored episode: {ep_id}\n")

    # store semantics
    print("2. storing semantic facts...")
    semantics = [
        Semantic(
            subject="crash keyword",
            predicate="indicates_criticality",
            object="Critical",
            context="payment flow"
        ),
        Semantic(
            subject="apple pay",
            predicate="has_recurring_issue",
            object="checkout crashes",
            context="ios app"
        ),
        Semantic(
            subject="refund request",
            predicate="correlates_with_sentiment",
            object="Angry"
        )
    ]

    for sem in semantics:
        sem_id = store.put(sem)
        print(f"   stored: {sem.subject} -> {sem.predicate} -> {sem.object}")
    print()

    # search
    print("3. searching for 'payment crashes'...")
    results = store.get("payment crashes during checkout", top_k=3)
    print(f"   found {len(results)} memories:\n")

    for i, mem in enumerate(results, 1):
        print(f"   {i}. {mem['memory_type']} (score: {mem['score']:.3f})")
        print(f"      {mem['text'][:80]}...\n")

    # count
    total = store.count()
    print(f"4. total memories: {total}\n")


async def test_memory_manager():
    """test claude memory extraction"""
    print("=== testing claude extraction ===\n")

    import os
    from dotenv import load_dotenv
    # load .env from parent directory
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("skipping extraction test - ANTHROPIC_API_KEY not in .env")
        print("(this is optional, storage works without it)")
        return

    try:
        manager = ClaudeMemoryManager()

        # simulate convo
        messages = [
            HumanMessage(content="classify this: 'app is slow and sometimes freezes'"),
            AIMessage(content="ill classify this review"),
            AIMessage(content="classification: major severity. keywords 'slow' and 'freezes' indicate performance issues"),
            HumanMessage(content="good but freezing should be critical not major"),
            AIMessage(content="youre right, updated. freezing = critical severity"),
        ]

        print("extracting memories...")
        extracted = manager.extract(messages)

        print(f"extracted {len(extracted)} memories:\n")
        for mem in extracted:
            if isinstance(mem, Episode):
                print(f"episode:")
                print(f"  obs: {mem.observation[:60]}...")
                print(f"  result: {mem.result[:60]}...\n")
            elif isinstance(mem, Semantic):
                print(f"semantic:")
                print(f"  {mem.subject} -> {mem.predicate} -> {mem.object}")
                if mem.context:
                    print(f"  context: {mem.context}")
                print()
    except Exception as e:
        print(f"extraction test failed: {e}")
        print("(this is optional, storage still works)")


async def main():
    try:
        await test_basic_memory()
        print("\n" + "="*50 + "\n")
        await test_memory_manager()
        print("\nsuccess! memory system working")
        print("memory stored in: ./memory_storage/")
    except Exception as e:
        print(f"error: {e}")
        print("\nmake sure you have:")
        print("1. pip install -r requirements.txt")
        print("2. MEMORY_ENABLED=true in .env")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
