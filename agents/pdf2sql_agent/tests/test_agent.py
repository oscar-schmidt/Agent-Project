#!/usr/bin/env python3
"""
Quick test script for the agent

Tests basic functionality without interactive mode.
"""

import sys
from src.agent.agent_graph import agent_app

def test_agent():
    """Test basic agent functionality"""

    print("🧪 Testing Agent Functionality\n")
    print("=" * 60)

    # Test configuration
    config = {
        "configurable": {
            "thread_id": "test_session"
        }
    }

    # Test 1: Simple help query
    print("\n📝 Test 1: Agent responds to help query")
    print("-" * 60)

    try:
        response = agent_app.invoke(
            {"messages": [{"role": "user", "content": "What can you help me with?"}]},
            config=config
        )

        if response and "messages" in response:
            last_message = response["messages"][-1]
            content = getattr(last_message, "content", "No response")
            print(f"✅ Agent responded: {content[:200]}...")
        else:
            print("❌ No response from agent")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("\nAgent is ready for use. Run: python run_agent.py")
    return True

if __name__ == "__main__":
    success = test_agent()
    sys.exit(0 if success else 1)
