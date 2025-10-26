#!/usr/bin/env python3
"""
Test agent interaction with both greetings and tool calls
"""

from src.agent.agent_graph import agent_app

def test_greeting():
    """Test greeting response"""
    import uuid
    print("Test 1: Greeting")
    print("-" * 60)

    config = {"configurable": {"thread_id": f"test_greeting_{uuid.uuid4().hex[:8]}"}}

    response = agent_app.invoke(
        {"messages": [{"role": "user", "content": "hey"}]},
        config=config
    )

    # Debug: Print all messages
    print(f"Total messages: {len(response['messages'])}")
    for i, msg in enumerate(response["messages"]):
        print(f"Message {i}: type={type(msg).__name__}, has_content={hasattr(msg, 'content')}, content_len={len(getattr(msg, 'content', ''))}")

    # Find the AI response
    for msg in reversed(response["messages"]):
        if hasattr(msg, "content"):
            content = msg.content
            if content and len(content) > 0:
                if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                    if not hasattr(msg, "name") or not msg.name:  # Not a tool message
                        print(f"[PASS] Agent responded: {content[:100]}...")
                        return True

    print("[FAIL] No response with content found")
    return False

def test_tool_call():
    """Test classification tool"""
    import uuid
    print("\nTest 2: Classification with tool call")
    print("-" * 60)

    config = {"configurable": {"thread_id": f"test_classify_{uuid.uuid4().hex[:8]}"}}

    response = agent_app.invoke(
        {"messages": [{"role": "user", "content": "Classify 2 reviews"}]},
        config=config
    )

    # Debug: Print all messages
    print(f"Total messages: {len(response['messages'])}")
    for i, msg in enumerate(response["messages"]):
        msg_type = type(msg).__name__
        has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls
        content_len = len(getattr(msg, 'content', ''))
        print(f"Message {i}: type={msg_type}, has_tool_calls={has_tool_calls}, content_len={content_len}")

    # Find the final AI response
    for msg in reversed(response["messages"]):
        if hasattr(msg, "content"):
            content = msg.content
            if content and len(content) > 0:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    continue  # Skip tool call messages
                if hasattr(msg, "name") and msg.name:
                    continue  # Skip tool response messages
                print(f"[PASS] Agent responded: {content[:200]}...")
                return True

    print("[FAIL] No response with content found")
    return False

if __name__ == "__main__":
    test1 = test_greeting()
    test2 = test_tool_call()

    print("\n" + "=" * 60)
    if test1 and test2:
        print("[PASS] All tests passed!")
    else:
        print("[FAIL] Some tests failed")
