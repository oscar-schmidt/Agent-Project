import sys
import argparse
import uuid
from typing import Optional
import asyncio

def cleanup_terminal():
    """Flush output buffers to ensure clean exit"""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass


def print_banner():
    """welcome banner"""
    print("\n" + "=" * 70)
    print("  REVIEW CLASSIFICATION AGENT")
    print("=" * 70 + "\n")


def format_message(message) -> str:
    """
    Format a message for display.
    """
    content = getattr(message, "content", "")

    #handle tool calls
    if hasattr(message, "tool_calls") and message.tool_calls:
        tool_info = []
        for tc in message.tool_calls:
            tool_name = tc.get("name", "unknown")
            tool_info.append(f"  -> Calling tool: {tool_name}")
        return "\n".join(tool_info)

    # Handle ToolMessage
    if hasattr(message, "name") and message.name:
        # not displaying raw tool messages they are handled by the agent
        return None

    #regular message content
    return content if content else None


def get_agent_response(messages: list) -> str:
    """
    Extract the final agent response from message history.
    """
    #find last AI message
    for message in reversed(messages):
        # Check if its an AI message 
        if hasattr(message, "content") and message.content:
            #skip if its a tool call without content
            if hasattr(message, "tool_calls") and message.tool_calls:
                continue
            # skip if its a tool message
            if hasattr(message, "name") and message.name:
                continue
            #this is the agents response
            return message.content

    return "(No response)"


async def run_interactive_session(thread_id, graph):
    """
    run an interactive session
    """
    # Generate or use provided thread ID
    if not thread_id:
        thread_id = f"session_{uuid.uuid4().hex[:8]}"

    print_banner()
    print(f"Conversation Thread: {thread_id}")
    print(f"(use this ID to resume: python run_agent.py --thread-id {thread_id})\n")

    # Configuration for this conversation
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            # Handle special commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye! Your conversation is saved.")
                print(f"Resume with: python run_agent.py --thread-id {thread_id}\n")
                cleanup_terminal()
                break

            if user_input.lower() == 'clear':
                thread_id = f"session_{uuid.uuid4().hex[:8]}"
                config["configurable"]["thread_id"] = thread_id
                print(f"\nStarted new conversation: {thread_id}\n")
                continue

            if not user_input:
                continue

            # Invoke the agent
            print("\nAgent: ", end="", flush=True)

            response = await graph.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config
            )

            # Extract and display the agent's final response
            if response and "messages" in response:
                agent_response = get_agent_response(response["messages"])
                print(agent_response)
            else:
                print("(No response)")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            print(f"Resume with: python run_agent.py --thread-id {thread_id}\n")
            cleanup_terminal()
            break

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.\n")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Interactive Review Classification Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agent.py                              # Start new conversation
  python run_agent.py --thread-id my_session       # Resume conversation
  python run_agent.py --enable-critique            # Enable self-review loop
  python run_agent.py --enable-memory              # Enable long-term memory
  python run_agent.py --enable-memory --enable-critique  # Both features

Commands:
  clear    Start new conversation
  exit     Exit the session
        """
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Thread ID to resume a previous conversation"
    )
    parser.add_argument(
        "--enable-critique",
        action="store_true",
        help="Enable the critique/self-review loop for better quality responses"
    )
    parser.add_argument(
        "--enable-memory",
        action="store_true",
        help="Enable long-term memory (learns from past interactions)"
    )

    args = parser.parse_args()

    # Check for required environment variables
    from agents.pdf2sql_agent.src.config import ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        print("\nError: ANTHROPIC_API_KEY not set in environment")
        print("Please add your Anthropic API key to the .env file:\n")
        print("ANTHROPIC_API_KEY=sk-ant-api03-your-key-here\n")
        sys.exit(1)

    # Build and compile the agent graph with memory checkpointing
    from agents.pdf2sql_agent.src.agent.agent_graph import create_agent_app
    agent_app = await create_agent_app(
        enable_critique=args.enable_critique,
        enable_memory=args.enable_memory
    )

    # Run interactive session
    try:
        await run_interactive_session(thread_id=args.thread_id, graph=agent_app)
    except Exception as e:
        print(f"\nFatal error: {str(e)}\n")
        cleanup_terminal()
        sys.exit(1)
    finally:
        # Close database connection if it exists
        if agent_app and hasattr(agent_app, 'checkpointer') and hasattr(agent_app.checkpointer, 'conn'):
            try:
                await agent_app.checkpointer.conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Ensure cleanup runs even if asyncio.run() fails
        cleanup_terminal()
