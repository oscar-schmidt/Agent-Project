#!/usr/bin/env python3
"""
Test Classification Agent's ability to receive and reply to messages from other agents.

This test validates that the prompt changes successfully enable the Classification Agent
to respond to incoming messages using the ContactOtherAgents tool.

Prerequisites:
- Communication server must be running (communication_server/server.py)
- ClassificationAgent must be running (agents/classification_agent/classification_agent.py)

Usage:
    python tests/test_agent_reply.py
"""

import asyncio
import websockets
import json
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

async def test_classification_agent_reply():
    """
    Test that ClassificationAgent receives a message and sends a reply.

    Returns:
        bool: True if test passed, False otherwise
    """

    # Test configuration
    server_uri = "ws://localhost:8765"
    test_agent_id = "TestSender"
    target_agent_id = "ClassificationAgent"
    test_message = "Hello ClassificationAgent! Can you tell me about your capabilities and what you can help me with?"
    reply_timeout = 30.0

    # Test state tracking
    registration_success = False
    message_sent = False
    reply_received = False
    reply_content = None

    print("\n" + "="*70)
    print("🧪 CLASSIFICATION AGENT REPLY TEST")
    print("="*70)
    print(f"Target: {target_agent_id}")
    print(f"Server: {server_uri}")
    print(f"Timeout: {reply_timeout}s")
    print("="*70 + "\n")

    try:
        # Step 1: Connect to WebSocket server
        logging.info(f"📡 Connecting to server: {server_uri}")
        async with websockets.connect(server_uri) as websocket:
            logging.info("✅ Connected to server")

            # Step 2: Register as test agent
            logging.info(f"📝 Registering as: {test_agent_id}")
            registration_msg = {
                "message_type": "register",
                "agent_id": test_agent_id,
                "description": "Test agent for validating Classification Agent reply functionality",
                "capabilities": ["testing", "validation"]
            }

            await websocket.send(json.dumps(registration_msg))
            registration_response = await websocket.recv()
            reg_data = json.loads(registration_response)

            if reg_data.get("status") == "registration successful":
                registration_success = True
                logging.info(f"✅ Registration successful: {test_agent_id}")
            else:
                logging.error(f"❌ Registration failed: {reg_data}")
                return False

            # Step 3: Send test message to ClassificationAgent
            logging.info(f"📤 Sending message to {target_agent_id}")
            logging.info(f"   Message: \"{test_message[:50]}...\"")

            test_msg = {
                "message_type": "message",
                "sender_id": test_agent_id,
                "recipient_id": target_agent_id,
                "message": test_message
            }

            await websocket.send(json.dumps(test_msg))
            message_sent = True
            logging.info("✅ Message sent successfully")

            # Step 4: Wait for reply
            logging.info(f"⏳ Waiting for reply from {target_agent_id} (timeout: {reply_timeout}s)...")

            try:
                reply = await asyncio.wait_for(websocket.recv(), timeout=reply_timeout)
                reply_data = json.loads(reply)

                # Validate reply structure
                if reply_data.get("message_type") == "message":
                    if reply_data.get("sender_id") == target_agent_id:
                        reply_received = True
                        reply_content = reply_data.get("message", "")

                        logging.info("✅ REPLY RECEIVED!")
                        logging.info(f"   From: {reply_data.get('sender_id')}")
                        logging.info(f"   Content length: {len(reply_content)} characters")
                        logging.info(f"   First 200 chars: \"{reply_content[:200]}...\"")
                    else:
                        logging.error(f"❌ Reply from unexpected sender: {reply_data.get('sender_id')}")
                        return False
                else:
                    logging.error(f"❌ Unexpected message type: {reply_data.get('message_type')}")
                    return False

            except asyncio.TimeoutError:
                logging.error(f"❌ TIMEOUT: No reply received within {reply_timeout} seconds")
                logging.error(f"   ClassificationAgent may not be replying to messages")
                logging.error(f"   Check ClassificationAgent terminal for errors")
                return False

        # Test completed successfully
        print("\n" + "="*70)
        print("✅ TEST PASSED!")
        print("="*70)
        print(f"✅ Registration: {test_agent_id} registered successfully")
        print(f"✅ Message Sent: Test message delivered to {target_agent_id}")
        print(f"✅ Reply Received: {target_agent_id} responded correctly")
        print(f"✅ Reply Content: {len(reply_content)} characters")
        print("="*70)
        print("\n🎉 ClassificationAgent reply functionality is WORKING!")
        print("   The prompt changes successfully enable agent-to-agent communication.\n")
        return True

    except websockets.exceptions.ConnectionRefusedError:
        logging.error("❌ CONNECTION REFUSED")
        logging.error("   The communication server is not running")
        logging.error("   Start it with: uv run python communication_server/server.py")
        return False

    except Exception as e:
        logging.error(f"❌ UNEXPECTED ERROR: {e}")
        logging.error(f"   Type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    print("\n" + "🤖 Classification Agent Multi-Agent Communication Test")
    print("   Testing: Agent receives messages and sends replies\n")

    # Check prerequisites
    print("📋 Prerequisites:")
    print("   1. Communication server running on ws://localhost:8765")
    print("   2. ClassificationAgent running and registered")
    print("   3. Network connectivity to localhost\n")

    input("Press ENTER to start test (or Ctrl+C to cancel)...")

    # Run async test
    success = asyncio.run(test_classification_agent_reply())

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test cancelled by user\n")
        sys.exit(130)
