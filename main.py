import asyncio
import json
import websockets
from Server.adaptor import AgentWebSocketAdapter
from backend.model.states.StateManager import StateManager
from frontend.home_ui import render_chat_section, render_sidebar
from nicegui import ui
from langgraph.checkpoint.serde.jsonplus import _option
from langgraph.checkpoint.serde.jsonplus import ormsgpack


adapter: AgentWebSocketAdapter = None


async def start():
    global adapter
    adapter = AgentWebSocketAdapter(
        agent_id="main_agent",
        description="Interactive client",
        capabilities=["qa", "summary"]
    )
    await adapter.start()

    state = StateManager.get_state()

    with ui.row().classes('w-full flex h-screen p-6 gap-6'):

        with ui.column().classes(' flex-3 w-1/4 h-full overflow-y-auto border rounded-lg p-4 bg-gray-50 shadow-sm'):
            render_sidebar()

        with ui.column().classes('flex-9  w-3/4 h-full overflow-y-auto border rounded-lg p-4 bg-gray-50 shadow-sm'):
            render_chat_section()


asyncio.get_event_loop().create_task(start())

ui.run(title='Graph AI Chat', reload=True, port=8080)
