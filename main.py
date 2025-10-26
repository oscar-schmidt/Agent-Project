import asyncio
from nicegui import ui
from Server.adaptor import Adaptor
from frontend.home_ui import render_chat_section, render_sidebar

adapter = Adaptor(
    agent_id="main_agent",
    description="Interactive client",
    capabilities=["qa", "summary"]
)


@ui.page('/')
def main_page():
    with ui.row().classes('w-full flex h-screen p-6 gap-6'):
        with ui.column().classes('flex-3 w-1/4 h-full overflow-y-auto border rounded-lg p-4 bg-gray-50 shadow-sm'):
            render_sidebar()

        with ui.column().classes('flex-9 w-3/4 h-full overflow-y-auto border rounded-lg p-4 bg-gray-50 shadow-sm'):
            render_chat_section(adapter)

    asyncio.get_event_loop().create_task(adapter.start())


ui.run(title='Graph AI Chat', reload=True, port=8080)
