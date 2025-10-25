from typing import Any
from langgraph.checkpoint.serde import jsonplus
from langgraph.checkpoint.serde.jsonplus import _msgpack_default
from langgraph.checkpoint.serde.jsonplus import _option
from langgraph.checkpoint.serde.jsonplus import ormsgpack
import asyncio
import os
from nicegui import ui, app
import tempfile
from agents.main_agent.backend.model.states.StateManager import StateManager
from langchain_core.messages import HumanMessage, AIMessage
from agents.main_agent.frontend.main_agent import AgentManager
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
# Reference from: https://github.com/langchain-ai/langgraph/issues/4956#issuecomment-3135374853
def message_to_dict(msg):
    """
    Recursively convert a message or object into a dict/str (safe for serialization).
    """
    if hasattr(msg, "to_dict"):
        return msg.to_dict()
    elif isinstance(msg, dict):
        return {k: message_to_dict(v) for k, v in msg.items()}
    elif isinstance(msg, (list, tuple)):
        return [message_to_dict(x) for x in msg]
    elif isinstance(msg, (str, int, float, bool, type(None))):
        return msg
    else:
        print("Serialization Fallback, type:", type(msg))
        print(msg)
        return {"role": getattr(msg, "role", "user"), "content": str(getattr(msg, "content", msg))}


def _msgpack_enc(data: Any) -> bytes:
    return ormsgpack.packb(message_to_dict(data), default=_msgpack_default, option=_option)


def monkey_patch():
    setattr(jsonplus, "_msgpack_enc", _msgpack_enc)





agent = AgentManager()
@ui.page('/')
def start():

    async def on_chat_submit():
        text = user_input.value.strip()
        if not text:
            return
        logging.info(f"User input: {text}")
        user_input.value = ''
        agent.chat_manager.messages.append({'role': 'user', 'content': text})
        update_chat_display()
        with chat_area:
            thinking = ui.chat_message('AI is thinking...', name='AI')
        await agent.task_queue.put(text)

        with chat_area:
            thinking.delete()
            ui.chat_message()

        user_input.value = ""




    def render_file_uploader():

        async def on_upload(e):
            state = StateManager.get_state()
            uploaded_file = e.file
            file_extention = os.path.splitext(uploaded_file.name)[1].lower()

            if file_extention not in ['.pdf', '.xlsx']:
                raise ValueError(f"Unsupported file type: {file_extention}")

            file_bytes = await uploaded_file.read()

            with tempfile.NamedTemporaryFile(suffix=file_extention, delete=False) as temp:
                temp.write(file_bytes)
                temp.flush()
                state.qa_state.doc_path = temp.name
                state.qa_state.doc_name = uploaded_file.name
                state.qa_state.is_upload = True
                StateManager.update_state(state)

        ui.upload(
            on_upload=on_upload,
            multiple=False,
            label='Select PDF',
            auto_upload=True,
            max_file_size=20_000_000
        ).classes('mr-4')

    def render_sidebar():
        state = StateManager.get_state()
        render_file_uploader()
        log_area = ui.column().classes(
            'h-[300px] w-full overflow-y-auto p-3 bg-gray-50'
        )
        # render_log(state.logs, log_area)

        # with ui.tabs().classes('w-full mb-4'):
        #     log_tab = ui.tab('Logs')
        #     config_tab = ui.tab('Configs')

        # with ui.tab_panels().classes('p-4 bg-white border rounded-lg shadow-sm'):
        # with ui.tab_panel(log_tab):
        #     ui.label('Logs').classes('text-lg font-semibold mb-2')

        # with ui.tab_panel(config_tab):
        #     render_config_panel(state)

    def render_config_panel(state):
        ui.label('⚙️ Graph Config').classes('text-lg font-semibold mb-2')

        def reset_graph():
            new_state = StateManager.get_state()
            new_state.logs.append("Graph has been reset.")
            StateManager.update_state(new_state)
            ui.notify('Graph Reset Successfully!')

        ui.button('Reset Graph', on_click=reset_graph, color='red') \
            .classes('mr-2')

        with ui.column().classes('gap-2'):
            ui.number('CHUNK_SIZE', value=state.graph_config.CHUNK_SIZE,
                      min=200, max=1000)
            ui.number('CHUNK_OVERLAP', value=state.graph_config.CHUNK_OVERLAP,
                      min=50, max=150)
            ui.number('RAG_THRESHOLD', value=state.graph_config.RAG_THRESHOLD,
                      min=0.0, max=1.0)
            ui.number('TOP_K', value=state.graph_config.TOP_K, min=1, max=50)
            ui.checkbox('SHOW_LOG', value=True)


    with ui.row().classes('w-full flex h-screen p-6 gap-6'):

        with ui.column().classes(' flex-3 w-1/4 h-full overflow-y-auto border rounded-lg p-4 bg-gray-50 shadow-sm'):
            render_sidebar()

        with ui.column().classes('flex-9  w-3/4 h-full overflow-y-auto border rounded-lg p-4 bg-gray-50 shadow-sm'):
            with ui.column().classes('w-full h-full gap-2'):
                chat_area = ui.column().classes(
                    'w-full h-[400px] overflow-y-auto  p-4 bg-gray-50 gap-2 flex-1'
                )

                with ui.row().classes('w-full gap-2 mt-2'):
                    user_input = ui.input(
                        'Type your message...',
                        placeholder='Enter your message here...'
                    ).props('autofocus').classes('flex-1').on('keydown.enter', on_chat_submit)
                ui.button('Send', on_click=on_chat_submit).classes('mt-4')

        def update_chat_display():
            chat_area.clear()
            with chat_area:
                for msg in agent.chat_manager.messages:
                    if msg["role"] == "user":
                        with ui.row().classes('w-full justify-start'):
                            ui.chat_message(msg["content"], name="You", sent=True).classes(
                                'rounded-[20px] px-[15px] py-[10px] m-[5px] max-w-[70%]'
                            )
                    elif msg["role"] == "agent":
                        with ui.row().classes('w-full justify-end'):
                            ui.chat_message(msg["content"], name="Agent", sent=False).classes(
                                'rounded-[20px] px-[15px] py-[10px] m-[5px] max-w-[70%]'
                            )
        agent.update_ui_callback = update_chat_display
app.on_startup(agent.startup)

if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title='Graph AI Chat', reload=True, port=8080)