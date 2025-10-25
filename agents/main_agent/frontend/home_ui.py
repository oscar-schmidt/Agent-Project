import asyncio
import os
from nicegui import ui
import tempfile
from agents.main_agent.backend.model.states.StateManager import StateManager
#from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from langchain_core.messages import HumanMessage, AIMessage
#from agents.main_agent.backend.depricated.graph.get_graph import get_graph
#from agents.main_agent.backend.utils import render_log


def render_chat_section():
    state = StateManager.get_state()

    with ui.column().classes('w-full h-full gap-2'):
        chat_area = ui.column().classes(
            'w-full h-[400px] overflow-y-auto  p-4 bg-gray-50 gap-2 flex-1'
        )

        for msg in state.messages.message_history:
            if isinstance(msg, HumanMessage):
                ui.chat_message(msg.content, name='User',
                                sent=True)
            elif isinstance(msg, AIMessage):
                ui.chat_message(msg.content, name='AI')

        with ui.row().classes('w-full gap-2 mt-2'):
            user_input = ui.input(
                'Type your message...',
                placeholder='Enter your message here...'
            ).props('autofocus').classes('flex-1')
            ui.button('Send', on_click=lambda: asyncio.create_task(
                on_chat_submit(user_input, chat_area)
            )).classes('mt-4')
            user_input.on('keydown.enter', lambda e: asyncio.create_task(
                on_chat_submit_wrapper(user_input, chat_area)
            ))


async def on_chat_submit_wrapper(user_input, chat_area):
    user_input.value = ''
    await on_chat_submit(user_input, chat_area)


async def on_chat_submit(user_input, chat_area):
    text = user_input.value.strip()
    if not text:
        return

    user_input.value = ''

    state = StateManager.get_state()
    state.messages.append(HumanMessage(content=text))
    StateManager.update_state(state)

    with chat_area:
        ui.chat_message(text, name='User', sent=True)
        thinking = ui.chat_message('AI is thinking...', name='AI')

    """compiled_graph = await get_graph(state)
    new_state = await compiled_graph.ainvoke(state, config={"thread_id": "123"})

    if isinstance(new_state, dict):
        new_state = GraphState(**new_state)

    StateManager.update_state(new_state)"""

    with chat_area:
        thinking.delete()
        ui.chat_message()
        #new_state.messages.message_history[-1].content, name='AI')

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
    #render_log(state.logs, log_area)

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

    ui.button('Reset Graph', on_click=reset_graph, color='red')\
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
