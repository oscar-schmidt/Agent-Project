import asyncio
import os
from nicegui import ui
import tempfile
from agents.main_agent.backend.dataBase_setup.chroma_setup import get_all_collection_name
from agents.main_agent.backend.model.states.StateManager import StateManager
from agents.main_agent.backend.model.states.graph_state.GraphState import GraphState
from langchain_core.messages import HumanMessage, AIMessage
from agents.main_agent.backend.graph.get_graph import get_graph
from agents.main_agent.backend.nodes.excel_node.process_excel_node import process_excel_node
from agents.main_agent.backend.nodes.pdf_node.process_pdf_node import process_pdf_node

os.environ["TOKENIZERS_PARALLELISM"] = "false"
state_initialized = False


@ui.refreshable
def render_chat_section():
    if not state_initialized:
        ui.label("Loading chat state...")

        async def background_init():
            global state_initialized
            await initialize_state()
            state_initialized = True
            render_chat_section.refresh()

        asyncio.create_task(background_init())
        return

    state = StateManager.get_state()

    with ui.column().classes('w-full h-full gap-2'):
        with ui.scroll_area().classes('w-full h-full p-4 bg-gray-50') as chat_scroll:
            chat_content = ui.column().classes('w-full gap-2')

            with chat_content:
                for msg in state.messages:
                    if isinstance(msg, HumanMessage):
                        ui.chat_message(msg.content, name='User', sent=True)
                    elif isinstance(msg, AIMessage):
                        ui.chat_message(msg.content, name='AI')

        chat_scroll.scroll_to(percent=1.0)

        with ui.row().classes('w-full gap-2 mt-2'):
            user_input = ui.input(
                'Type your message...',
                placeholder='Enter your message here...'
            ).props('autofocus').classes('flex-1')

            ui.button('Send', on_click=lambda e: asyncio.create_task(
                on_chat_submit_wrapper(
                    user_input, chat_scroll, chat_content)
            )).classes('mt-4')

            user_input.on('keydown.enter', lambda e: asyncio.create_task(
                on_chat_submit_wrapper(
                    user_input, chat_scroll, chat_content)
            ))


async def on_chat_submit_wrapper(user_input, chat_scroll, chat_content):
    text = user_input.value.strip()
    user_input.value = ""
    await on_chat_submit(text, chat_scroll, chat_content)


async def on_chat_submit(text, chat_scroll, chat_content):
    if not text:
        return

    state = StateManager.get_state()
    state.messages.append(HumanMessage(content=text))
    StateManager.update_state(state)

    with chat_content:
        ui.chat_message(text, name='User', sent=True)

    chat_scroll.scroll_to(percent=1.0)

    with chat_content:
        thinking_msg = ui.chat_message('AI is thinking...', name='AI')

    chat_scroll.scroll_to(percent=1.0)

    compiled_graph = await get_graph()

    config = {
        "configurable": {
            "thread_id": "123"
        }
    }

    new_state = await compiled_graph.ainvoke(state, config=config)
    if isinstance(new_state, dict):
        new_state = GraphState(**new_state)

    try:
        thinking_msg.delete()
    except RuntimeError:
        pass

    with chat_content:
        ai_msg = ui.chat_message(new_state.messages[-1].content, name='AI')

    chat_scroll.scroll_to(percent=1.0)

    StateManager.update_state(new_state)


def render_file_uploader():

    ui.upload(
        on_upload=on_upload,
        multiple=False,
        label='Select File',
        auto_upload=True,
        max_file_size=20_000_000
    ).classes('mr-4')


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

    async def process_in_background():
        try:
            if file_extention == ".pdf":
                await process_pdf_node(state)
            else:
                await process_excel_node(state)

            render_collection_list.refresh()

        except Exception as e:
            ui.notify(
                f"Error processing {uploaded_file.name}: {e}", color="red")

    asyncio.get_event_loop().create_task(process_in_background())
    return


def render_sidebar():
    state = StateManager.get_state()
    render_file_uploader()
    ui.label("Knowledge Base:")
    render_collection_list()


@ui.refreshable
def render_collection_list():
    state = StateManager.get_state()
    collection_names = get_all_collection_name()
    with ui.column().classes('h-[300px] w-full overflow-y-auto p-3 bg-gray-50'):
        if collection_names:
            for collection_name in collection_names:
                ui.markdown(collection_name)
        else:
            ui.label("No collections yet")
    StateManager.update_state(state)


async def initialize_state():
    compiled_graph = await get_graph()
    config = {"configurable": {"thread_id": "123"}}

    try:
        snapshot = await compiled_graph.aget_state(config)
        if snapshot and snapshot.values:
            restored_state = GraphState(**snapshot.values)
            restored_state.qa_state = type(restored_state.qa_state)()
            restored_state.summary_state = type(restored_state.summary_state)()
            StateManager.update_state(restored_state)
            return
    except Exception as e:
        print(f"No previous state found, starting fresh: {e}")

    StateManager.update_state(GraphState())
