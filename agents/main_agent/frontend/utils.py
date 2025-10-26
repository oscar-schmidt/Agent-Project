from __future__ import annotations
from nicegui import ui
from langchain_core.messages import HumanMessage, AIMessage


def render_message(messages, container: ui.Element):
    if not messages or not getattr(messages, "message_history", None):
        return

    with container:
        for message in messages.message_history:
            if isinstance(message, HumanMessage):
                ui.chat_message(message.content, name='User',
                                sent=True).classes('w-full')
            elif isinstance(message, AIMessage):
                ui.chat_message(message.content, name='AI').classes('w-full')
