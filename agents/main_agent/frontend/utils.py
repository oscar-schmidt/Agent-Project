from __future__ import annotations
from nicegui import ui
from langchain_core.messages import HumanMessage, AIMessage

from backend.model.stores.LogStore import LogStore


def render_log(log_store: LogStore, container: ui.Element):
    if not log_store or not getattr(log_store, "logs", None):
        with container:
            ui.label("No logs yet.").classes('text-gray-500')
        return

    with container:
        for log in log_store.logs[-20:]:
            ui.label(f"• {log}").classes('text-sm text-gray-700')


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
