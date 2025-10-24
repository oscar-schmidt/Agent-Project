from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List, Union
import json

from .schemas import Episode, Semantic


class ClaudeMemoryManager:
    """extracts memories from convos using claude"""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: str = None):
        # load api key from config if not provided
        if api_key is None:
            from src.config import ANTHROPIC_API_KEY
            api_key = ANTHROPIC_API_KEY

        self.llm = ChatAnthropic(
            model=model,
            anthropic_api_key=api_key,
            temperature=0.0
        )

    def extract(self, messages: List) -> List[Union[Episode, Semantic]]:
        """extract episodes and semantics from conversation"""
        if not messages:
            return []

        try:
            system_prompt = """extract memories from this conversation.

extract Episodes (classification experiences):
- observation: what user requested and review content
- thoughts: why we chose this classification
- action: tools called with params
- result: outcome, user feedback, lessons learned

extract Semantics (knowledge facts):
- subject: main entity (eg "crash keyword", "payment issue")
- predicate: relationship (eg "indicates_criticality", "correlates_with")
- object: target value (eg "Critical", "Angry sentiment")
- context: when applicable (optional)

return as json:
{
  "episodes": [{"observation": "...", "thoughts": "...", "action": "...", "result": "..."}],
  "semantics": [{"subject": "...", "predicate": "...", "object": "...", "context": "..."}]
}

only extract meaningful stuff, skip generic chatter"""

            # format msgs
            convo = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    role = getattr(msg, 'type', 'unknown')
                    convo.append(f"{role}: {msg.content}")

            convo_text = "\n\n".join(convo)

            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"conversation:\n\n{convo_text}")
            ])

            return self._parse_response(response.content)

        except Exception as e:
            print(f"extraction error: {e}")
            return []

    def _parse_response(self, response: str) -> List[Union[Episode, Semantic]]:
        """parse claude response into memory objs"""
        memories = []

        try:
            # handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            data = json.loads(response.strip())

            # parse episodes
            for ep in data.get("episodes", []):
                try:
                    memories.append(Episode(**ep))
                except:
                    pass

            # parse semantics
            for sem in data.get("semantics", []):
                try:
                    memories.append(Semantic(**sem))
                except:
                    pass

            return memories

        except:
            return []
