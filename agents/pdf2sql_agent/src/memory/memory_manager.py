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
            system_prompt = """CRITICAL: Only extract memories from ACTUAL review data mentioned in this conversation. DO NOT infer, generalize, or create patterns that weren't explicitly present.

extract Episodes (classification experiences) - ONLY if actual review IDs/content were discussed:
- observation: EXACT review ID and content from the conversation (e.g., "REV-0556: poster design feedback")
- thoughts: specific reasoning used for THIS review
- action: actual tools called with actual parameters
- result: actual outcome and user feedback (if any)

extract Semantics (knowledge facts) - ONLY from ACTUAL reviews processed:
- subject: specific element from the ACTUAL review (e.g., "REV-0557 carbonation issue")
- predicate: factual relationship observed
- object: actual classification/sentiment from the review
- context: the specific review ID or product

RULES:
1. NEVER create generic patterns like "crash keyword -> Critical" unless you saw an ACTUAL review with that pattern
2. NEVER infer rules from hypothetical scenarios
3. If user just asked questions without processing reviews, return empty lists
4. Include review IDs (REV-XXXX) in all memories for traceability
5. Skip generic conversations, greetings, or clarifications

return as json:
{
  "episodes": [{"observation": "...", "thoughts": "...", "action": "...", "result": "..."}],
  "semantics": [{"subject": "...", "predicate": "...", "object": "...", "context": "..."}]
}

When in doubt, extract NOTHING rather than hallucinate patterns."""

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
