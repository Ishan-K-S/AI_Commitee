from pydantic import BaseModel, Field
from typing import Dict, List
import re


class Filter:
    class Valves(BaseModel):
        ENABLED: bool = Field(default=True, description="Enable automatic mistake-analysis nudging")

    def __init__(self):
        self.valves = self.Valves()

    # Detect whether the recent assistant message looks like the router's MCQ format.
    def _looks_like_router_mcq(self, messages: List[Dict]) -> bool:
        for m in reversed(messages[:-1]):
            if m.get("role") != "assistant":
                continue

            content = (m.get("content") or "").strip()
            if not content:
                continue

            if (
                "**question:**" in content.lower()
                and "a)" in content.lower()
                and "b)" in content.lower()
                and "c)" in content.lower()
                and "d)" in content.lower()
            ):
                return True

        return False

    # Detect very short answer attempts that the original pipe may not classify as "thinking".
    def _looks_like_answer_attempt(self, text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False

        lowered = text.lower()

        # Already explicit enough for the original classifier.
        if any(k in lowered for k in ["answer is", "i think", "my answer"]):
            return False

        # Bare multiple-choice answers: A / B / C / D, optionally with punctuation.
        if re.fullmatch(r"\(?\s*[abcd]\s*\)?[\.!?]?", lowered):
            return True

        # Slightly longer answer forms that still imply an answer selection.
        if re.fullmatch(r"(is it|maybe|probably)?\s*\(?\s*[abcd]\s*\)?[\.!?]?", lowered):
            return True

        return False

    # Rewrite the message so the unchanged pipe routes it into the existing "thinking" path.
    def _rewrite_for_mistake_analysis(self, text: str) -> str:
        cleaned = (text or "").strip()
        letter_match = re.search(r"([ABCD])", cleaned, re.IGNORECASE)

        if letter_match:
            answer = letter_match.group(1).upper()
            return (
                "My answer is "
                + answer
                + ". Please check whether it is correct and briefly explain the mistake if it is wrong."
            )

        return (
            "Please check my answer to the previous multiple-choice question and briefly explain "
            "the mistake if it is wrong.\n\n"
            + cleaned
        )

    # Minimal inlet-only filter so the original pipe code can remain unchanged.
    def inlet(self, body: dict) -> dict:
        if not self.valves.ENABLED:
            return body

        messages = body.get("messages", [])
        if not messages:
            return body

        last_message = messages[-1]
        if last_message.get("role") != "user":
            return body

        last_text = last_message.get("content") or ""

        if not self._looks_like_router_mcq(messages):
            return body

        if not self._looks_like_answer_attempt(last_text):
            return body

        # Replace only the last user message; everything else stays unchanged.
        messages[-1] = {
            **last_message,
            "content": self._rewrite_for_mistake_analysis(last_text),
        }
        body["messages"] = messages
        return body
