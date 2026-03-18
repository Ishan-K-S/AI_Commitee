from pydantic import BaseModel, Field
from typing import Any, Optional
import re


class Filter:
    class Valves(BaseModel):
        ENABLE_GUARDRAILS: bool = Field(
            default=True, description="Master switch for this filter"
        )
        MAX_HISTORY_MESSAGES: int = Field(
            default=8, description="Keep only the most recent messages"
        )
        MAX_USER_CHARS: int = Field(
            default=4000, description="Trim very long user inputs"
        )
        DROP_TOOL_MESSAGES: bool = Field(
            default=True,
            description="Remove tool messages before they reach the model/router",
        )
        BLOCK_SUSPICIOUS_PROMPTS: bool = Field(
            default=True,
            description="Block obvious prompt injection attempts",
        )
        NEUTRALIZE_ROLE_MARKERS: bool = Field(
            default=True,
            description="Rewrite lines that try to impersonate system/developer roles",
        )
        CHECK_RESPONSE_LEAKAGE: bool = Field(
            default=True,
            description="Detect system prompt leakage in model responses",
        )
        CHECK_RESPONSE_OFFTOPIC: bool = Field(
            default=True,
            description="Detect off-topic (non-math) responses",
        )
        CHECK_RESPONSE_HARMFUL: bool = Field(
            default=True,
            description="Detect harmful or inappropriate content in responses",
        )

    def __init__(self):
        self.valves = self.Valves()

        self._suspicious_patterns = [
            re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.I),
            re.compile(r"disregard\s+(all\s+)?previous\s+instructions?", re.I),
            re.compile(r"reveal\s+(the\s+)?system\s+prompt", re.I),
            re.compile(r"show\s+(me\s+)?(your\s+)?hidden\s+instructions?", re.I),
            re.compile(r"(you\s+are\s+now|act\s+as)\s+(the\s+)?system", re.I),
            re.compile(r"(you\s+are\s+now|act\s+as)\s+(the\s+)?developer", re.I),
            re.compile(r"override\s+(your\s+)?safety", re.I),
            re.compile(r"do\s+not\s+follow\s+your\s+rules", re.I),
            re.compile(r"pretend\s+to\s+be\s+an?\s+unfiltered", re.I),
            re.compile(r"jailbreak", re.I),
        ]

        self._role_prefix_re = re.compile(
            r"(?im)^\s*(system|developer|assistant|tool)\s*:\s*"
        )

        # Patterns that suggest the model is leaking its system prompt
        self._leakage_patterns = [
            re.compile(r"my\s+(system\s+)?instructions?\s+(are|say|state|tell me)", re.I),
            re.compile(r"i\s+(was|am|have been)\s+(told|instructed|prompted|configured)\s+to", re.I),
            re.compile(r"(as\s+per|according\s+to)\s+my\s+(system\s+)?prompt", re.I),
            re.compile(r"my\s+(system\s+)?prompt\s+(says|states|tells)", re.I),
            re.compile(r"i\s+am\s+not\s+supposed\s+to\s+(tell|share|reveal|show)", re.I),
            re.compile(r"(here\s+(are|is)|these\s+are)\s+my\s+(hidden\s+)?(instructions?|rules|guidelines)", re.I),
            re.compile(r"i\s+(cannot|can't)\s+reveal\s+my\s+(system\s+)?prompt", re.I),
        ]

        # Patterns that suggest harmful or inappropriate content
        self._harmful_patterns = [
            re.compile(r"\b(kill|murder|suicide|self.harm|hurt\s+(yourself|myself|himself|herself))\b", re.I),
            re.compile(r"\b(bomb|weapon|explosive|firearm|drug\s+synthesis)\b", re.I),
            re.compile(r"\b(racist|sexist|slur|hate\s+speech)\b", re.I),
            re.compile(r"(sexual|explicit|porn|nude|naked)\s+(content|material|image)", re.I),
            re.compile(r"\b(hack|exploit|malware|phishing|ransomware)\b", re.I),
        ]

        # Keywords that suggest the response has math or study related content
        self._math_keywords = re.compile(
            r"\b(math|algebra|geometry|arithmetic|fraction|equation|calculus|"
            r"problem|question|answer|solution|calculate|number|multiply|divide|"
            r"subtract|add|sum|product|factor|variable|formula|theorem|proof|"
            r"integer|decimal|percent|ratio|area|volume|angle|triangle|circle|"
            r"square|rectangle|graph|coordinate|function|derivative|integral|"
            r"matrix|vector|polynomial|exponent|logarithm|statistics|probability|"
            r"mean|median|mode|practice|study|exam|test|option|choice|correct|"
            r"incorrect|wrong|right|explain|understand|learn)\b",
            re.I,
        )

        # Fallback messages per violation type
        self._fallback_leakage = (
            "I'm sorry, I can't share information about my internal instructions. "
            "Would you like a new math practice problem, or do you have a math question I can help with?"
        )
        self._fallback_harmful = (
            "I'm sorry, I can't help with that. "
            "I'm here to help you study math! Would you like a practice problem or have a math question?"
        )
        self._fallback_offtopic = (
            "I'm here to help you with math! I don't think my response was quite on track. "
            "Would you like a new practice problem, or is there a math concept I can explain?"
        )

    def _content_to_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    if part.strip():
                        parts.append(part)
                    continue
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in {"text", "input_text", "output_text"}:
                    text = part.get("text", "")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)
                elif isinstance(part.get("text"), str) and part["text"].strip():
                    parts.append(part["text"])
                elif part_type in {"image_url", "input_image"}:
                    parts.append("[image]")
            return "\n".join(parts)
        if isinstance(content, dict):
            if isinstance(content.get("text"), str):
                return content["text"]
            nested = content.get("content")
            if nested is not None:
                return self._content_to_text(nested)
            return ""
        return str(content)

    def _normalize_role(self, role: Any) -> str:
        if not isinstance(role, str):
            return "user"
        role = role.strip().lower()
        return role if role in {"system", "user", "assistant", "tool"} else "user"

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _neutralize_role_markers(self, text: str) -> str:
        return self._role_prefix_re.sub("[quoted-role-text] ", text)

    def _is_suspicious(self, text: str) -> bool:
        return any(p.search(text) for p in self._suspicious_patterns)

    def _is_leaking(self, text: str) -> bool:
        return any(p.search(text) for p in self._leakage_patterns)

    def _is_harmful(self, text: str) -> bool:
        return any(p.search(text) for p in self._harmful_patterns)

    def _is_offtopic(self, text: str) -> bool:
        # Only flag as off-topic if the response is long enough to reasonably judge
        # Short responses like "Great job!" or "Good luck!" should not be flagged
        if len(text.split()) < 20:
            return False
        return not bool(self._math_keywords.search(text))

    def _sanitize_text(self, text: str) -> str:
        text = self._normalize_whitespace(text)
        if len(text) > self.valves.MAX_USER_CHARS:
            text = text[: self.valves.MAX_USER_CHARS].rstrip() + "\n\n[truncated]"
        if self.valves.NEUTRALIZE_ROLE_MARKERS:
            text = self._neutralize_role_markers(text)
        return text

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__=None,
    ) -> dict:
        if not self.valves.ENABLE_GUARDRAILS:
            return body

        messages = body.get("messages", [])
        if not isinstance(messages, list):
            return body

        recent_messages = messages[-self.valves.MAX_HISTORY_MESSAGES :]
        cleaned_messages = []
        latest_user_text = ""

        for msg in recent_messages:
            if not isinstance(msg, dict):
                continue

            role = self._normalize_role(msg.get("role", "user"))

            if self.valves.DROP_TOOL_MESSAGES and role == "tool":
                continue

            content = msg.get("content")
            text = self._content_to_text(content)

            if text:
                sanitized = self._sanitize_text(text)
                msg = dict(msg)
                msg["role"] = role
                msg["content"] = sanitized
                if role == "user":
                    latest_user_text = sanitized
            else:
                msg = dict(msg)
                msg["role"] = role

            cleaned_messages.append(msg)

        if latest_user_text and self._is_suspicious(latest_user_text):
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Blocked: possible prompt injection attempt detected.",
                            "done": True,
                            "hidden": False,
                        },
                    }
                )
            if self.valves.BLOCK_SUSPICIOUS_PROMPTS:
                raise ValueError(
                    "Request blocked by guardrail filter: prompt appears to contain instruction-override text."
                )

        body = dict(body)
        body["messages"] = cleaned_messages
        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__=None,
    ) -> dict:
        if not self.valves.ENABLE_GUARDRAILS:
            return body

        messages = body.get("messages", [])
        if not isinstance(messages, list):
            return body

        # Find the last assistant message to check
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], dict) and messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            return body

        response_text = self._content_to_text(messages[last_assistant_idx].get("content", ""))

        if not response_text.strip():
            return body

        fallback = None

        # Check in order of severity: harmful first, then leakage, then off-topic
        if self.valves.CHECK_RESPONSE_HARMFUL and self._is_harmful(response_text):
            fallback = self._fallback_harmful
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Response filtered: harmful content detected.",
                            "done": True,
                            "hidden": False,
                        },
                    }
                )

        elif self.valves.CHECK_RESPONSE_LEAKAGE and self._is_leaking(response_text):
            fallback = self._fallback_leakage
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Response filtered: possible system prompt leakage detected.",
                            "done": True,
                            "hidden": False,
                        },
                    }
                )

        elif self.valves.CHECK_RESPONSE_OFFTOPIC and self._is_offtopic(response_text):
            fallback = self._fallback_offtopic
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Response filtered: off-topic content detected.",
                            "done": True,
                            "hidden": False,
                        },
                    }
                )

        if fallback:
            body = dict(body)
            messages = list(messages)
            messages[last_assistant_idx] = {
                **messages[last_assistant_idx],
                "content": fallback,
            }
            body["messages"] = messages

        return body