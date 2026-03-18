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

    def __init__(self):
        self.valves = self.Valves()

        # Keep the first version intentionally simple and deterministic.
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

    def _content_to_text(self, content: Any) -> str:
        """
        Convert OpenWebUI-style content into plain text for scanning.
        Supports strings, dicts, and list-based multimodal message content.
        """
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
        # Keep paragraphs, but collapse excessive empty lines / spaces.
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _neutralize_role_markers(self, text: str) -> str:
        """
        Turns lines like 'SYSTEM: do X' into '[quoted system-like text] do X'
        so the model sees them as user-provided text, not role instructions.
        """
        return self._role_prefix_re.sub("[quoted-role-text] ", text)

    def _is_suspicious(self, text: str) -> bool:
        lowered = text.lower()
        return any(p.search(lowered) for p in self._suspicious_patterns)

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

        # Keep only a bounded amount of history.
        recent_messages = messages[-self.valves.MAX_HISTORY_MESSAGES :]

        cleaned_messages = []
        latest_user_text = ""

        for msg in recent_messages:
            if not isinstance(msg, dict):
                continue

            role = self._normalize_role(msg.get("role", "user"))

            # Optional: remove tool outputs entirely so they cannot inject instructions.
            if self.valves.DROP_TOOL_MESSAGES and role == "tool":
                continue

            content = msg.get("content")
            text = self._content_to_text(content)

            # Sanitize only text-bearing messages.
            if text:
                sanitized = self._sanitize_text(text)

                # Replace content with a plain text version.
                # This is simple and predictable for a first pass.
                msg = dict(msg)
                msg["role"] = role
                msg["content"] = sanitized

                if role == "user":
                    latest_user_text = sanitized
            else:
                msg = dict(msg)
                msg["role"] = role

            cleaned_messages.append(msg)

        # Block only on the latest user turn, not the full transcript.
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
