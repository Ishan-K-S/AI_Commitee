from pydantic import BaseModel, Field
from typing import Optional, Iterator, Any # added: Any
import requests
import json


class Pipe:
    class Valves(BaseModel):
        GROQ_API_KEY: str = Field(
            default="", description="Your Groq API key (REQUIRED)"
        )
        GROQ_BASE_URL: str = Field(default="https://api.groq.com/openai/v1")

        QUESTION_MODEL: str = Field(
            default="meta-llama/llama-4-scout-17b-16e-instruct",
            description="Model used to generate practice problems",
        )

        THINKING_MODEL: str = Field(
            default="openai/gpt-oss-120b",
            description="Model used for reasoning and explanations",
        )

        JUDGE_MODEL: str = Field(
            default="llama-3.3-70b-versatile",
            description="Model used to classify user intent",
        )

        # added: timeout values
        CONNECT_TIMEOUT: float = Field(
            default=20.0, description='Max seconds to establish connection'
        )

        READ_TIMEOUT: float = Field(
            default=20.0, description='Max seconds to wait for data'
        )

    def __init__(self):
        self.id = "math_mcq_router"
        self.name = "Math MCQ Router"
        self.valves = self.Valves()

    # added: convert message content into text
    def _content_to_text(self, content: Any) -> str:
        """
        Convert OpenWebUI-style message content into plain text that is safe for filtering and intent classification.
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
                    # not a string or dict -> invalid
                    continue

                part_type = part.get("type")

                if part_type in {"text", "input_text", "output_text"}:
                    text = part.get("text", "")
                    if isinstance(text, str) and text.strip():
                        parts.append(text)

                elif part_type in {"image_url", "input_image"}:
                    parts.append("[image]") # placeholder
                    # TODO: add vision capability

                elif isinstance(part.get("text"), str) and part["text"].strip():
                    parts.append(part["text"])

            return "\n".join(parts)

        if isinstance(content, dict):
            if isinstance(content.get("text"), str):
                return content["text"]

            nested = content.get("content")
            if nested is not None:
                return self._content_to_text(nested)

            return ""

        return str(content)
    
    def _normalize_message(self, message: Any) -> Optional[dict]:
        """
        Return a safe message dict or None if the item is unusable.
        Adds a plain-text view under 'text_content' for router logic.
        """
        if not isinstance(message, dict):
            return None

        normalized = dict(message)

        role = normalized.get("role", "user") # if no role, defaults to user
        if not isinstance(role, str) or not role.strip():
            role = "user"

        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"

        normalized["role"] = role
        normalized["text_content"] = self._content_to_text(
            normalized.get("content")
        ).strip()

        return normalized

    def classify_intent(self, messages: list) -> str:
        if not self.valves.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set")

        recent = messages[-4:]

        # added: new formatting
        formatted_lines = []

        for m in recent:
            role = m.get("role", "user").upper()
            text = m.get("text_content", "")
            formatted_lines.append(f"{role}: {text}")

        formatted = "\n".join(formatted_lines)

        judge_prompt = f"""You are an intent classifier.

Here is the recent conversation:
{formatted}

Classify the LAST user message into exactly one category:
- "mcq" → the user wants a new practice problem
- "thinking" → the user is answering, checking, or asking for explanation

Reply with ONLY one word: mcq or thinking"""

        response = requests.post(
            f"{self.valves.GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.valves.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.valves.JUDGE_MODEL,
                "messages": [{"role": "user", "content": judge_prompt}],
                "max_tokens": 5,
                "temperature": 0,
            },
            timeout=(self.valves.CONNECT_TIMEOUT, self.valves.READ_TIMEOUT), # added: timeout
        )

        if response.status_code != 200:
            raise Exception(
                f"Classification failed (HTTP {response.status_code}): {response.text}"
            )

        result = response.json()["choices"][0]["message"]["content"].strip().lower()
        return "mcq" if result == "mcq" else "thinking" # changed: check for exact match rather than a substring

    def pipe(self, body: dict) -> Iterator[str]:
        messages = body.get("messages", [])

        # changed: check for if messages exists
        if not isinstance(messages, list) or not messages:
            yield "No messages received."
            return

        if not self.valves.GROQ_API_KEY:
            yield "ERROR: GROQ_API_KEY is not configured. Please set your API key in valves."
            return
        
        # added: normalize messages
        normalized_messages = []
        for message in messages:
            normalized = self._normalize_message(message)
            if normalized is not None:
                normalized_messages.append(normalized)

        # note: technically not user messages anymore, just non-system messages
        user_messages = [m for m in normalized_messages if m.get("role") != "system"]
        user_messages = [m for m in user_messages if m.get("text_content")]

        if not user_messages:
            yield "ERROR: No valid user messages found."
            return

        try:
            intent = self.classify_intent(user_messages)
        except Exception as e:
            yield f"[Router error during classification: {str(e)}]"
            return

        if intent == "mcq":
            model = self.valves.QUESTION_MODEL
            system_prompt = """You are a math question generator. Generate exactly ONE multiple choice question.

Use this exact format:

**Question:** [Clear math question here]

A) [option]
B) [option]
C) [option]
D) [option]

Rules:
- Cover algebra, arithmetic, geometry, fractions, or word problems
- Exactly one answer must be correct
- Do NOT reveal the answer
- End with: "Take your time and choose your answer!"
"""
        else:
            model = self.valves.THINKING_MODEL
            system_prompt = """You are a math tutor helping with multiple choice questions.

Your role:
- If student gave an answer (A/B/C/D), tell them if correct, then explain
- If they asked for the answer, reveal it and explain why
- If they asked for explanation, show all working clearly
- Be encouraging and patient
"""

        payload_messages = []
        payload_messages.append({"role": "system", "content": system_prompt})

        # changed: payload building
        for msg in user_messages:
            clean_msg = {k: v for k, v in msg.items() if k != "text_content"} # remove helper plaintext version
            payload_messages.append(clean_msg)

        try:
            response = requests.post(
                f"{self.valves.GROQ_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.valves.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": payload_messages,
                    "stream": True,
                    "temperature": 0.2, # changed: temp value
                    "max_tokens": 1024,
                },
                stream=True,
                timeout=(self.valves.CONNECT_TIMEOUT, self.valves.READ_TIMEOUT), # added: timeout
            )

            if response.status_code != 200:
                yield f"[API Error HTTP {response.status_code}]: {response.text[:300]}"
                return

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")

                    if line.startswith("data: "):
                        data = line[6:]

                        if data.strip() == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            choices = chunk.get("choices", [])
                            if choices and len(choices) > 0:
                                delta = choices[0].get("delta", {}).get("content", "")
                                if delta:
                                    yield delta
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            yield f"[Router error calling model: {str(e)}]"
