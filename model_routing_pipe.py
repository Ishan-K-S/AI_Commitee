from pydantic import BaseModel, Field
from typing import Iterator, List, Dict
import requests
import json
import time


class Pipe:
    class Valves(BaseModel):
        GROQ_API_KEY: str = Field(default="", description="Your Groq API key (REQUIRED)")
        GROQ_BASE_URL: str = Field(default="https://api.groq.com/openai/v1")

        # Models
        QUESTION_MODEL: str = Field(default="llama-3.1-8b-instant")
        THINKING_MODEL: str = Field(default="llama-3.1-70b-versatile")
        JUDGE_MODEL: str = Field(default="llama-3.1-8b-instant")
        DEFAULT_MODEL: str = Field(default="llama-3.1-8b-instant")

        TIMEOUT: int = 30
        MAX_RETRIES: int = 3

    def __init__(self):
        self.id = "math_mcq_router"
        self.name = "Math MCQ Router"
        self.valves = self.Valves()

    def _post(self, payload: dict, stream: bool = False):
        url = f"{self.valves.GROQ_BASE_URL}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.valves.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        last_error = None

        for attempt in range(self.valves.MAX_RETRIES):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    stream=stream,
                    timeout=(5, self.valves.TIMEOUT),
                )

                if response.status_code == 200:
                    return response

                if response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue

                raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt == self.valves.MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)

        raise Exception(f"Max retries exceeded: {last_error}")

    def classify_intent(self, messages: List[Dict]) -> str:
        last_msg = messages[-1]["content"].lower()

        # Fast heuristic
        if any(k in last_msg for k in ["mcq", "multiple choice", "quiz"]):
            return "mcq"
        if any(k in last_msg for k in ["answer is", "i think", "my answer"]):
            return "thinking"

        recent = messages[-6:]
        formatted = "\n".join(
            [f"{m.get('role','').upper()}: {m.get('content','')}" for m in recent]
        )

        prompt = f"""You are an intent classifier.

{formatted}

Classify the LAST user message:

- mcq → user wants a multiple choice question
- thinking → user is solving or checking an answer
- default → anything else

Reply with ONLY one word: mcq, thinking, or default."""

        try:
            response = self._post(
                {
                    "model": self.valves.JUDGE_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 5,
                    "temperature": 0,
                }
            )

            data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                .lower()
            )

            content = content.replace(".", "").split()[0]

            if content in ["mcq", "thinking", "default"]:
                return content

        except Exception:
            pass

        return "default"

    def _trim_messages(self, messages: List[Dict], max_msgs: int = 8):
        return messages[-max_msgs:]

    def _extract_content(self, choice: dict) -> str:
        if not choice:
            return ""

        if "delta" in choice and isinstance(choice["delta"], dict):
            return choice["delta"].get("content", "")

        if "message" in choice and isinstance(choice["message"], dict):
            return choice["message"].get("content", "")

        return ""

    def pipe(self, body: dict) -> Iterator[str]:
        messages = body.get("messages", [])

        if not messages:
            yield "No messages received."
            return

        if not self.valves.GROQ_API_KEY.strip():
            yield "ERROR: GROQ_API_KEY is not configured."
            return

        # Clean messages
        user_messages = []
        for m in messages:
            content = m.get("content") or ""
            if content.strip():
                user_messages.append({
                    "role": m.get("role", ""),
                    "content": content,
                })

        if not user_messages:
            yield "ERROR: No valid messages."
            return

        # Intent detection
        try:
            intent = self.classify_intent(user_messages)
        except Exception as e:
            yield f"[Classification error: {str(e)}]"
            return

        # Routing
        if intent == "mcq":
            model = self.valves.QUESTION_MODEL
            system_prompt = """Generate ONE math MCQ.

Format:

**Question:** ...

A) ...
B) ...
C) ...
D) ...

- One correct answer
- Do NOT reveal answer
- End with: Take your time and choose your answer!
"""
        elif intent == "thinking":
            model = self.valves.THINKING_MODEL
            system_prompt = """You are a math tutor.

- If user gave A/B/C/D → check + explain
- If asked for answer → reveal + explain
- Be clear and encouraging
- End by asking if they understand
"""
        else:
            model = self.valves.DEFAULT_MODEL
            system_prompt = """Handle casual or unclear messages.

- Casual → respond + ask if they want a problem
- Off-topic → redirect to math
- Unclear → ask for clarification
"""

        payload_messages = [{"role": "system", "content": system_prompt}]
        payload_messages.extend(self._trim_messages(user_messages))

        try:
            response = self._post(
                {
                    "model": model,
                    "messages": payload_messages,
                    "stream": True,
                    "temperature": 0.7,
                    "max_tokens": 1024,
                },
                stream=True,
            )

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue

                try:
                    line = raw_line.decode("utf-8").strip()
                except Exception:
                    continue

                # Flexible parsing
                if "data:" not in line:
                    continue

                data = line.split("data:", 1)[-1].strip()

                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # Safe choices handling
                choices = chunk.get("choices")
                if not choices or not isinstance(choices, list):
                    continue

                choice = choices[0]
                content = self._extract_content(choice)

                if content:
                    yield content

        except Exception as e:
            yield f"[Streaming error: {str(e)}]"
