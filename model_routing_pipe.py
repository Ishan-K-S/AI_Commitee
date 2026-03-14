from pydantic import BaseModel, Field
from typing import Optional, Iterator
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
            default="llama-3.3-70b-versatile",
            # tried to use openai/gpt-oss-120b, but Groq doesn't support it i think
            # I could be wrong though 
            # I used llama cuz i searched up something that groq does support
            # default= "openai/gpt-oss-120b",
            description="Model used for reasoning and explanations (must be available on GROQ_BASE_URL)",
        )

        JUDGE_MODEL: str = Field(
            default="llama-3.3-70b-versatile",
            description="Model used to classify user intent",
        )

    def __init__(self):
        self.id = "math_mcq_router"
        self.name = "Math MCQ Router"
        self.valves = self.Valves()

    @staticmethod
    def _extract_text_content(content) -> str:
        """
        FIX: message content can be a string OR a list of content parts
        (e.g. vision messages). Safely extract plain text in both cases.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
            return " ".join(parts)
        return ""

    def classify_intent(self, messages: list) -> str:
        if not self.valves.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set")

        recent = messages[-4:]

        # run a message through a helper to use only the text parts
        formatted = "\n".join(
            [f"{m['role'].upper()}: {self._extract_text_content(m.get('content', ''))}" for m in recent]
        )

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
        )

        if response.status_code != 200:
            raise Exception(
                f"Classification failed (HTTP {response.status_code}): {response.text}"
            )

        result = response.json()["choices"][0]["message"]["content"].strip().lower()

        # use exact word match instead of substring
        return "mcq" if result == "mcq" else "thinking"

    def pipe(self, body: dict) -> Iterator[str]:
        messages = body.get("messages", [])

        if not messages:
            yield "No messages received."
            return

        if not self.valves.GROQ_API_KEY:
            yield "ERROR: GROQ_API_KEY is not configured. Please set your API key in valves."
            return

        user_messages = [m for m in messages if m.get("role") != "system"]

        # Extract all text available from each message
        # This prevents images from being dropped
        user_messages = [
            m for m in user_messages
            if self._extract_text_content(m.get("content", "")).strip()
        ]

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

        payload_messages = [{"role": "system", "content": system_prompt}]
        payload_messages.extend(user_messages)

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
                    "temperature": 1,
                    "max_tokens": 1024,
                },
                stream=True,
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

                            # check that the chunk actually has something in it
                            if choices:
                                delta = choices[0].get("delta", {}).get("content", "")
                                if delta:
                                    yield delta
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            yield f"[Router error calling model: {str(e)}]"
