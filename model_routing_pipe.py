from pydantic import BaseModel, Field
from typing import Iterator, List, Dict, Optional
import requests
import json
import time
import re


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

        # [ADDED] Lower temperature only for mistake-analysis style judging.
        JUDGE_TEMPERATURE: float = 0.0

    def __init__(self):
        self.id = "math_mcq_router"
        self.name = "Math MCQ Router"
        self.valves = self.Valves()

        # [ADDED] Minimal hidden state for the most recently generated MCQ.
        # This is the smallest change that lets the pipe analyze a student's mistake
        # against a stored answer key instead of re-solving from scratch each time.
        self.last_mcq: Optional[Dict] = None

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

    # [ADDED] Small helper for non-streamed completions.
    def _complete(self, payload: dict) -> str:
        response = self._post(payload, stream=False)
        data = response.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )

    # [ADDED] Extract A/B/C/D from a short student response when possible.
    def _extract_answer_letter(self, text: str) -> str:
        if not text:
            return ""

        t = text.strip().upper()
        if t in {"A", "B", "C", "D"}:
            return t

        patterns = [
            r"\b(?:MY ANSWER IS|I THINK|I CHOOSE|I PICK|ANSWER IS)\s*([ABCD])\b",
            r"\b([ABCD])\)",
            r"\bOPTION\s*([ABCD])\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, t)
            if match:
                return match.group(1)

        return ""

    def classify_intent(self, messages: List[Dict]) -> str:
        last_msg = messages[-1]["content"].lower()

        # Fast heuristic
        if any(k in last_msg for k in ["mcq", "multiple choice", "quiz"]):
            return "mcq"

        # [CHANGED] Detect explicit mistake-analysis requests or short answer submissions.
        if any(k in last_msg for k in [
            "why was i wrong",
            "analyze my mistake",
            "what did i do wrong",
            "what mistake did i make",
            "check my answer",
            "was i wrong",
        ]):
            return "mistake_analysis"

        if self._extract_answer_letter(messages[-1]["content"]):
            return "mistake_analysis"

        if any(k in last_msg for k in ["answer is", "i think", "my answer"]):
            return "thinking"

        recent = messages[-6:]
        formatted = "\n".join(
            [f"{m.get('role','').upper()}: {m.get('content','')}" for m in recent]
        )

        # [CHANGED] Added mistake_analysis to classifier output space.
        prompt = f"""You are an intent classifier.

{formatted}

Classify the LAST user message:

- mcq → user wants a multiple choice question
- mistake_analysis → user is giving an answer or asking why they were wrong
- thinking → user is solving or checking an answer in a general tutoring way
- default → anything else

Reply with ONLY one word: mcq, mistake_analysis, thinking, or default."""

        try:
            response = self._post(
                {
                    "model": self.valves.JUDGE_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 8,
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

            if content in ["mcq", "mistake_analysis", "thinking", "default"]:
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

    # [ADDED] Generate MCQ as JSON so the answer key can stay hidden but stored.
    def _generate_mcq(self, user_messages: List[Dict]) -> Dict:
        payload_messages = [{
            "role": "system",
            "content": """Generate ONE math MCQ and reply with ONLY valid JSON.

Return exactly this schema:
{
  "question": "...",
  "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "correct_answer": "A or B or C or D",
  "solution_outline": "brief explanation of why the correct answer is right",
  "common_mistake": "brief description of the most likely student mistake"
}

Rules:
- One correct answer only
- Do not include markdown fences
- Keep the question self-contained
- Keep solution_outline short"""
        }]
        payload_messages.extend(self._trim_messages(user_messages))

        raw = self._complete(
            {
                "model": self.valves.QUESTION_MODEL,
                "messages": payload_messages,
                "temperature": 0.7,
                "max_tokens": 500,
            }
        )

        data = json.loads(raw)
        choices = data.get("choices", {})
        required = ["question", "correct_answer", "solution_outline"]
        if not all(k in data for k in required):
            raise ValueError("MCQ JSON missing required fields")
        if not all(letter in choices for letter in ["A", "B", "C", "D"]):
            raise ValueError("MCQ JSON missing answer choices")

        return data

    # [ADDED] Render the stored MCQ metadata into the student-facing format.
    def _format_mcq(self, mcq: Dict) -> str:
        choices = mcq.get("choices", {})
        return (
            f"**Question:** {mcq.get('question', '')}\n\n"
            f"A) {choices.get('A', '')}\n"
            f"B) {choices.get('B', '')}\n"
            f"C) {choices.get('C', '')}\n"
            f"D) {choices.get('D', '')}\n\n"
            f"Take your time and choose your answer!"
        )

    # [ADDED] Analyze a student's answer against the hidden answer key.
    def _analyze_mistake(self, user_messages: List[Dict]) -> str:
        if not self.last_mcq:
            return "I can analyze your mistake after I generate a question first. Ask me for an MCQ, then send your answer."

        last_user_text = user_messages[-1].get("content", "")
        student_answer = self._extract_answer_letter(last_user_text)

        if not student_answer:
            # Minimal fallback: try to infer answer choice from the recent conversation.
            recent = "\n".join(
                [f"{m.get('role','').upper()}: {m.get('content','')}" for m in user_messages[-6:]]
            )
            infer_prompt = f"""Read the conversation and reply with ONLY one letter: A, B, C, or D.
If the student's answer is unclear, reply with ONLY UNKNOWN.

{recent}"""
            inferred = self._complete(
                {
                    "model": self.valves.JUDGE_MODEL,
                    "messages": [{"role": "user", "content": infer_prompt}],
                    "temperature": self.valves.JUDGE_TEMPERATURE,
                    "max_tokens": 5,
                }
            ).strip().upper()

            if inferred in {"A", "B", "C", "D"}:
                student_answer = inferred

        if student_answer not in {"A", "B", "C", "D"}:
            return "Please send just your answer choice (A, B, C, or D), or ask explicitly for mistake analysis."

        mcq = self.last_mcq
        correct_answer = (mcq.get("correct_answer") or "").strip().upper()
        verdict = "correct" if student_answer == correct_answer else "incorrect"

        analysis_prompt = f"""You are a math tutor doing mistake analysis.

Question: {mcq.get('question', '')}
Choices:
A) {mcq.get('choices', {}).get('A', '')}
B) {mcq.get('choices', {}).get('B', '')}
C) {mcq.get('choices', {}).get('C', '')}
D) {mcq.get('choices', {}).get('D', '')}

Student answer: {student_answer}
Correct answer: {correct_answer}
Verdict: {verdict}
Known correct-solution outline: {mcq.get('solution_outline', '')}
Likely common mistake: {mcq.get('common_mistake', '')}

Write a short response for the student.
Rules:
- If correct, briefly confirm and explain why.
- If incorrect, explain what likely went wrong.
- Be specific, but concise.
- End by asking whether they want another question on the same skill."""

        return self._complete(
            {
                "model": self.valves.THINKING_MODEL,
                "messages": [{"role": "user", "content": analysis_prompt}],
                "temperature": 0.3,
                "max_tokens": 300,
            }
        )

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

        # [CHANGED] Minimal dedicated mistake-analysis route.
        if intent == "mcq":
            try:
                self.last_mcq = self._generate_mcq(user_messages)
                yield self._format_mcq(self.last_mcq)
            except Exception as e:
                yield f"[MCQ generation error: {str(e)}]"
            return

        elif intent == "mistake_analysis":
            try:
                yield self._analyze_mistake(user_messages)
            except Exception as e:
                yield f"[Mistake analysis error: {str(e)}]"
            return

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
