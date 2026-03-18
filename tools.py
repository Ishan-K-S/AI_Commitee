from open_webui.utils.middleware import chat_code_interpreter_handler
from pydantic import BaseModel
from openai import OpenAI
import aiohttp
import asyncio
import json


class Filter:
    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="YOUR_OPENROUTER_KEY"
        )

    async def python_interpreter(self, code: str):
        try:
            payload = {"code": code}

            result = await chat_code_interpreter_handler(payload)

            if isinstance(result, dict):
                stdout = result.get("stdout", "")
                stderr = result.get("stderr", "")
                output = result.get("output", "")

                combined = "\n".join(
                    part for part in [stdout, output, stderr] if part
                )

                return combined or "Code executed successfully with no output."

            return str(result)

        except Exception as e:
            return f"Python interpreter error: {str(e)}"

    async def web_search(self, query: str):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        "https://api.duckduckgo.com/",
                        params={"q": query, "format": "json"}
                ) as response:

                    data = await response.json()

                    if response.status != 200:
                        return f"Search API error {response.status}"

                    if data.get("AbstractText"):
                        return data["AbstractText"]

                    return "No detailed result found."

        except Exception as e:
            return str(e)

    async def filter(self, body: dict, **kwargs):

        messages = body.get("messages", [])

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "python_interpreter",
                    "description": "Execute Python code for analysis or calculations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"}
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for real-time information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        while True:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="nvidia/nemotron-3-nano-30b-a3b:free",
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            message = response.choices[0].message

            if not message.tool_calls:
                body["messages"].append({"role": "assistant", "content": messages.content})
                return body

            tasks = []
            tool_calls = []

            for tool_call in message.tool_calls:
                function_name = tool_call.function.name

                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    tasks.append(asyncio.sleep(0, result="Passed arguments raise JSONDecodeError"))

                tool_calls.append(tool_call)

                try:
                    if function_name == "python_interpreter":
                        tasks.append(self.python_interpreter(arguments["code"]))

                    elif function_name == "web_search":
                        tasks.append(self.web_search(arguments["query"]))

                    else:
                        tasks.append(asyncio.sleep(0, result="Unknown tool"))
                except KeyError as e:
                    tasks.append(asyncio.sleep(0, result=str(e)))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            if len(results) != len(tool_calls):
                return "Tool execution mismatch"

            tool_messages = []
            for tool_call, result in zip(tool_calls, results):
                result = result[:32000]  # desired response length/context size
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    }
                )

            messages += [message] + tool_messages