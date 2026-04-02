from pydantic import BaseModel
from openai import OpenAI
import aiohttp
import asyncio
import json
import tempfile
import os


BASE_URL = "https://openrouter.ai/api"


def _normalize_base_url(url: str):
    return url.rstrip("/") + "/v1"


class Filter:
    class Valves(BaseModel):
        pass

    class UserValves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

        self.client = OpenAI(
            base_url=_normalize_base_url(BASE_URL),
            api_key="YOUR_OPENROUTER_KEY",
        )

        # persistent HTTP session
        self.session = aiohttp.ClientSession()

    async def close(self):
        await self.session.close()

    async def python_interpreter(self, code: str):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as tmp:
                tmp.write(code.encode())
                tmp_path = tmp.name

            process = await asyncio.create_subprocess_exec(
                "python",
                tmp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=10  # prevent infinite loops
                )
            except asyncio.TimeoutError:
                process.kill()
                return "Execution timed out."

            finally:
                os.remove(tmp_path)

            output = stdout.decode().strip()
            error = stderr.decode().strip()

            if error:
                return f"Error:\n{error}"

            return output or "Code executed successfully with no output."

        except Exception as e:
            return f"Python interpreter error: {str(e)}"

    async def web_search(self, query: str):
        try:
            async with self.session.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json"},
            ) as response:

                if response.status != 200:
                    return f"Search API error {response.status}"

                data = await response.json()

                if data.get("AbstractText"):
                    return data["AbstractText"]

                return "No detailed result found."

        except Exception as e:
            return f"Web search error: {str(e)}"

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
                tool_choice="auto",
            )

            message = response.choices[0].message

            if not message.tool_calls:
                body["messages"].append(
                    {
                        "role": "assistant",
                        "content": message.content,
                    }
                )
                return body

            tasks = []
            tool_calls = []

            for tool_call in message.tool_calls:

                function_name = tool_call.function.name

                try:
                    arguments = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    tasks.append(
                        asyncio.sleep(
                            0,
                            result="Passed arguments raised JSONDecodeError",
                        )
                    )
                    tool_calls.append(tool_call)
                    continue

                tool_calls.append(tool_call)

                try:

                    if function_name == "python_interpreter":
                        tasks.append(
                            self.python_interpreter(arguments["code"])
                        )

                    elif function_name == "web_search":
                        tasks.append(
                            self.web_search(arguments["query"])
                        )

                    else:
                        tasks.append(
                            asyncio.sleep(0, result="Unknown tool")
                        )

                except KeyError as e:
                    tasks.append(asyncio.sleep(0, result=str(e)))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            if len(results) != len(tool_calls):
                return "Tool execution mismatch"

            tool_messages = []

            for tool_call, result in zip(tool_calls, results):

                result = str(result)[:32000]

                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

            messages += [message] + tool_messages