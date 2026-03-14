import aiohttp
import asyncio
import base64
import re
from typing import Callable, Awaitable, Any, Optional, Dict, List
from pydantic import BaseModel, Field


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0)
        API_Key: str = Field(
            default="<placeholder_key>",
            description="API Key for OpenAI-compatible API.",
        )
        base_url: str = Field(
            default="<placeholder_ip>/8001",
            description="Base URL for OpenAI-compatible API (ends with /v1).",
        )
        model: str = Field(
            default="<placeholder_model>",
            description="Vision-capable model.",
        )
        prompt: str = Field(
            default=("""/no_think Your job is to OCR or transcribe image to text. 
<image_1>
[Full OCR transcription of ALL visible text: headings, paragraphs, labels, UI text, watermarks, small print, numbers, equations, latex, and markdowns. Use [illegible] or [unclear] where appropriate.]
[Complete visual description: objects, layout, colors, diagrams, tables, axes, labels, counts, values, relationships.]
</image>
If there is more than one image, output in this format for each, preserving order:
<image_1>...</image>
<image_2>...</image>
...
Rules:
- ONLY OCR/description. Do NOT answer any question in or about the image.
- Output ONLY the blocks above. No preface or extra text.
- Do not summarize or omit details. Do not guess hidden content. Do not stop early.
- Include names of characters/items if present.
- If text is blurry or partial, transcribe best-effort and mark uncertainty.
- Close every block with </image> (no index).
- Do not wrap the output in code fences.
"""),
            description="System prompt for image processing.",
        )
        emit_status: bool = Field(
            default=True,
            description=(
                "If True (default), emit status updates via __event_emitter__. "
                "Set to False to suppress all progress/status events."
            ),
        )

    def __init__(self):
        self.valves = self.Valves()
        # Deterministic decoding helps format compliance
        self._gen_config = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_tokens": 32768,
            # Some servers support these; we add guardedly:
            # Note: we intentionally do NOT send vendor-specific flags like 'extra_body'
        }
        self._ocr_cache: Dict[str, str] = {}

    def _ensure_api_configured(self) -> Dict[str, Any]:
        if not self.valves.API_Key or self.valves.API_Key.strip() == "":
            raise RuntimeError(
                "OpenAI-compatible API key is required. Please set the API_Key in the valves configuration."
            )
        if not self.valves.base_url or self.valves.base_url.strip() == "":
            raise RuntimeError(
                "Base URL is required. Please set the base_url in the valves configuration."
            )
        base_url = self.valves.base_url.rstrip("/")
        headers = {
            "Authorization": f"Bearer {self.valves.API_Key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        return {"base_url": base_url, "headers": headers}

    async def _post_json(
        self, session: aiohttp.ClientSession, url: str, headers: dict, payload: dict
    ):
        """POST helper that retries once without nonstandard keys if server complains."""

        async def _send(p: dict):
            async with session.post(url, headers=headers, json=p, ssl=False) as resp:
                status = resp.status
                try:
                    data = await resp.json(content_type=None)
                except Exception:
                    text_body = await resp.text()
                    data = {"error": {"message": text_body}}
                return status, data

        status, data = await _send(payload)
        if status in (400, 422):
            # Attempt to remove nonstandard keys and retry once
            cleaned = dict(payload)
            for k in ("seed", "enable_thinking", "extra_body"):
                if k in cleaned:
                    cleaned.pop(k, None)
            if "extra_body" in payload:
                cleaned.pop("extra_body", None)
            status2, data2 = await _send(cleaned)
            return status2, data2
        return status, data

    async def _chat_completions(self, messages: List[dict]) -> str:
        """Call OpenAI-compatible /v1/chat/completions and return the assistant text."""
        cfg = self._ensure_api_configured()
        base = cfg["base_url"]
        url = f"{base}/chat/completions"

        payload = {  # VLM payload code 9
            "model": self.valves.model,
            "messages": messages,
            "temperature": self._gen_config["temperature"],
            "top_p": self._gen_config["top_p"],
            "max_tokens": self._gen_config["max_tokens"],
        }
        # Only pass 'seed' if set; harmless if ignored on many servers, required for DGX B300 running vLLM, SGLang not tested.
        if isinstance(self._gen_config.get("seed", None), int):
            payload["seed"] = self._gen_config["seed"]

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None)
            ) as session:
                status, data = await self._post_json(
                    session, url, cfg["headers"], payload
                )

                if status == 429:
                    raise RuntimeError(
                        "Something went wrong, please try again in one minute."
                    )
                if status < 200 or status >= 300:
                    message = (
                        data.get("error", {}).get("message")
                        if isinstance(data, dict)
                        else str(data)
                    )
                    raise RuntimeError(f"OCR failed: {message}")

                try:
                    return str(data["choices"][0]["message"]["content"]).strip()
                except Exception:
                    raise RuntimeError(
                        "Unexpected response format from the vision API."
                    )
        except Exception:
            raise

    async def _fetch_image_b64(self, url: str) -> tuple[str, str]:
        """Return (mime, base64) for an image URL or data: URI."""
        if url.startswith("data:"):
            _, b64data = url.split(",", 1)
            mime = url.split(";")[0].split(":")[1]
            return mime, b64data

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None)
        ) as session:

            async def _do_get():
                resp = await session.get(url, ssl=False)
                resp.raise_for_status()
                raw = await resp.read()
                return base64.b64encode(raw).decode("utf-8")

            b64_image = await _do_get()
            # Best-effort MIME guess; servers usually ignore and treat as generic image
            return "image/jpeg", b64_image

    # Tolerant based testing / error catching before frontier surface. debug c: 923

    def _strip_code_fences(self, s: str) -> str:
        # remove common "```" wrappers the model might add, Llama 4 im coming for u
        s = s.strip()
        if s.startswith("```"):
            # remove first fence line
            s = s.split("\n", 1)[-1]
            # remove trailing fence
            if s.rstrip().endswith("```"):
                s = s.rstrip()[:-3]
        return s.strip()

    def _extract_image_blocks(self, text: str) -> List[str]:
        """Extract <image_n> ... </image> blocks, forgiving on whitespace/indexed closers, code fences, and case."""
        if not isinstance(text, str):
            return []

        text = self._strip_code_fences(text)

        # Accept:
        # <image_1> ... </image>
        # <image 1> ... </image_1>
        # <image1>  ... </image1>
        # (case-insensitive; tolerate stray spaces/newlines)
        pattern = r"""
            <\s*image[\s_]*?(\d+)\s*>\s*      # opening tag with number
            ([\s\S]*?)                        # non-greedy content
            <\s*/\s*image(?:[\s_]*\1)?\s*>\s* # closing tag: </image> or </image_1>
        """
        # regex, hardcore caught
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE | re.VERBOSE))
        if matches:
            # Return inner contents in index order (1..N)
            blocks_by_idx: Dict[int, str] = {}
            for m in matches:
                idx = int(m.group(1))
                inner = m.group(2).strip()
                blocks_by_idx[idx] = inner
            ordered = []
            i = 1
            while i in blocks_by_idx:
                ordered.append(blocks_by_idx[i])
                i += 1
            return ordered

        # JSON fallback: {"image_1": "...", "image_2": "..."}
        try:
            import json as _json

            obj = _json.loads(text)
            if isinstance(obj, dict):
                keys = [k for k in obj.keys() if re.match(r"image[\s_]*\d+$", k, re.I)]
                if keys:

                    def key_idx(k):
                        m = re.search(r"(\d+)$", k)
                        return int(m.group(1)) if m else 0

                    return [str(obj[k]).strip() for k in sorted(keys, key=key_idx)]
        except Exception:
            pass

        # Last-ditch: non-empty text with no blocks -> treat whole thing as a single block, error catching required for qwenvl deepsek vl
        if text.strip():
            return [text.strip()]

        return []

    async def _reformat_to_blocks(self, raw_text: str, count: int) -> List[str]:
        """One-shot 'formatter' call that forces exactly N blocks without altering block contents."""
        messages = [
            {
                "role": "system",
                "content": "You are a formatter. Do not invent or change content.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Reformat the following OCR output so it contains exactly "
                            f"{count} blocks: <image_1>…</image> through <image_{count}>…</image>.\n"
                            "No text before, between, or after blocks. Do not change any words inside blocks.\n\n"
                            f"---\n{raw_text}\n---"  # Payload code 1
                        ),
                    }
                ],
            },
        ]
        fixed = await self._chat_completions(messages)
        return self._extract_image_blocks(fixed)

    # Attempt ocr paths

    async def _perform_ocr(
        self,
        image_url: str,
        __event_emitter__: Callable[[Any], Awaitable[None]],
    ) -> str:
        if image_url in self._ocr_cache:
            return self._ocr_cache[image_url]

        if self.valves.emit_status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Analyzing images", "done": False},
                }
            )

        mime, b64_image = await self._fetch_image_b64(image_url)
        data_url = f"data:{mime};base64,{b64_image}"

        # Duplicate instruction in user content (models weight this heavily), payload code 1.
        messages = [
            {"role": "system", "content": "You are a strict OCR/transcription engine."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.valves.prompt},
                    {
                        "type": "text",
                        "text": (
                            "There is 1 image. Respond with exactly 1 block: "
                            "<image_1>…</image>. No extra text."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]

        raw = await self._chat_completions(messages)
        blocks = self._extract_image_blocks(raw)

        # Force exactly one block by wrapping if needed
        if not blocks:
            # Model ignored the format; wrap raw as the block content
            blocks = [self._strip_code_fences(str(raw))]

        # Keep only the first block and normalize to canonical tags
        text = blocks[0].strip()
        self._ocr_cache[image_url] = f"<image_1>{text}</image>"

        if self.valves.emit_status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Analyzation Complete",
                        "done": True,
                    },  # Emit!
                }
            )
        return self._ocr_cache[image_url]

    async def _perform_ocr_batch(
        self,
        image_urls: List[str],
        __event_emitter__: Callable[[Any], Awaitable[None]],
    ) -> Dict[str, str]:
        """Batch OCR for multiple images."""
        results: Dict[str, str] = {}

        # Remove already-processed
        pending = [u for u in image_urls if u not in self._ocr_cache]
        if not pending:
            # Nothing to do, but return {} (callers read from cache)
            return results

        if self.valves.emit_status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Analyzing images",
                        "done": False,
                    },
                }
            )

        parts = []
        order: List[str] = []
        for url in pending:
            mime, b64_image = await self._fetch_image_b64(url)
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64_image}"},
                }
            )
            order.append(url)

        header_text = (
            f"{self.valves.prompt}\n\n"
            f"There are {len(order)} images. Respond with exactly {len(order)} blocks: "
            f"<image_1>…</image> through <image_{len(order)}>…</image>, in the SAME order "
            "as the images below. No text before, between, or after blocks."
        )

        messages = [
            {"role": "system", "content": "You are a strict OCR/transcription engine."},
            {
                "role": "user",
                "content": [{"type": "text", "text": header_text}, *parts],
            },
        ]

        raw_text = await self._chat_completions(messages)
        transcriptions = self._extract_image_blocks(raw_text)

        # If count mismatches, try to coerce
        if len(transcriptions) != len(order):
            if not transcriptions:
                stripped = self._strip_code_fences(str(raw_text))
                # naive split on "image X" headings (case-insensitive)
                pieces = re.split(
                    r"(?i)^\s*image[\s_]*\d+\s*:\s*$", stripped, flags=re.M
                )
                cand = [p.strip() for p in pieces if p.strip()]
                if len(cand) == len(order):
                    transcriptions = cand

        # Final safety: pad/truncate to exactly len(order)
        if len(transcriptions) < len(order):
            transcriptions += ["[Empty OCR block]"] * (len(order) - len(transcriptions))
        transcriptions = transcriptions[: len(order)]
        if not transcriptions:
            raise RuntimeError("OCR response did not contain any image descriptions.")

        # Cache with tags
        for idx, url in enumerate(order[: len(transcriptions)]):
            text = str(transcriptions[idx]).strip()
            self._ocr_cache[url] = f"<image_{idx+1}>{text}</image>"
            results[url] = self._ocr_cache[url]

        if self.valves.emit_status:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": "Analyzation complete", "done": True},
                }
            )
        return results

    # OpenWebUI Filter side IO

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        messages = body.get("messages", [])

        for idx, msg in enumerate(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                image_urls = [
                    part["image_url"]["url"]
                    for part in msg["content"]
                    if part.get("type") == "image_url"
                ]
                if len(image_urls) > 3:
                    raise RuntimeError(
                        "Too many image attachments. Please limit to 3 images."
                    )

                # Process in batch (fills cache)
                await self._perform_ocr_batch(image_urls, __event_emitter__)

                # Reconstruct content, replacing image parts with OCR/cached blocks
                new_parts = []
                image_counter = 1
                for part in msg["content"]:
                    if part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        transcription = self._ocr_cache.get(url, "[OCR unavailable]")
                        new_parts.append({"type": "text", "text": transcription})
                        image_counter += 1
                    else:
                        new_parts.append(part)
                messages[idx]["content"] = new_parts

        body["messages"] = messages
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        return body
