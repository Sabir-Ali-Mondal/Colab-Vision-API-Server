````markdown
# Qwen3.5 Python Client — Quick Start (Multimodal + Chat)

A clean, OpenRouter-style Python client for your Qwen3.5 server.

Supports:
- Text chat
- Images, PDFs, videos
- Local files + URLs
- Streaming responses
- Automatic endpoint routing

---

## Install

```bash
pip install requests
````

---

## The Client

Save as `client.py`:

```python
import base64
import mimetypes
import requests
from pathlib import Path


# ======================================================
# CLIENT
# ======================================================
class QwenClient:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _encode_file(self, path: str) -> str:
        path = Path(path)
        mime, _ = mimetypes.guess_type(path)

        if mime is None:
            mime = "application/octet-stream"

        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        return f"data:{mime};base64,{b64}"

    def _prepare_files(self, files=None, urls=None):
        output = []

        if files:
            for f in files:
                output.append(self._encode_file(f))

        if urls:
            output.extend(urls)

        return output

    def run(
        self,
        prompt: str,
        files=None,
        urls=None,
        task="general",
        stream=False,
        temperature=0.7,
        max_tokens=1024,
    ):
        all_files = self._prepare_files(files, urls)

        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        use_agent = len(all_files) > 0

        if use_agent:
            endpoint = "/v1/agent"
            payload = {
                "prompt": prompt,
                "files": all_files,
                "task": task,
                "stream": stream,
            }
        else:
            endpoint = "/v1/chat/completions"
            payload = {
                "model": "Qwen/Qwen3.5-4B",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }

        response = requests.post(
            self.base_url + endpoint,
            headers=headers,
            json=payload,
            stream=stream,
            timeout=300,
        )

        if not response.ok:
            raise Exception(response.text)

        # STREAM MODE
        if stream:
            full = ""

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode()

                if not line.startswith("data:"):
                    continue

                raw = line[5:].strip()

                if raw == "[DONE]":
                    break

                try:
                    chunk = requests.utils.json.loads(raw)
                    delta = chunk.get("choices", [{}])[0] \
                                 .get("delta", {}) \
                                 .get("content", "")

                    print(delta, end="", flush=True)
                    full += delta

                except Exception:
                    pass

            print()
            return full

        # NORMAL MODE
        data = response.json()

        return (
            data.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            or data.get("error")
            or "No response"
        )
```

---

## Basic Usage

```python
from client import QwenClient

client = QwenClient(
    base_url="https://xxxx-xxxx.trycloudflare.com",
    api_key=""
)

response = client.run(
    prompt="Explain quantum computing simply."
)

print(response)
```

---

## Multimodal Example (Files + URLs)

```python
response = client.run(
    prompt="Analyze all inputs and give a summary.",

    files=[
        "image.jpg",
        "document.pdf",
        "video.mp4"
    ],

    urls=[
        "https://example.com/chart.png"
    ],

    task="reasoning"
)

print(response)
```

---

## Streaming Output (Live Tokens)

```python
client.run(
    prompt="Write a story about space.",
    stream=True
)
```

---

## How It Works

| Input Type | Endpoint Used          |
| ---------- | ---------------------- |
| Text only  | `/v1/chat/completions` |
| With files | `/v1/agent`            |

---

## Task Modes

| Task        | Description                |
| ----------- | -------------------------- |
| `general`   | Normal Q&A, summaries      |
| `coding`    | Code generation, debugging |
| `reasoning` | Complex analysis, math     |

---

## Supported Inputs

| Type   | Local File     | URL |
| ------ | -------------- | --- |
| Image  | jpg, png, gif  | ✓   |
| PDF    | pdf            | ✓   |
| Video  | mp4, webm, mov | ✓   |
| Base64 | auto-generated | ✓   |

---

## Notes

* Local files are automatically converted to base64
* URLs are sent directly (no download needed)
* No plugins required (PDF parsing handled server-side)
* Works with any Qwen3.5 server instance

---

## Run

```bash
python your_script.py
```
