
# Qwen3.5 Python Client — Single File Quick Start

A simple single-file Python client for your Qwen3.5 server.

It supports:
- Text chat
- Images, PDFs, videos
- Local files and URLs
- Streaming responses
- Automatic routing to the correct endpoint

---

## Install

```bash
pip install requests
```

---

## Single File Code

Save this as `qwen_client.py`:

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
                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )

                    print(delta, end="", flush=True)
                    full += delta

                except Exception:
                    pass

            print()
            return full

        data = response.json()

        return (
            data.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            or data.get("error")
            or "No response"
        )


# ======================================================
# EXAMPLE USAGE
# ======================================================
if __name__ == "__main__":
    client = QwenClient(
        base_url="https://xxxx-xxxx.trycloudflare.com",
        api_key=""
    )

    # Example 1: text only
    print(client.run("What is the capital of France?"))

    # Example 2: local files
    # print(client.run(
    #     prompt="Analyze all inputs and give a summary.",
    #     files=["image.jpg", "document.pdf", "video.mp4"],
    #     task="reasoning"
    # ))

    # Example 3: URLs
    # print(client.run(
    #     prompt="Describe this image.",
    #     urls=["https://example.com/chart.png"]
    # ))

    # Example 4: streaming
    # client.run("Write a short story about space.", stream=True)
```

---

## Run

```bash
python qwen_client.py
```

---

## Task Options

| Task        | Use for                              |
| ----------- | ------------------------------------ |
| `general`   | Normal chat, summaries, descriptions |
| `coding`    | Code generation and debugging        |
| `reasoning` | Math, comparisons, deeper analysis   |

---

## Supported Inputs

| Type  | Local file | URL |
| ----- | ---------- | --- |
| Image | ✓          | ✓   |
| PDF   | ✓          | ✓   |
| Video | ✓          | ✓   |

---

## Notes

* Local files are converted to base64 automatically
* URLs are sent directly
* If files are present, the client uses `/v1/agent`
* If no files are present, it uses `/v1/chat/completions`

```

If you want, I can also :contentReference[oaicite:0]{index=0}.
```
