# Qwen3.5 Python Client — Single File Quick Start

A simple single-file Python client for your Qwen3.5 server. 

It supports:
- Text chat (with Reasoning/Thinking toggle)
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
"""
Qwen3.5 Gateway — Python Client
Supports: text, images, PDFs, videos | local files + URLs | streaming
"""

import base64
import json
import mimetypes
import requests
from pathlib import Path


class QwenClient:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key  = api_key
        self.model    = "Qwen/Qwen3.5-4B"  # updated from /v1/models on connect

    # ------------------------------------------------------------------
    # Connection + health
    # ------------------------------------------------------------------
    def connect(self) -> bool:
        """
        Check /health and fetch the real model name from /v1/models.
        Returns True if server is reachable.
        """
        try:
            r = requests.get(f"{self.base_url}/health", timeout=10)
            r.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Server unreachable at {self.base_url}: {e}")

        try:
            r = requests.get(f"{self.base_url}/v1/models", timeout=10)
            data = r.json()
            models = data.get("data", [])
            if models:
                self.model = models[0]["id"]
        except Exception:
            pass  # keep default model name

        print(f"[OK] Connected — model: {self.model}")
        return True

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------
    def _encode_file(self, path: str) -> str:
        """Read a local file and return a base64 data URI."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        mime, _ = mimetypes.guess_type(p)
        mime = mime or "application/octet-stream"
        b64  = base64.b64encode(p.read_bytes()).decode()
        return f"data:{mime};base64,{b64}"

    def _prepare_sources(self, files=None, urls=None) -> list:
        sources = []
        for f in (files or []):
            if isinstance(f, str) and f.startswith("data:"):
                sources.append(f)          # already base64
            else:
                sources.append(self._encode_file(str(f)))
        for u in (urls or []):
            sources.append(str(u))         # pass URL as-is; server handles download
        return sources

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------
    def run(
        self,
        prompt: str,
        files=None,
        urls=None,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        sources = self._prepare_sources(files, urls)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if sources:
            # Multimodal path — /v1/agent
            endpoint = "/v1/agent"
            payload  = {
                "prompt":      prompt,
                "files":       sources,
                "stream":      stream,
                "temperature": temperature,
                "max_tokens":  max_tokens,
            }
        else:
            # Text-only path — /v1/chat/completions
            endpoint = "/v1/chat/completions"
            payload  = {
                "model":       self.model,
                "messages":    [{"role": "user", "content": prompt}],
                "stream":      stream,
                "temperature": temperature,
                "max_tokens":  max_tokens,
            }

        try:
            resp = requests.post(
                self.base_url + endpoint,
                headers=headers,
                json=payload,
                stream=stream,
                timeout=300,
            )
        except Exception as e:
            raise ConnectionError(f"Request failed: {e}")

        if not resp.ok:
            raise RuntimeError(f"API error ({resp.status_code}): {resp.text}")

        # ── Streaming ─────────────────────────────────────────
        if stream:
            full = ""
            try:
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if delta:
                            print(delta, end="", flush=True)
                            full += delta
                    except Exception:
                        continue
            except Exception as e:
                print(f"\n[Stream error] {e}")
            print()
            return full

        # ── Non-streaming ─────────────────────────────────────
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError("Server returned invalid JSON")

        return (
            data.get("choices", [{}])[0].get("message", {}).get("content")
            or data.get("error")
            or "No response"
        )

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------
    def chat(self, prompt: str, **kw) -> str:
        return self.run(prompt, **kw)

    def analyze(self, prompt: str, files=None, urls=None, **kw) -> str:
        return self.run(prompt, files=files, urls=urls, **kw)


# ======================================================================
# Example usage — runs all 7 tests when executed directly
# ======================================================================
if __name__ == "__main__":
    # 1. Setup
    BASE_URL = "https://xxxx-xxxx.trycloudflare.com"
    if "xxxx-xxxx" in BASE_URL:
        BASE_URL = input("Paste your TryCloudflare URL: ").strip()
        while not BASE_URL.startswith("http"):
            print("Must start with https://")
            BASE_URL = input("Paste your TryCloudflare URL: ").strip()

    client = QwenClient(base_url=BASE_URL)
    client.connect()  # health check + fetch real model name
    print("=" * 50 + "\n")

    # ── Test 1: text only ────────────────────────────────────
    print("[1/7] Text only")
    print(client.run("What is the capital of France?", temperature=0))
    print("-" * 40)

    # ── Test 2: image URL ────────────────────────────────────
    print("\n[2/7] Image URL")
    print(client.run(
        "Describe this image.",
        urls=["https://raw.githubusercontent.com/Sabir-Ali-Mondal/Colab-Vision-API-Server/main/sample-data/test-img.jpg"],
    ))
    print("-" * 40)

    # ── Test 3: PDF URL ──────────────────────────────────────
    print("\n[3/7] PDF URL")
    print(client.run(
        "Summarise this PDF in 3 bullet points.",
        urls=["https://raw.githubusercontent.com/Sabir-Ali-Mondal/Colab-Vision-API-Server/main/sample-data/report-pdf.pdf"],
    ))
    print("-" * 40)

    # ── Test 4: video URL ────────────────────────────────────
    print("\n[4/7] Video URL")
    print(client.run(
        "Describe what is happening in this video.",
        urls=["https://raw.githubusercontent.com/Sabir-Ali-Mondal/Colab-Vision-API-Server/main/sample-data/sample-10s-360p.mp4"],
    ))
    print("-" * 40)

    # ── Test 5: local files ──────────────────────────────────
    print("\n[5/7] Local files")
    try:
        print(client.run(
            "Analyse both files.",
            files=["test-img.jpg", "report-pdf.pdf"],
        ))
    except FileNotFoundError:
        print("[SKIPPED] test-img.jpg / report-pdf.pdf not found locally.")
    print("-" * 40)

    # ── Test 6: base64 image ─────────────────────────────────
    print("\n[6/7] Base64 image (1×1 pixel)")
    tiny_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    print(client.run("What color is this image?", files=[tiny_png]))
    print("-" * 40)

    # ── Test 7: streaming ────────────────────────────────────
    print("\n[7/7] Streaming text")
    client.run("Write a 3-sentence sci-fi story about a robot.", stream=True)
    print("\n" + "-" * 40)

    print("\n[SUCCESS] All tests done.")
```

---

## Run

```bash
python qwen_client.py
```

*When you run it, it will ask for your URL if you have not hardcoded it in the script, then automatically run through all 7 tests sequentially.*

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
| Image | Yes        | Yes |
| PDF   | Yes        | Yes |
| Video | Yes        | Yes |

---

## Notes

* Local files are converted to base64 automatically.
* URLs are sent directly to the server to download.
* If files are present, the client automatically uses the Multimodal `/v1/agent` endpoint.
* If no files are present, it seamlessly routes to `/v1/chat/completions`.
* To see the internal model thought process, leave `thinking=True`. To get just the final answer quickly, pass `thinking=False`.
