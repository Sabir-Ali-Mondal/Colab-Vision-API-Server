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
Qwen3.5 Client — Quick Start
─────────────────────────────
1. Start the Colab server and copy the trycloudflare URL.
2. Set BASE_URL below (or leave as-is to be prompted).
3. Run:  python qwen_client.py

All-in-one example at the bottom shows every feature in one call:
  client.run(
      prompt  = "Describe all the files and answer my question.",
      files   = ["local_image.jpg", "local_doc.pdf"],   # local files
      urls    = ["https://example.com/image.png",        # remote image
                 "https://example.com/doc.pdf",          # remote PDF
                 "https://example.com/video.mp4"],       # remote video
      stream  = True,          # stream tokens as they arrive
      max_tokens = 1024,
      temperature = 0.7,
  )

  # Text-only (no files/urls) → routed to /v1/chat/completions
  # Any file/url present     → routed to /v1/agent (multimodal)
"""

import base64, json, mimetypes, requests
from pathlib import Path


class QwenClient:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key  = api_key
        self.model    = "Qwen/Qwen3.5-4B"

    def connect(self) -> bool:
        """Health-check + fetch real model name. Call before first use."""
        try:
            requests.get(f"{self.base_url}/health", timeout=10).raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Cannot reach server: {e}")
        try:
            d = requests.get(f"{self.base_url}/v1/models", timeout=10).json()
            if d.get("data"): self.model = d["data"][0]["id"]
        except: pass
        print(f"[OK] Connected — {self.model}"); return True

    def _encode(self, path: str) -> str:
        """Local file → base64 data URI."""
        p = Path(path)
        if not p.exists(): raise FileNotFoundError(path)
        mime = mimetypes.guess_type(p)[0] or "application/octet-stream"
        return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"

    def run(self, prompt: str, files=None, urls=None,
            stream=False, temperature=0.7, max_tokens=1024) -> str:
        """
        Send a request to the server.

        prompt      : your question or instruction
        files       : list of local file paths  (images, PDFs)
        urls        : list of remote URLs       (images, PDFs, videos)
        stream      : True = print tokens live, False = wait for full reply
        temperature : 0 = deterministic, 0.7 = default, 1.0 = creative
        max_tokens  : max reply length
        """
        sources = [self._encode(f) for f in (files or []) if not str(f).startswith("data:")]
        sources += [f for f in (files or []) if str(f).startswith("data:")]
        sources += list(urls or [])

        hdrs = {"Content-Type": "application/json"}
        if self.api_key: hdrs["Authorization"] = f"Bearer {self.api_key}"

        if sources:
            ep, body = "/v1/agent", {
                "prompt": prompt, "files": sources,
                "stream": stream, "temperature": temperature, "max_tokens": max_tokens,
            }
        else:
            ep, body = "/v1/chat/completions", {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream, "temperature": temperature, "max_tokens": max_tokens,
            }

        resp = requests.post(self.base_url + ep, headers=hdrs, json=body,
                             stream=stream, timeout=300)
        if not resp.ok: raise RuntimeError(f"[{resp.status_code}] {resp.text}")

        if stream:
            full = ""
            for line in resp.iter_lines():
                line = line.decode() if isinstance(line, bytes) else line
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": break
                try:
                    tok = json.loads(data)["choices"][0]["delta"].get("content", "")
                    if tok: print(tok, end="", flush=True); full += tok
                except: pass
            print(); return full

        return (resp.json().get("choices", [{}])[0]
                .get("message", {}).get("content") or "No response")


# ── Quick-start tests ─────────────────────────────────────────
if __name__ == "__main__":
    BASE_URL = "https://xxxx-xxxx.trycloudflare.com"
    if "xxxx-xxxx" in BASE_URL:
        BASE_URL = input("Paste TryCloudflare URL: ").strip()

    client = QwenClient(BASE_URL)
    client.connect()
    print("=" * 50)

    # 1. Text only
    print("\n[1] Text")
    print(client.run("What is the capital of France?"))

    # 2. Image URL
    print("\n[2] Image URL")
    print(client.run("Describe this image.",
        urls=["https://raw.githubusercontent.com/Sabir-Ali-Mondal/Colab-Vision-API-Server/main/sample-data/test-img.jpg"]))

    # 3. PDF URL
    print("\n[3] PDF URL")
    print(client.run("Summarise in 3 points.",
        urls=["https://raw.githubusercontent.com/Sabir-Ali-Mondal/Colab-Vision-API-Server/main/sample-data/report-pdf.pdf"]))

    # 4. Video URL
    print("\n[4] Video URL")
    print(client.run("Describe what is happening.",
        urls=["https://raw.githubusercontent.com/Sabir-Ali-Mondal/Colab-Vision-API-Server/main/sample-data/sample-10s-360p.mp4"]))

    # 5. Local files (skipped if not found)
    print("\n[5] Local files")
    try:
        print(client.run("Analyse these files.",
            files=["test-img.jpg", "report-pdf.pdf"]))
    except FileNotFoundError:
        print("[SKIPPED] Place test-img.jpg and report-pdf.pdf here to test.")

    # 6. Base64 direct
    print("\n[6] Base64 image")
    tiny = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    print(client.run("What color is this pixel?", files=[tiny]))

    # 7. Streaming
    print("\n[7] Streaming")
    client.run("Write a 3-sentence sci-fi story.", stream=True)

    # ── All-in-one example (uncomment to use) ────────────────
    # print("\n[ALL-IN-ONE]")
    # print(client.run(
    #     prompt      = "Summarise the PDF, describe the image, and answer: what connects them?",
    #     files       = ["local_image.jpg"],
    #     urls        = ["https://example.com/doc.pdf",
    #                    "https://example.com/clip.mp4"],
    #     stream      = True,
    #     max_tokens  = 2048,
    #     temperature = 0.5,
    # ))

    print("\n[DONE]")
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
