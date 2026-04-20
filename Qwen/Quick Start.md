```markdown
# Qwen3.5 Multimodal Agent — Quick Start

Send images, PDFs, videos, and text to your Qwen3.5 server in one call.

---

## Install

```bash
pip install requests
```

---

## The Code

Save as `agent.py`:

```python
import base64
import requests
import mimetypes
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────
BASE_URL = "https://xxxx-xxxx.trycloudflare.com/v1/agent"
API_KEY  = ""   # leave empty if you did not set one
# ─────────────────────────────────────────────────────────────

def encode_file(path: str) -> str:
    """Convert a local file to base64 data URL."""
    path = Path(path)
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        raise ValueError(f"Cannot detect MIME type for {path}")
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{encoded}"


def ask(
    prompt: str,
    files: list = None,
    urls:  list = None,
    task:  str  = "general"
) -> str:
    """
    prompt : your question or instruction
    files  : list of local file paths  (pdf, jpg, png, mp4, etc.)
    urls   : list of remote URLs       (pdf, jpg, png, mp4, etc.)
    task   : "general" | "coding" | "reasoning"
    """
    payload_files = []

    if files:
        for f in files:
            payload_files.append(encode_file(f))

    if urls:
        payload_files.extend(urls)

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    response = requests.post(
        BASE_URL,
        headers=headers,
        json={
            "prompt": prompt,
            "files":  payload_files,
            "task":   task,
            "stream": False
        },
        timeout=300
    )

    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


# ── EXAMPLES — edit and run ───────────────────────────────────
if __name__ == "__main__":

    # Example 1: plain text question
    print(ask("What is the capital of France?"))

    # Example 2: analyze a local image
    # print(ask("Describe this image.", files=["photo.jpg"]))

    # Example 3: summarize a local PDF
    # print(ask("Summarize this document.", files=["report.pdf"]))

    # Example 4: analyze image from URL
    # print(ask("What is in this image?", urls=["https://example.com/chart.png"]))

    # Example 5: analyze PDF from URL
    # print(ask("Extract key points.", urls=["https://example.com/report.pdf"]))

    # Example 6: mixed — local file + remote URL + question
    # print(ask(
    #     "Compare these two documents and summarize differences.",
    #     files=["local.pdf"],
    #     urls=["https://example.com/other.pdf"],
    #     task="reasoning"
    # ))
```

---

## Run

```bash
python agent.py
```

---

## Task Options

| Task | When to use |
|---|---|
| `general` | Summaries, questions, descriptions |
| `coding` | Code review, debugging, generation |
| `reasoning` | Math, comparisons, multi-step analysis |

---

## Supported File Types

| Type | Local file | Remote URL |
|---|---|---|
| Image | `jpg, png, gif, bmp, tiff` | ✓ |
| PDF | `pdf` | ✓ |
| Video | `mp4, mov, avi, webm, mkv` | ✓ |
| Base64 data URL | `data:image/jpeg;base64,...` | — |
```
