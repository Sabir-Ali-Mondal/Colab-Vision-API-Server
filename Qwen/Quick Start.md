
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
import json
from pathlib import Path


# ======================================================
# CLIENT
# ======================================================
class QwenClient:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    # -----------------------------
    # Encode local file → base64
    # -----------------------------
    def _encode_file(self, path: str) -> str:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        mime, _ = mimetypes.guess_type(path)
        if mime is None:
            mime = "application/octet-stream"

        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        return f"data:{mime};base64,{b64}"

    # -----------------------------
    # Prepare files (paths + URLs + base64)
    # -----------------------------
    def _prepare_files(self, files=None, urls=None):
        output = []

        if files:
            for f in files:
                # Already base64
                if isinstance(f, str) and f.startswith("data:"):
                    output.append(f)

                # Local file path
                else:
                    output.append(self._encode_file(f))

        if urls:
            for u in urls:
                if not isinstance(u, str):
                    raise ValueError("URL must be a string")
                output.append(u)

        return output

    # -----------------------------
    # MAIN RUN
    # -----------------------------
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

        headers = {
            "Content-Type": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Decide endpoint
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

        try:
            response = requests.post(
                self.base_url + endpoint,
                headers=headers,
                json=payload,
                stream=stream,
                timeout=300,
            )
        except Exception as e:
            raise Exception(f"Connection failed: {e}")

        if not response.ok:
            raise Exception(f"API Error: {response.text}")

        # -----------------------------
        # STREAM MODE
        # -----------------------------
        if stream:
            full = ""

            try:
                for line in response.iter_lines():
                    if not line:
                        continue

                    line = line.decode("utf-8", errors="ignore")

                    if not line.startswith("data:"):
                        continue

                    raw = line[5:].strip()

                    if raw == "[DONE]":
                        break

                    try:
                        chunk = json.loads(raw)
                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )

                        print(delta, end="", flush=True)
                        full += delta

                    except Exception:
                        continue

            except Exception as e:
                print(f"\n[Stream Error] {e}")

            print()
            return full

        # -----------------------------
        # NORMAL MODE
        # -----------------------------
        try:
            data = response.json()
        except Exception:
            raise Exception("Invalid JSON response")

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

    # =====================================================
    # 1. TEXT ONLY
    # =====================================================
    print("\n--- TEXT ONLY ---\n")
    print(client.run("What is the capital of France?"))

    # =====================================================
    # 2. IMAGE (URL)
    # =====================================================
    # print(client.run(
    #     prompt="Describe this image.",
    #     urls=[
    #         "https://raw.githubusercontent.com/Sabir-Ali-Mondal/Colab-Vision-API-Server/main/sample-data/test-img.jpg"
    #     ]
    # ))

    # =====================================================
    # 3. PDF
    # =====================================================
    # print(client.run(
    #     prompt="Summarize this PDF.",
    #     urls=[
    #         "https://raw.githubusercontent.com/Sabir-Ali-Mondal/Colab-Vision-API-Server/main/sample-data/report-pdf.pdf"
    #     ]
    # ))

    # =====================================================
    # 4. VIDEO
    # =====================================================
    # print(client.run(
    #     prompt="Describe this video.",
    #     urls=[
    #         "https://raw.githubusercontent.com/Sabir-Ali-Mondal/Colab-Vision-API-Server/main/sample-data/sample-10s-360p.mp4"
    #     ]
    # ))

    # =====================================================
    # 5. LOCAL FILES
    # =====================================================
    # print(client.run(
    #     prompt="Analyze all local files.",
    #     files=[
    #         "test-img.jpg",
    #         "report-pdf.pdf"
    #     ]
    # ))

    # =====================================================
    # 6. BASE64 DIRECT
    # =====================================================
    # base64_img = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
    # print(client.run(
    #     prompt="Analyze this base64 image.",
    #     files=[base64_img]
    # ))

    # =====================================================
    # 7. STREAMING
    # =====================================================
    # client.run(
    #     prompt="Write a short sci-fi story.",
    #     stream=True
    # )
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
