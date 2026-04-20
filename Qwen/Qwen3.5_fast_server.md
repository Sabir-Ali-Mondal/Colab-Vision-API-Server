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
    # Encode local file -> base64
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
        thinking=True,
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
                "thinking": thinking,
                "max_tokens": max_tokens,
            }
        else:
            endpoint = "/v1/chat/completions"
            payload = {
                "model": "Qwen/Qwen3.5-4B",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                "chat_template_kwargs": {"enable_thinking": thinking},
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
            raise Exception(f"API Error ({response.status_code}): {response.text}")

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

                        if delta:
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
    
    # 1. URL Setup & Check
    base_url = "https://xxxx-xxxx.trycloudflare.com"
    
    if "xxxx-xxxx" in base_url:
        print("[WARNING] Placeholder URL detected!")
        base_url = input("> Please paste your actual TryCloudflare URL: ").strip()
        while not base_url.startswith("http"):
            print("[ERROR] Invalid URL. It should start with https://")
            base_url = input("> Please paste your actual TryCloudflare URL: ").strip()

    client = QwenClient(base_url=base_url, api_key="")
    print(f"\n[INFO] Connected to: {client.base_url}\n")
    print("="*50)

    # =====================================================
    # 1. TEXT ONLY
    # =====================================================
    print("\n[1/7] --- TEXT ONLY (Thinking Disabled) ---")
    print(client.run("What is the capital of France?", thinking=False))
    print("-" * 40)

    # =====================================================
    # 2. IMAGE (URL)
    # =====================================================
    print("\n[2/7] --- IMAGE URL ---")
    print("Analyzing remote image...")
    print(client.run(
        prompt="Describe this image.",
        urls=[
            "https://raw.githubusercontent.com/Sabir-Ali-Mondal/Colab-Vision-API-Server/main/sample-data/test-img.jpg"
        ]
    ))
    print("-" * 40)

    # =====================================================
    # 3. PDF
    # =====================================================
    print("\n[3/7] --- PDF URL ---")
    print("Extracting and summarizing remote PDF...")
    print(client.run(
        prompt="Summarize this PDF.",
        urls=[
            "https://raw.githubusercontent.com/Sabir-Ali-Mondal/Colab-Vision-API-Server/main/sample-data/report-pdf.pdf"
        ]
    ))
    print("-" * 40)

    # =====================================================
    # 4. VIDEO
    # =====================================================
    print("\n[4/7] --- VIDEO URL ---")
    print("Analyzing remote video file...")
    print(client.run(
        prompt="Describe this video.",
        urls=[
            "https://raw.githubusercontent.com/Sabir-Ali-Mondal/Colab-Vision-API-Server/main/sample-data/sample-10s-360p.mp4"
        ]
    ))
    print("-" * 40)

    # =====================================================
    # 5. LOCAL FILES
    # =====================================================
    print("\n[5/7] --- LOCAL FILES ---")
    try:
        print("Analyzing local files...")
        print(client.run(
            prompt="Analyze all local files.",
            files=[
                "test-img.jpg",
                "report-pdf.pdf"
            ]
        ))
    except FileNotFoundError:
        print("[SKIPPED] You do not have 'test-img.jpg' or 'report-pdf.pdf' saved in this folder.")
    print("-" * 40)

    # =====================================================
    # 6. BASE64 DIRECT
    # =====================================================
    print("\n[6/7] --- BASE64 IMAGE ---")
    # A valid tiny 1x1 pixel PNG so the server does not crash
    valid_base64_img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
    print("Analyzing direct base64 string (1x1 pixel)...")
    print(client.run(
        prompt="What color is this image?",
        files=[valid_base64_img]
    ))
    print("-" * 40)

    # =====================================================
    # 7. STREAMING
    # =====================================================
    print("\n[7/7] --- STREAMING (Thinking Enabled) ---")
    print("Writing story...\n")
    client.run(
        prompt="Write a 3-sentence sci-fi story about a robot.",
        stream=True,
        thinking=True
    )
    print("\n" + "-" * 40)
    print("\n[SUCCESS] All tests completed successfully!")
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
